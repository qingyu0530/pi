//! Provider 抽象：一次模型调用怎么被“执行”。
//!
//! 前面的 Model 描述“模型是什么”，Context 描述“一次调用要发送什么”。
//! Provider 负责把后者变成一串流式事件。真实 Provider 会走网络（SSE），
//! FauxProvider 只是在内存里造一串事件，方便测试和教学。
//!
//! C++ 对照：
//!   Provider     ≈ 抽象基类（接口）
//!   FauxProvider ≈ 它的一个具体子类

use std::cell::RefCell;
use std::collections::VecDeque;

use serde_json::{Map, Value};

use crate::content::{AssistantContent, TextContent, ToolCall};
use crate::context::Context;
use crate::event::AssistantMessageEvent;
use crate::message::{
    AssistantMessage, AssistantRole, ConversationMessage, StopReason, Usage, UsageCost,
    UserMessageContent,
};
use crate::model::Model;

/// 运行时单元：持有模型目录，并把一次请求转换成流式事件。
/// 把一次对话请求真正变成 模型回复的那个执行者
/*
Agent 拿到用户消息
  └─> 组成 Context
        └─> provider.stream(model, context)   // 这里可以是 Faux，也可以是真实 OpenAI
              └─> 逐个 event 收集，组出 AssistantMessage
                    └─> 若是 ToolCall，执行工具，回填 ToolResultMessage
                          └─> 再组新一轮 Context，再调 provider

*/
pub trait Provider {
    /// Provider 的唯一 id，例如 "faux"。
    fn id(&self) -> &str;

    /// 展示用名称。
    fn name(&self) -> &str;

    /// 当前已知的模型。约定不抛错；返回空表示“没有模型”。
    fn get_models(&self) -> &[Model];

    /// 把一次请求上下文转换成流式事件序列。
    ///
    /// 返回的是 Box<dyn Iterator<...>>——一个“被装箱的泛型迭代器对象”。
    /// 调用方只需 next() 逐个取出事件，不关心底层是 Vec 还是别的来源。
    /// C++ 对照：类似一个返回生成器/范围的虚函数。
    fn stream(
        &self,
        model: &Model,
        context: &Context,
    ) -> Box<dyn Iterator<Item = AssistantMessageEvent>>;
}

/// FauxProvider 脚本里的一段预设回复。
///
/// 没有脚本时，FauxProvider 回退为“回显最后一条用户文本”。
#[derive(Clone, Debug, PartialEq)]
pub enum FauxResponse {
    /// 直接回一段文本。
    Text(String),
    /// 发起一次工具调用。
    ToolCall {
        id: String,
        name: String,
        arguments: Map<String, Value>,
    },
}

impl FauxResponse {
    /// 便捷构造一次工具调用。
    #[must_use]
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: Map<String, Value>,
    ) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }
}

/// 仅用于测试/学习的 Provider 实现。
///
/// 它不访问网络，保存一份模型清单，并按脚本依次返回回复；
/// 脚本用完后回退为“回显最后一条用户文本”。
// 模拟 Provider。
// 真实 Provider（OpenAI、Anthropic 等）需要：网络、API Key、花钱、结果不确定。
// 开发和学习时不可能每次都真调。
// 所以造一个「假」的实现：

// 不联网、不要 Key、不花钱
// 结果完全可控（脚本化）
// 它和真实 Provider 实现同一个 Provider trait
// C++ 对照：Provider 是抽象基类，
// 真实 Provider 和 FauxProvider 是它的两个子类。
// 测试时用假的子类替换真的，就是「依赖注入 / 打桩（stub/mock）」。
#[derive(Clone, Debug, Default)]
pub struct FauxProvider {
    models: Vec<Model>,
    /// 待发送的脚本回复。`stream` 只借用 `&self`，
    /// 但需要推进队列，所以用 `RefCell` 做内部可变性。
    /// C++ 对照：类似一个 `mutable` 成员。
    script: RefCell<VecDeque<FauxResponse>>,
}

impl FauxProvider {
    #[must_use]
    pub fn new(models: Vec<Model>) -> Self {
        Self {
            models,
            script: RefCell::new(VecDeque::new()),
        }
    }

    /// 用一段脚本创建 Provider，按顺序消费。
    #[must_use]
    pub fn with_script(models: Vec<Model>, script: Vec<FauxResponse>) -> Self {
        Self {
            models,
            script: RefCell::new(script.into()),
        }
    }

    /// 追加一段脚本回复到队尾。
    pub fn push_response(&self, response: FauxResponse) {
        self.script.borrow_mut().push_back(response);
    }
}

impl Provider for FauxProvider {
    fn id(&self) -> &str {
        "faux"
    }

    fn name(&self) -> &str {
        "Faux Provider"
    }

    fn get_models(&self) -> &[Model] {
        &self.models
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
    ) -> Box<dyn Iterator<Item = AssistantMessageEvent>> {
        let scripted = self.script.borrow_mut().pop_front();
        let events = match scripted {
            Some(FauxResponse::Text(text)) => text_events(model, text),
            Some(FauxResponse::ToolCall {
                id,
                name,
                arguments,
            }) => tool_call_events(model, id, name, arguments),
            // 脚本用完后回退：用最后一条 user 文本来生成回复。
            None => {
                let reply = last_user_text(context).unwrap_or_else(|| "Hello from Faux".to_owned());
                text_events(model, reply)
            }
        };
        Box::new(events.into_iter())
    }
}

/// 构造一段文本回复的事件序列：start -> text_start -> text_delta -> text_end -> done。
fn text_events(model: &Model, reply: String) -> Vec<AssistantMessageEvent> {
    let content = vec![AssistantContent::Text(TextContent {
        text: reply.clone(),
        text_signature: None,
    })];
    // 中间事件里的 partial 是“还没结束”的助手消息，所以 stop_reason 用 Pending。
    let partial = assistant_message(model, content.clone(), StopReason::Pending);
    vec![
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
        AssistantMessageEvent::TextStart {
            content_index: 0,
            partial: partial.clone(),
        },
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: reply.clone(),
            partial: partial.clone(),
        },
        AssistantMessageEvent::TextEnd {
            content_index: 0,
            content: reply,
            partial: partial.clone(),
        },
        AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: assistant_message(model, content, StopReason::Stop),
        },
    ]
}

/// 构造一段工具调用的事件序列：start -> toolcall_start -> toolcall_delta -> toolcall_end -> done。
fn tool_call_events(
    model: &Model,
    id: String,
    name: String,
    arguments: Map<String, Value>,
) -> Vec<AssistantMessageEvent> {
    let tool_call = ToolCall {
        id,
        name,
        arguments,
        thought_signature: None,
        namespace: None,
    };
    let content = vec![AssistantContent::ToolCall(tool_call.clone())];
    let partial = assistant_message(model, content.clone(), StopReason::Pending);
    // 工具参数在流式协议里是一段 JSON 文本。
    let delta = serde_json::to_string(&tool_call.arguments).unwrap_or_default();
    vec![
        AssistantMessageEvent::Start {
            partial: partial.clone(),
        },
        AssistantMessageEvent::ToolcallStart {
            content_index: 0,
            partial: partial.clone(),
        },
        AssistantMessageEvent::ToolcallDelta {
            content_index: 0,
            delta,
            partial: partial.clone(),
        },
        AssistantMessageEvent::ToolcallEnd {
            content_index: 0,
            tool_call,
            partial: partial.clone(),
        },
        AssistantMessageEvent::Done {
            reason: StopReason::ToolUse,
            message: assistant_message(model, content, StopReason::ToolUse),
        },
    ]
}

/// 取上下文中最后一条用户文本，作为回退回复来源。
fn last_user_text(context: &Context) -> Option<String> {
    for message in context.messages.iter().rev() {
        if let ConversationMessage::User(user) = message {
            if let UserMessageContent::Text(text) = &user.content {
                return Some(text.clone());
            }
        }
    }
    None
}

/// 构造一条助手消息。
fn assistant_message(
    model: &Model,
    content: Vec<AssistantContent>,
    stop_reason: StopReason,
) -> AssistantMessage {
    AssistantMessage {
        role: AssistantRole::Assistant,
        content,
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 0,
            cost: UsageCost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        },
        stop_reason,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}
