//! Provider 抽象：一次模型调用怎么被“执行”。
//!
//! 前面的 Model 描述“模型是什么”，Context 描述“一次调用要发送什么”。
//! Provider 负责把后者变成一串流式事件。真实 Provider 会走网络（SSE），
//! FauxProvider 只是在内存里造一串事件，方便测试和教学。
//!
//! C++ 对照：
//!   Provider     ≈ 抽象基类（接口）
//!   FauxProvider ≈ 它的一个具体子类

use crate::content::AssistantContent;
use crate::content::TextContent;
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

/// 仅用于测试/学习的 Provider 实现。
///
/// 它不访问网络，保存一份模型清单，并对任何请求返回一段文本回复，
/// 以 start -> text_start -> text_delta -> text_end -> done 的流式事件输出。
#[derive(Clone, Debug, Default)]
pub struct FauxProvider {
    models: Vec<Model>,
}

impl FauxProvider {
    #[must_use]
    pub fn new(models: Vec<Model>) -> Self {
        Self { models }
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
        // 用最后一条 user 文本来生成回复；没有就回默认文本。
        let reply = last_user_text(context).unwrap_or_else(|| "Hello from Faux".to_owned());
        // 中间事件里的 partial 是“还没结束”的助手消息，所以 stop_reason 用 Pending。
        let partial = text_assistant(model, reply.clone(), StopReason::Pending);
        let events = vec![
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
                content: reply.clone(),
                partial: partial.clone(),
            },
            AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message: text_assistant(model, reply, StopReason::Stop),
            },
        ];
        Box::new(events.into_iter())
    }
}

/// 取上下文中最后一条用户文本，作为回复来源。
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

/// 构造一条只含单个文本块的 AssistantMessage。
fn text_assistant(model: &Model, text: String, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        role: AssistantRole::Assistant,
        content: vec![AssistantContent::Text(TextContent {
            text,
            text_signature: None,
        })],
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
