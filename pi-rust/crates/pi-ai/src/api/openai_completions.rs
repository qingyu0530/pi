//! OpenAI-compatible Chat Completions 协议（请求方向）。
//!
//! 这一层只关心「和线上 API 交换的 JSON 长什么样」：
//! 把 pi 的 `Model` + `Context` 翻译成 `POST /chat/completions` 的请求体。
//! 它不负责网络，也不负责解析流式响应（后续步骤）。
//!
//! C++ 对照：这里的 `Chat*` 结构体相当于只用来序列化的 DTO（数据传输对象），
//! 字段名必须和线上 JSON 完全一致，否则服务端读不懂。

use serde::Serialize;
use serde_json::Value;

use crate::content::{AssistantContent, ToolResultContent, UserContent};
use crate::context::{Context, Tool};
use crate::message::{
    AssistantMessage, ConversationMessage, ToolResultMessage, UserMessage, UserMessageContent,
};
use crate::model::Model;

/// `POST /chat/completions` 的请求体。
///
/// 只包含当前需要的最小字段。OpenAI 还有 top_p、frequency_penalty、
/// logprobs 等可选参数，按需再加。
#[derive(Clone, Debug, Serialize)]
pub struct ChatRequest {
    /// 要使用的模型 id，例如 "gpt-4o-mini"。服务端据此选择权重。
    pub model: String,
    /// 对话消息，按时间顺序排列。
    pub messages: Vec<ChatMessage>,
    /// 是否流式返回。`true` 时服务端按 SSE 一行行推 chunk。
    pub stream: bool,
    /// 流式模式的额外选项：让服务端在最后补一个带 token 用量的 chunk。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// 本次最多生成多少 token（OpenAI 新字段名）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    /// 采样温度，越大输出越随机。
    /// 控制模型输出有多随机
    /// 模型每一步不是直接输出一个词，而是先算出所有候选词的概率分布。温度 T 在归一化成概率之前缩放这些分数（logits）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// 本次可用的工具定义。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,
}

/// 流式请求的选项。
#[derive(Clone, Copy, Debug, Serialize)]
pub struct StreamOptions {
    /// 让服务端在流末尾发送一个包含 token 用量的 chunk。
    /// 不加这个字段时，流式响应通常不会返回 usage。
    pub include_usage: bool,
}

/// 一条 OpenAI chat 消息，用 `role` 字段区分四种形状。
///
/// C++ 对照：`#[serde(tag = "role")]` 相当于序列化时先写 role，
/// 再写变体自己的字段；不同 role 的字段集合不同，靠枚举保证类型安全。
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "role")]
pub enum ChatMessage {
    /// 系统提示，整个对话最前面一条。
    #[serde(rename = "system")]
    System { content: String },

    /// 用户输入：纯文本，或文本 + 图片的内容块数组。
    #[serde(rename = "user")]
    User { content: ChatUserContent },

    /// 模型之前说过的话。`content` 可以是 `null`；
    /// 只要它发起过工具调用，就必须用 `tool_calls` 带上。
    #[serde(rename = "assistant")]
    Assistant {
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ChatToolCall>>,
    },

    /// 工具执行结果，用 `tool_call_id` 指回它回应的是哪次调用。
    #[serde(rename = "tool")]
    Tool {
        tool_call_id: String,
        content: String,
    },
}

/// user 消息的 `content`：纯文本字符串，或者内容块数组。
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum ChatUserContent {
    Text(String),
    Parts(Vec<ChatUserPart>),
}

/// user 内容块数组里的一项。
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum ChatUserPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ChatImageUrl },
}

/// 图片以 data URL 形式内嵌：`data:<mime>;base64,<data>`。
#[derive(Clone, Debug, Serialize)]
pub struct ChatImageUrl {
    pub url: String,
}

/// assistant 发起的一次工具调用。
///
/// 注意 `arguments` 是 **JSON 字符串**，不是 JSON 对象：
/// 协议要求把参数对象再编码成字符串，例如 `"{\"path\":\"a.txt\"}"`。
#[derive(Clone, Debug, Serialize)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: ToolCallKind,
    pub function: ChatFunctionCall,
}

/// 工具调用类型。当前只有普通函数调用。
#[derive(Clone, Copy, Debug, Serialize)]
pub enum ToolCallKind {
    #[serde(rename = "function")]
    Function,
}

/// 工具调用的函数名与参数字符串。
#[derive(Clone, Debug, Serialize)]
pub struct ChatFunctionCall {
    pub name: String,
    /// 参数的 JSON 文本，例如 "{\"path\":\"a.txt\"}"。
    pub arguments: String,
}

/// 发给模型的工具定义。
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum ChatTool {
    #[serde(rename = "function")]
    Function { function: ChatFunctionDef },
}

/// 函数工具的具体定义。
#[derive(Clone, Debug, Serialize)]
pub struct ChatFunctionDef {
    pub name: String,
    pub description: String,
    /// 参数的 JSON Schema，服务端据此约束模型输出。
    pub parameters: Value,
    /// 是否要求模型严格遵守 schema；`None` 时不发送该字段。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// 把 pi 的模型和上下文翻译成 OpenAI 请求体。
#[must_use]
pub fn build_request(model: &Model, context: &Context) -> ChatRequest {
    ChatRequest {
        model: model.id.clone(),
        messages: convert_messages(context),
        stream: true,
        stream_options: Some(StreamOptions {
            include_usage: true,
        }),
        max_completion_tokens: Some(model.max_tokens),
        temperature: None,
        tools: context
            .tools
            .as_ref()
            .map(|tools| tools.iter().map(convert_tool).collect()),
    }
}

/// 把对话上下文翻译成 OpenAI 消息数组。
///
/// 顺序很关键：system 在最前；`assistant` 的工具调用后面必须紧跟
/// 对应的 `tool` 消息，否则服务端会拒绝。
#[must_use]
pub fn convert_messages(context: &Context) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    if let Some(system_prompt) = &context.system_prompt {
        messages.push(ChatMessage::System {
            content: system_prompt.clone(),
        });
    }
    for message in &context.messages {
        match message {
            ConversationMessage::User(user) => messages.push(convert_user(user)),
            ConversationMessage::Assistant(assistant) => {
                // 既无 content 又无 tool_calls 的助手消息会被部分服务端拒绝，跳过。
                if let Some(message) = convert_assistant(assistant) {
                    messages.push(message);
                }
            }
            ConversationMessage::ToolResult(result) => messages.push(convert_tool_result(result)),
        }
    }
    messages
}

/// user 消息：文本保持文本，图片转成 data URL。
fn convert_user(message: &UserMessage) -> ChatMessage {
    let content = match &message.content {
        UserMessageContent::Text(text) => ChatUserContent::Text(text.clone()),
        UserMessageContent::Blocks(blocks) => ChatUserContent::Parts(
            blocks
                .iter()
                .map(|block| match block {
                    UserContent::Text(text) => ChatUserPart::Text {
                        text: text.text.clone(),
                    },
                    UserContent::Image(image) => ChatUserPart::ImageUrl {
                        image_url: ChatImageUrl {
                            url: format!("data:{};base64,{}", image.mime_type, image.data),
                        },
                    },
                })
                .collect(),
        ),
    };
    ChatMessage::User { content }
}

/// assistant 消息：文本拼成一个字符串，工具调用编码成 `tool_calls`。
///
/// 对话历史里通常有文本，而标准 Chat Completions 的 `content` 是字符串，
/// 所以多个文本块直接拼接。thinking 块不回传（标准协议不接收）。
fn convert_assistant(message: &AssistantMessage) -> Option<ChatMessage> {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for block in &message.content {
        match block {
            AssistantContent::Text(content) => text.push_str(&content.text),
            AssistantContent::ToolCall(call) => tool_calls.push(ChatToolCall {
                id: call.id.clone(),
                kind: ToolCallKind::Function,
                function: ChatFunctionCall {
                    name: call.name.clone(),
                    // 参数对象 -> JSON 字符串。
                    arguments: serde_json::to_string(&Value::Object(call.arguments.clone()))
                        .unwrap_or_else(|_| "{}".to_owned()),
                },
            }),
            AssistantContent::Thinking(_) => {}
        }
    }
    if text.is_empty() && tool_calls.is_empty() {
        return None;
    }
    Some(ChatMessage::Assistant {
        content: (!text.is_empty()).then_some(text),
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
    })
}

/// 工具结果消息：文本块用换行拼接；只有图片时给服务端一个占位文本。
fn convert_tool_result(message: &ToolResultMessage) -> ChatMessage {
    let text = message
        .content
        .iter()
        .filter_map(|block| match block {
            ToolResultContent::Text(content) => Some(content.text.as_str()),
            ToolResultContent::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    ChatMessage::Tool {
        tool_call_id: message.tool_call_id.clone(),
        content: if text.is_empty() {
            "(no tool output)".to_owned()
        } else {
            text
        },
    }
}

/// 把 pi 的工具定义翻译成 OpenAI 的 function 工具。
fn convert_tool(tool: &Tool) -> ChatTool {
    ChatTool::Function {
        function: ChatFunctionDef {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
            strict: None,
        },
    }
}
