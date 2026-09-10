//! OpenAI-compatible Chat Completions 协议（请求方向）。
//!
//! 这一层只关心「和线上 API 交换的 JSON 长什么样」：
//! 把 pi 的 `Model` + `Context` 翻译成 `POST /chat/completions` 的请求体。
//! 它不负责网络，也不负责解析流式响应（后续步骤）。
//!
//! C++ 对照：这里的 `Chat*` 结构体相当于只用来序列化的 DTO（数据传输对象），
//! 字段名必须和线上 JSON 完全一致，否则服务端读不懂。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::content::{AssistantContent, ToolResultContent, UserContent};
use crate::context::{Context, Tool};
use crate::message::{
    AssistantMessage, ConversationMessage, StopReason, ToolResultMessage, Usage, UsageCost,
    UserMessage, UserMessageContent,
};
use crate::model::{Model, calculate_cost};

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

// -----------------------------------------------------------------------------
// 响应方向：流式 chunk
// -----------------------------------------------------------------------------

/// 流式返回的单个 chunk，对应 SSE 的一行 `data: {...}`。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChatCompletionChunk {
    /// 本次补全的唯一 id，整个流里所有 chunk 相同。
    #[serde(default)]
    pub id: Option<String>,
    /// 实际服务模型；网关可能路由到与请求不同的模型。
    #[serde(default)]
    pub model: Option<String>,
    /// 候选回复数组，`n = 1` 时只有一个。
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    /// token 用量，仅在带 `stream_options.include_usage` 时出现在最后一个 chunk。
    #[serde(default)]
    pub usage: Option<ChunkUsage>,
}

/// 一个候选回复。流式增量都在 `delta` 里。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChunkChoice {
    /// 候选序号（`n > 1` 时区分）。
    #[serde(default)]
    pub index: u32,
    /// 本 chunk 的增量。流结束时可能没有 delta。
    #[serde(default)]
    pub delta: Option<ChunkDelta>,
    /// 结束原因：流中间为 `null`，最后一个 chunk 才有值。
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// 少数厂商把用量放在 choice 里（非标准），做兜底。
    #[serde(default)]
    pub usage: Option<ChunkUsage>,
}

/// 一个 chunk 的增量内容，字段大多是可选的。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChunkDelta {
    /// 角色，通常只在第一个 chunk 出现。
    #[serde(default)]
    pub role: Option<String>,
    /// 正文文本片段，需要累加。
    #[serde(default)]
    pub content: Option<String>,
    /// 推理文本片段（llama.cpp 等）。
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// 推理文本片段（其它 OpenAI-compatible 端点）。
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub reasoning_text: Option<String>,
    /// 工具调用片段，按 `index` 分组累加。
    #[serde(default)]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

/// 工具调用片段。第一块带 `id` 和 `function.name`，
/// 后续块只带 `function.arguments` 字符串片段。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChunkToolCall {
    /// 标识这是第几个工具调用（同一次回复可能调用多个）。
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub function: Option<ChunkFunction>,
}

/// 工具调用的函数名与参数片段。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChunkFunction {
    /// 只在第一块出现。
    #[serde(default)]
    pub name: Option<String>,
    /// 参数 JSON 的字符串片段，需要拼接后再解析。
    #[serde(default)]
    pub arguments: Option<String>,
}

/// 一个 chunk 携带的 token 用量（原始字段）。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChunkUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// DeepSeek 系把缓存读放在顶层。
    #[serde(default)]
    pub prompt_cache_hit_tokens: Option<u64>,
    /// Kimi 等把缓存读放在顶层。
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

/// `prompt_tokens_details` 里的细节。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct PromptTokensDetails {
    /// 缓存读（命中）token。
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    /// 缓存写 token（OpenAI 不发，OpenRouter 系会发）。
    #[serde(default)]
    pub cache_write_tokens: Option<u64>,
}

/// `completion_tokens_details` 里的细节。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct CompletionTokensDetails {
    /// 其中用于推理的 token。
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
}

/// 把 OpenAI 的 `finish_reason` 映射成 pi 的 `StopReason`。
///
/// 返回第二个值是出错时的错误信息（未知原因或 `content_filter`）。
#[must_use]
pub fn map_stop_reason(reason: Option<&str>) -> (StopReason, Option<String>) {
    match reason {
        None => (StopReason::Stop, None),
        Some("stop" | "end") => (StopReason::Stop, None),
        Some("length") => (StopReason::Length, None),
        Some("function_call" | "tool_calls") => (StopReason::ToolUse, None),
        Some("content_filter") => (
            StopReason::Error,
            Some("Provider finish_reason: content_filter".to_owned()),
        ),
        Some("network_error") => (
            StopReason::Error,
            Some("Provider finish_reason: network_error".to_owned()),
        ),
        Some(other) => (
            StopReason::Error,
            Some(format!("Provider finish_reason: {other}")),
        ),
    }
}

/// 把原始 chunk 用量换算成 pi 的 `Usage`，并按模型价格表算出费用。
///
/// 规则（对齐原版）：
/// - `cache_read`：`prompt_tokens_details.cached_tokens`，否则 `prompt_cache_hit_tokens`，否则 `cached_tokens`。
/// - `cache_write`：`prompt_tokens_details.cache_write_tokens`。
/// - `input = prompt_tokens - cache_read - cache_write`（缓存单列，不重复计）。
/// - `output = completion_tokens`（已包含 reasoning）。
/// - `total = input + output + cache_read + cache_write`。
#[must_use]
pub fn parse_usage(raw: &ChunkUsage, model: &Model) -> Usage {
    let cache_read = raw
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
        .or(raw.prompt_cache_hit_tokens)
        .or(raw.cached_tokens)
        .unwrap_or(0);
    let cache_write = raw
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cache_write_tokens)
        .unwrap_or(0);
    let output = raw.completion_tokens;
    let input = raw
        .prompt_tokens
        .saturating_sub(cache_read)
        .saturating_sub(cache_write);
    let total_tokens = input + output + cache_read + cache_write;
    let mut usage = Usage {
        input,
        output,
        cache_read,
        cache_write,
        cache_write_1h: None,
        reasoning: raw
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens),
        total_tokens,
        cost: UsageCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
    };
    usage.cost = calculate_cost(model, &usage);
    usage
}
