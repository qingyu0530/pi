//! OpenAI-compatible Chat Completions 协议（请求方向）。
//!
//! 这一层只关心「和线上 API 交换的 JSON 长什么样」：
//! 把 pi 的 `Model` + `Context` 翻译成 `POST /chat/completions` 的请求体。
//! 它不负责网络，也不负责解析流式响应（后续步骤）。
//!
//! C++ 对照：这里的 `Chat*` 结构体相当于只用来序列化的 DTO（数据传输对象），
//! 字段名必须和线上 JSON 完全一致，否则服务端读不懂。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::api::http::{HttpError, HttpRequest, HttpTransport};
use crate::api::openai_compat::{
    ResolvedOpenAICompletionsCompat, resolve_openai_completions_compat,
};
use crate::compat::{MaxTokensField, OpenRouterRouting, VercelGatewayRouting};
use crate::content::{
    AssistantContent, TextContent, ThinkingContent, ToolCall, ToolResultContent, UserContent,
};
use crate::context::{Context, Tool};
use crate::event::AssistantMessageEvent;
use crate::message::{
    AssistantMessage, AssistantRole, ConversationMessage, StopReason, ToolResultMessage, Usage,
    UsageCost, UserMessage, UserMessageContent,
};
use crate::model::{Model, calculate_cost};
use crate::provider::{Provider, RequestOptions};

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
    /// 本次最多生成多少 token（老字段名，部分兼容服务商仍在使用）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// 是否让服务端存储本次请求（`false` 表示不存）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// OpenRouter 供应商路由偏好。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<OpenRouterRouting>,
    /// Vercel AI Gateway 路由偏好。
    #[serde(rename = "providerOptions", skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<VercelProviderOptions>,
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

/// Vercel AI Gateway 的 `providerOptions`：把路由偏好包在 `gateway` 下。
#[derive(Clone, Debug, Serialize)]
pub struct VercelProviderOptions {
    pub gateway: VercelGatewayRouting,
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

    /// 系统提示的另一种角色名，部分新模型（推理模型）要求用 `developer`。
    #[serde(rename = "developer")]
    Developer { content: String },

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
        /// 部分 Provider（DeepSeek）重放 assistant 消息时必须带该字段（可为空串）。
        #[serde(skip_serializing_if = "Option::is_none")]
        // 重放历史时用的字段。
        reasoning_content: Option<String>,
    },

    /// 工具执行结果，用 `tool_call_id` 指回它回应的是哪次调用。
    #[serde(rename = "tool")]
    Tool {
        tool_call_id: String,
        content: String,
        /// 部分 Provider 要求工具结果带工具名。
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
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
/// 把 pi 的 Model + Context（内存对象）翻译成一份可以直接 POST /chat/completions 的 JSON 请求体
/// 请求构造函数多收一个「本次选项」，这样它才能把温度、上限写进请求体
#[must_use]
pub fn build_request(model: &Model, context: &Context, options: &RequestOptions) -> ChatRequest {
    // 解析一次，后面复用
    let compat = resolve_openai_completions_compat(model);
    // 请求选项里的 max_tokens 优先，否则用模型默认。
    let max_tokens_value = options.max_tokens.unwrap_or(model.max_tokens);
    // 按兼容设置选择最大 token 字段名：新接口用 max_completion_tokens，老接口用 max_tokens。
    let (max_completion_tokens, max_tokens) = match compat.max_tokens_field {
        MaxTokensField::MaxTokens => (None, Some(max_tokens_value)),
        MaxTokensField::MaxCompletionTokens => (Some(max_tokens_value), None),
    };
    // Vercel 只有在设了 only/order 时才发 providerOptions。
    // 这个模型有没有 Vercel 路由配置，而且里面真的写了规则
    // 有就构造出一个 providerOptions，没有就是 None。
    let provider_options = compat.vercel_gateway_routing.as_ref().and_then(|routing| {
        (routing.only.is_some() || routing.order.is_some()).then(|| VercelProviderOptions {
            gateway: routing.clone(),
        })
    });
    ChatRequest {
        model: model.id.clone(),
        messages: convert_messages(model, context),
        stream: true,
        stream_options: compat.supports_usage_in_streaming.then_some(StreamOptions {
            include_usage: true,
        }),
        max_completion_tokens,
        max_tokens,
        store: compat.supports_store.then_some(false),
        // 把上一步算好的 Vercel 路由，和 OpenRouter 的路由，分别放进请求体的两个字段。
        provider: compat.open_router_routing.clone(),
        provider_options,
        temperature: options.temperature,
        // 把 Context 里的工具列表逐条翻译成 OpenAI 的工具定义；每条都要把 compat 传进去，因为「要不要带 strict」取决于 compat
        tools: context.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|tool| convert_tool(tool, &compat))
                .collect()
        }),
    }
}

/// 把对话上下文翻译成 OpenAI 消息数组。
///
/// 顺序很关键：system 在最前；`assistant` 的工具调用后面必须紧跟
/// 对应的 `tool` 消息，否则服务端会拒绝。
///
/// 按兼容设置处理两件事：
/// - 系统提示用 `system` 还是 `developer` 角色（推理模型可能要求后者）；
/// - 工具结果后紧跟用户消息时，是否要插一条合成 assistant 消息。
#[must_use]
pub fn convert_messages(model: &Model, context: &Context) -> Vec<ChatMessage> {
    // 先解析出这家的兼容设置。后面判断「角色名」「是否插 assistant」「工具结果带不带 name」都用它。
    // 两个条件都要满足。model.reasoning 表示这是推理模型；supports_developer_role 表示服务商接受 developer 角色。只有都真，系统提示才用 developer，否则用 system。
    let compat = resolve_openai_completions_compat(model);
    let use_developer_role = model.reasoning && compat.supports_developer_role;
    let mut messages = Vec::new();
    if let Some(system_prompt) = &context.system_prompt {
        messages.push(if use_developer_role {
            ChatMessage::Developer {
                content: system_prompt.clone(),
            }
        } else {
            ChatMessage::System {
                content: system_prompt.clone(),
            }
        });
    }
    let mut previous_was_tool_result = false;
    // 遍历对话历史
    for message in &context.messages {
        // 模式匹配三种消息之一
        match message {
            ConversationMessage::User(user) => {
                // 部分 Provider 不允许用户消息直接跟在工具结果后面，插一条 assistant 过渡。
                if compat.requires_assistant_after_tool_result && previous_was_tool_result {
                    messages.push(ChatMessage::Assistant {
                        content: Some("I have processed the tool results.".to_owned()),
                        tool_calls: None,
                        reasoning_content: None,
                    });
                }
                messages.push(convert_user(user));
                previous_was_tool_result = false;
            }
            //  模型说的话（文本、思考、工具调用，外加用量、停止原因、模型名）
            ConversationMessage::Assistant(assistant) => {
                // 既无 content 又无 tool_calls 的助手消息会被部分服务端拒绝，跳过。
                if let Some(message) = convert_assistant(assistant, &compat) {
                    messages.push(message);
                }
                previous_was_tool_result = false;
            }
            ConversationMessage::ToolResult(result) => {
                messages.push(convert_tool_result(result, &compat));
                previous_was_tool_result = true;
            }
        }
    }
    messages
}

/// user 消息：文本保持文本，图片转成 data URL。
fn convert_user(message: &UserMessage) -> ChatMessage {
    let content = match &message.content {
        UserMessageContent::Text(text) => ChatUserContent::Text(text.clone()),
        // 内容块数组
        UserMessageContent::Blocks(blocks) => ChatUserContent::Parts(
            blocks
                .iter()
                .map(|block| match block {
                    UserContent::Text(text) => ChatUserPart::Text {
                        text: text.text.clone(),
                    },
                    // 图片
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
fn convert_assistant(
    // 让这个函数能读到「这家 Provider 重放助手消息时要不要带 reasoning_content
    message: &AssistantMessage,
    compat: &ResolvedOpenAICompletionsCompat,
) -> Option<ChatMessage> {
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
    // 返回这条 assistant 消息；对 DeepSeek 这类厂商额外塞一个空的 reasoning_content。
    Some(ChatMessage::Assistant {
        content: (!text.is_empty()).then_some(text),
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        // DeepSeek 等要求重放的 assistant 消息带 reasoning_content（可为空串）。
        reasoning_content: compat
            .requires_reasoning_content_on_assistant_messages
            .then(String::new),
    })
}

/// 工具结果消息：文本块用换行拼接；只有图片时给服务端一个占位文本。
fn convert_tool_result(
    message: &ToolResultMessage,
    compat: &ResolvedOpenAICompletionsCompat,
) -> ChatMessage {
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
        name: compat
            .requires_tool_result_name
            .then(|| message.tool_name.clone()),
    }
}

/// 把 pi 的工具定义翻译成 OpenAI 的 function 工具。
fn convert_tool(tool: &Tool, compat: &ResolvedOpenAICompletionsCompat) -> ChatTool {
    ChatTool::Function {
        function: ChatFunctionDef {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
            // 支持 strict 的 Provider 带上 strict: false（表示不强制）。
            strict: compat.supports_strict_mode.then_some(false),
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

// -----------------------------------------------------------------------------
// 流式聚合：把一串 chunk 拼装成完整的 AssistantMessage
// -----------------------------------------------------------------------------

/// 累积中的一个内容块。
///
/// 流式响应里，文本/思考/工具调用都是一小片一小片到达的；
/// 这里先把它们按顺序攒起来，等流结束时再拼成最终的 `AssistantContent`。
#[derive(Clone, Debug)]
enum Block {
    /// 文本块，`text` 是所有文本片段的累加。
    Text(String),
    /// 思考块，`signature` 记录推理字段名。
    Thinking { text: String, signature: String },
    /// 工具调用块，参数先按原始 JSON 字符串累加，结束时再解析。
    ToolCall {
        id: String,
        name: String,
        arguments_json: String,
    },
}

/// 把一个 SSE chunk 序列聚合成完整助手消息与流式事件。
// SSE = Server-Sent Events（服务器推送事件），是一种基于 HTTP 的流式传输协议：客户端发一个请求后，
// 服务器不一次性返回，而是保持连接，像「挤牙膏」一样一行行把文本吐回来。
pub struct ChatCompletionStream {
    model: Model,
    response_id: Option<String>, // 服务端返回的本次补全 id（第一个 chunk 里取）
    response_model: Option<String>, // 实际服务的模型名（网关可能路由到不同模型）。
    usage: Usage,                // token 用量 + 费用，拿到最后一个 chunk 的 usage 后填
    stop_reason: StopReason,
    raw_stop_reason: Option<String>,
    error_message: Option<String>,
    has_finish_reason: bool, // 是否收到过 finish_reason
    blocks: Vec<Block>,      // 累积的内容块列表，顺序即最终顺序
    text_block: Option<usize>,
    thinking_block: Option<usize>,
    tool_blocks: HashMap<u32, usize>,
    timestamp: u64,
}

impl ChatCompletionStream {
    /// 为指定模型创建一个聚合器。
    /// 聚合器不是把 chunk 存成一堆，而是边收边合并，把一个有状态的过程做出来
    #[must_use]
    pub fn new(model: Model) -> Self {
        Self {
            model,
            response_id: None,
            response_model: None,
            usage: zero_usage(),
            stop_reason: StopReason::Pending,
            raw_stop_reason: None,
            error_message: None,
            has_finish_reason: false,
            blocks: Vec::new(),
            text_block: None,
            thinking_block: None,
            tool_blocks: HashMap::new(),
            timestamp: now_millis(),
        }
    }

    /// 流开始时发出的 `start` 事件。
    #[must_use]
    pub fn start_event(&self) -> AssistantMessageEvent {
        AssistantMessageEvent::Start {
            partial: self.snapshot(),
        }
    }

    /// 处理一个 chunk，返回它产生的中间事件（text/thinking/toolcall 的 start/delta）。
    pub fn handle_chunk(&mut self, chunk: &ChatCompletionChunk) -> Vec<AssistantMessageEvent> {
        let mut events = Vec::new();
        // 记录 response id
        // 服务端给这次补全一个唯一 id（如 "cmpl_1"），每个 chunk 都带同一个。
        // 我们只取第一次遇到的（所以先判断 is_none()）。
        if self.response_id.is_none() {
            self.response_id.clone_from(&chunk.id);
        }
        // 记录实际模型名
        // 你请求的是 "auto"，网关可能实际路由到 "anthropic/claude-3"；
        // 或者请求名和返回名不同。我们记录服务端实际用的模型。
        if let Some(chunk_model) = &chunk.model {
            if !chunk_model.is_empty()
                && chunk_model != &self.model.id
                && self.response_model.is_none()
            {
                self.response_model = Some(chunk_model.clone());
            }
        }
        // 记录 token 用量
        if let Some(usage) = &chunk.usage {
            self.usage = parse_usage(usage, &self.model);
        }
        // 取第一个候选回复
        let Some(choice) = chunk.choices.first() else {
            return events;
        };

        // 少数厂商把用量放在 choice 里，做兜底。
        if chunk.usage.is_none() {
            if let Some(usage) = &choice.usage {
                self.usage = parse_usage(usage, &self.model);
            }
        }
        // 处理结束原因
        if let Some(reason) = &choice.finish_reason {
            self.raw_stop_reason = Some(reason.clone());
            let (stop_reason, error_message) = map_stop_reason(Some(reason));
            self.stop_reason = stop_reason;
            if let Some(message) = error_message {
                self.error_message = Some(message);
            }
            self.has_finish_reason = true;
        }
        // delta 是这一 chunk 的增量内容，结束 chunk 可能没有 delta
        let Some(delta) = &choice.delta else {
            return events;
        };
        // 一个 delta 里可能同时有文本、推理、工具调用片段，所以三者都要处理。
        self.handle_text(delta, &mut events);
        self.handle_thinking(delta, &mut events);
        self.handle_tool_calls(delta, &mut events);

        events
    }
    /*
    handle_chunk 做四件事——

    1.记录元信息（response id / 实际模型 / 用量）；
    2.处理结束原因（finish_reason → stop_reason）；
    3.把 delta 分派给文本/推理/工具三个处理器；
    4.返回本次产生的事件。

    它自己不直接改内容，内容修改都在 handle_text/handle_thinking/handle_tool_calls 里。

    */

    /// 处理文本增量。
    fn handle_text(&mut self, delta: &ChunkDelta, events: &mut Vec<AssistantMessageEvent>) {
        let Some(content) = &delta.content else {
            return; // 没有文本就退出
        };
        if content.is_empty() {
            return; // 空串也退出
        }
        // 些 chunk 的 content 是 ""（空串），没有实际内容。
        // 空串也算「没有」，直接跳过，避免产生无意义的事件。

        let index = match self.text_block {
            Some(index) => index, // 已经有文本块了，直接用它
            None => {
                // 还没有，说明这是第一片文本，要新建
                self.blocks.push(Block::Text(String::new()));
                let index = self.blocks.len() - 1;
                self.text_block = Some(index);
                events.push(AssistantMessageEvent::TextStart {
                    content_index: index as u32,
                    partial: self.snapshot(),
                });
                index
            }
        };
        // 把文本片追加进去
        if let Block::Text(text) = &mut self.blocks[index] {
            text.push_str(content);
        }
        // 发 增量事件
        events.push(AssistantMessageEvent::TextDelta {
            content_index: index as u32,
            delta: content.clone(),
            partial: self.snapshot(),
        });
    }

    /// 处理推理增量（reasoning_content / reasoning / reasoning_text）。
    fn handle_thinking(&mut self, delta: &ChunkDelta, events: &mut Vec<AssistantMessageEvent>) {
        let Some((signature, text)) = first_reasoning(delta) else {
            return;
        };
        let index = match self.thinking_block {
            Some(index) => index,
            None => {
                self.blocks.push(Block::Thinking {
                    text: String::new(),
                    signature,
                });
                let index = self.blocks.len() - 1;
                self.thinking_block = Some(index);
                events.push(AssistantMessageEvent::ThinkingStart {
                    content_index: index as u32,
                    partial: self.snapshot(),
                });
                index
            }
        };
        if let Block::Thinking { text: thinking, .. } = &mut self.blocks[index] {
            thinking.push_str(&text);
        }
        events.push(AssistantMessageEvent::ThinkingDelta {
            content_index: index as u32,
            delta: text,
            partial: self.snapshot(),
        });
    }

    /// 处理工具调用增量（按 chunk 的 index 分组累加）。
    fn handle_tool_calls(&mut self, delta: &ChunkDelta, events: &mut Vec<AssistantMessageEvent>) {
        let Some(tool_calls) = &delta.tool_calls else {
            return;
        };
        for tool_call in tool_calls {
            let index = if let Some(&index) = self.tool_blocks.get(&tool_call.index) {
                index
            } else {
                let id = tool_call.id.clone().unwrap_or_default();
                let name = tool_call
                    .function
                    .as_ref()
                    .and_then(|function| function.name.clone())
                    .unwrap_or_default();
                self.blocks.push(Block::ToolCall {
                    id,
                    name,
                    arguments_json: String::new(),
                });
                let index = self.blocks.len() - 1;
                self.tool_blocks.insert(tool_call.index, index);
                events.push(AssistantMessageEvent::ToolcallStart {
                    content_index: index as u32,
                    partial: self.snapshot(),
                });
                index
            };

            // 第一块可能只带 id/name，后续补齐。
            if let Block::ToolCall { id, name, .. } = &mut self.blocks[index] {
                if id.is_empty() {
                    if let Some(new_id) = &tool_call.id {
                        id.clone_from(new_id);
                    }
                }
                if name.is_empty() {
                    if let Some(function) = &tool_call.function {
                        if let Some(new_name) = &function.name {
                            name.clone_from(new_name);
                        }
                    }
                }
            }

            if let Some(arguments) = tool_call
                .function
                .as_ref()
                .and_then(|function| function.arguments.as_ref())
            {
                if !arguments.is_empty() {
                    if let Block::ToolCall { arguments_json, .. } = &mut self.blocks[index] {
                        arguments_json.push_str(arguments);
                    }
                    events.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: index as u32,
                        delta: arguments.clone(),
                        partial: self.snapshot(),
                    });
                }
            }
        }
    }

    /// 收尾：为每个内容块发 end 事件，最后发 `done` 或 `error`。
    pub fn finish(&mut self) -> Vec<AssistantMessageEvent> {
        let mut events = Vec::new();
        for (index, block) in self.blocks.iter().enumerate() {
            match block {
                // 按块类型发 end
                // 文本/思考：content 是累积后的完整文本（不是片段
                Block::Text(text) => events.push(AssistantMessageEvent::TextEnd {
                    content_index: index as u32,
                    content: text.clone(),
                    partial: self.snapshot(),
                }),
                Block::Thinking { text, .. } => events.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: index as u32,
                    content: text.clone(),
                    partial: self.snapshot(),
                }),
                // 工具调用：把累积的 arguments_json 字符串解析成对象，组装成完整 ToolCall。
                // parse_arguments(arguments_json)：这是唯一把工具参数变成对象的地方（流中一直存字符串）。
                Block::ToolCall {
                    id,
                    name,
                    arguments_json,
                } => events.push(AssistantMessageEvent::ToolcallEnd {
                    content_index: index as u32,
                    tool_call: ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: parse_arguments(arguments_json),
                        thought_signature: None,
                        namespace: None,
                    },
                    partial: self.snapshot(),
                }),
            }
        }
        // 推断结束原因
        // 没收到 finish_reason 时，按是否有工具调用推断结束原因。
        if !self.has_finish_reason && self.stop_reason == StopReason::Pending {
            self.stop_reason = if self
                .blocks
                .iter()
                .any(|block| matches!(block, Block::ToolCall { .. }))
            {
                StopReason::ToolUse
            } else {
                StopReason::Stop
            };
        }
        // 发最终事件
        if self.stop_reason == StopReason::Error {
            events.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: self.snapshot(),
            });
        } else {
            events.push(AssistantMessageEvent::Done {
                reason: self.stop_reason,
                message: self.snapshot(),
            });
        }
        events
    }

    /// 当前累积状态快照（克隆成一条完整助手消息）。
    // 把当前所有累积状态克隆成一条完整助手消息，供事件携带。
    #[must_use]
    pub fn snapshot(&self) -> AssistantMessage {
        AssistantMessage {
            role: AssistantRole::Assistant,
            content: self.blocks.iter().map(block_to_content).collect(),
            api: self.model.api.clone(),
            provider: self.model.provider.clone(),
            model: self.model.id.clone(),
            response_model: self.response_model.clone(),
            response_id: self.response_id.clone(),
            diagnostics: None,
            usage: self.usage.clone(),
            stop_reason: self.stop_reason,
            deferred: None,
            error_message: self.error_message.clone(),
            raw_stop_reason: self.raw_stop_reason.clone(),
            end_turn: None,
            timestamp: self.timestamp,
        }
    }
}

/// 把内部块转成最终内容。
fn block_to_content(block: &Block) -> AssistantContent {
    match block {
        Block::Text(text) => AssistantContent::Text(TextContent {
            text: text.clone(),
            text_signature: None,
        }),
        Block::Thinking { text, signature } => AssistantContent::Thinking(ThinkingContent {
            thinking: text.clone(),
            thinking_signature: if signature.is_empty() {
                None
            } else {
                Some(signature.clone())
            },
            redacted: None,
        }),
        Block::ToolCall {
            id,
            name,
            arguments_json,
        } => AssistantContent::ToolCall(ToolCall {
            id: id.clone(),
            name: name.clone(),
            arguments: parse_arguments(arguments_json),
            thought_signature: None,
            namespace: None,
        }),
    }
}

/// 解析流式累积的工具参数 JSON；解析失败时按空对象处理。
fn parse_arguments(arguments_json: &str) -> Map<String, Value> {
    if arguments_json.is_empty() {
        return Map::new();
    }
    match serde_json::from_str::<Value>(arguments_json) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    }
}

/// 取第一个非空的推理字段，返回（字段名，文本）。
/// 不同 Provider 用不同字段名放推理文本。
/// 为了避免同一内容被重复处理（有的同时返回两个相同字段），只取第一个非空的。
fn first_reasoning(delta: &ChunkDelta) -> Option<(String, String)> {
    let candidates = [
        ("reasoning_content", &delta.reasoning_content),
        ("reasoning", &delta.reasoning),
        ("reasoning_text", &delta.reasoning_text),
    ];
    for (name, value) in candidates {
        if let Some(text) = value {
            if !text.is_empty() {
                return Some((name.to_owned(), text.clone()));
            }
        }
    }
    None
}
/// 返回一个全 0 的用量，作为流开始时的初始值
fn zero_usage() -> Usage {
    Usage {
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
    }
}
/// 取当前时间的 Unix 毫秒数。
fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

// -----------------------------------------------------------------------------
// Provider：把「构造请求 → 传输 → 聚合」串起来
// -----------------------------------------------------------------------------

/// 基于 OpenAI-compatible Chat Completions 的 Provider。
///
/// 它持有一个 HTTP 传输层；`stream` 时构造请求、发送、把 SSE 响应聚合成事件。
pub struct OpenAiCompletionsProvider {
    transport: Box<dyn HttpTransport>,
    api_key: String,
}

impl OpenAiCompletionsProvider {
    /// 用传输层和 API Key 创建 Provider。
    #[must_use]
    pub fn new(transport: Box<dyn HttpTransport>, api_key: impl Into<String>) -> Self {
        Self {
            transport,
            api_key: api_key.into(),
        }
    }
    // 调用你之前写好的请求体转换，得到 ChatRequest。
    /// 构造发给 `/chat/completions` 的 HTTP 请求。
    /// 把请求体序列化成 JSON 字符串。改动就是把 options 继续往下传给 build_request。这是一层中转
    fn build_http_request(
        &self,
        model: &Model,
        context: &Context,
        options: &RequestOptions,
    ) -> Result<HttpRequest, HttpError> {
        let body = serde_json::to_string(&build_request(model, context, options))
            .map_err(|error| HttpError::new(format!("序列化请求失败: {error}")))?;
        let url = format!("{}/chat/completions", model.base_url.trim_end_matches('/'));
        Ok(HttpRequest {
            url,
            headers: vec![
                ("Content-Type".to_owned(), "application/json".to_owned()),
                (
                    "Authorization".to_owned(),
                    format!("Bearer {}", self.api_key),
                ),
            ],
            body,
        })
    }

    /// 发请求并把 SSE 响应体聚合成事件；失败时返回单个 error 事件。
    /// 发送成功 → aggregate_sse 把响应体聚合成事件
    /// 实际含义：把所有失败都收敛成「一个 error 事件」，
    /// 这样上层（Agent）不用区分错误类型，统一在事件流里看到错误。
    /// 发请求并聚合
    fn run(
        &self,
        model: &Model,
        context: &Context,
        options: &RequestOptions,
    ) -> Vec<AssistantMessageEvent> {
        let request = match self.build_http_request(model, context, options) {
            Ok(request) => request,
            Err(error) => return vec![error_event(model, &error.message)],
        };
        match self.transport.post(&request) {
            Ok(body) => aggregate_sse(model, &body),
            Err(error) => vec![error_event(model, &error.message)],
        }
    }
}

impl Provider for OpenAiCompletionsProvider {
    fn id(&self) -> &str {
        "openai-completions"
    }

    fn name(&self) -> &str {
        "OpenAI-compatible Chat Completions"
    }

    fn get_models(&self) -> &[Model] {
        &[]
    }
    // 接口实现，把 run 返回的 Vec 变成装箱迭代器。改动就是把 options 一路传进来。
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &RequestOptions,
    ) -> Box<dyn Iterator<Item = AssistantMessageEvent>> {
        Box::new(self.run(model, context, options).into_iter())
    }
}

/// 把一段 SSE 响应体聚合成事件序列。
///
/// 逐行读：以 `data:` 开头的行取出 JSON，解析成 chunk 喂给聚合器；
/// 遇到 `[DONE]` 或读完即结束。
#[must_use]
pub fn aggregate_sse(model: &Model, body: &str) -> Vec<AssistantMessageEvent> {
    let mut stream = ChatCompletionStream::new(model.clone());
    let mut events = vec![stream.start_event()];
    for line in body.lines() {
        let Some(payload) = parse_sse_data(line) else {
            continue;
        };
        if payload == "[DONE]" {
            break;
        }
        if let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(payload) {
            events.extend(stream.handle_chunk(&chunk));
        }
    }
    events.extend(stream.finish());
    events
}

/// 从 SSE 行里取出 `data:` 后面的内容；不是数据行或为空时返回 `None`。
fn parse_sse_data(line: &str) -> Option<&str> {
    line.strip_prefix("data:")
        .map(str::trim)
        .filter(|payload| !payload.is_empty())
}

/// 构造一个只带错误信息的助手消息，用于传输/解析失败时。
fn error_event(model: &Model, message: &str) -> AssistantMessageEvent {
    let error = AssistantMessage {
        role: AssistantRole::Assistant,
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: zero_usage(),
        stop_reason: StopReason::Error,
        deferred: None,
        error_message: Some(message.to_owned()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_millis(),
    };
    AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error,
    }
}
