//! Anthropic Messages 协议（请求方向 + 流式响应）。
//!
//! 对应原版 `pi/packages/ai/src/api/anthropic-messages.ts`。
//!
//! 它与 OpenAI Chat Completions 的三大区别：
//! 1. `system` 是独立数组字段（`[{type:"text", text, cache_control}]`），不是一条消息；
//! 2. SSE 用 `event:` + `data:` 两行，空行才触发 flush；内容块按顶层 `index` 索引，
//!    并有显式的 `content_block_start` / `content_block_stop`；
//! 3. thinking 参数是三模式对象（adaptive / enabled / disabled）。
//!
//! C++ 对照：`MessagesRequest` 这类结构体是只用于序列化的 DTO，
//! 字段名必须和线上 JSON 完全一致（注意这里是 snake_case，不是 camelCase）。

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::api::http::{HttpError, HttpRequest, HttpTransport};
use crate::compat::{AnthropicAllowedFallbackModel, AnthropicMessagesCompat};
use crate::content::{
    AssistantContent, ImageContent, TextContent, ThinkingContent, ToolCall, ToolResultContent,
    UserContent,
};
use crate::context::{ConstrainedSamplingConfig, Context, JsonSchemaStrict, Tool};
use crate::event::AssistantMessageEvent;
use crate::message::{
    AssistantMessage, AssistantRole, ConversationMessage, StopReason, ToolResultMessage, Usage,
    UsageCost, UserMessageContent,
};
use crate::model::{Model, ModelCompat, ModelThinkingLevel, calculate_cost};
use crate::provider::{CacheRetention, Provider, RequestOptions};

/// 细粒度工具流式 beta（Anthropic 要求显式声明才返回工具参数增量）。
const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
/// 交错思考 beta（工具调用之间也能思考）。
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

/// SSE 事件名白名单；不在此列的事件直接跳过（与 `ANTHROPIC_MESSAGE_EVENTS` 一致）。
const ANTHROPIC_MESSAGE_EVENTS: [&str; 6] = [
    "message_start",
    "message_delta",
    "message_stop",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
];

// -----------------------------------------------------------------------------
// 兼容配置：解析成确定值
// -----------------------------------------------------------------------------

/// 解析后的 Anthropic 兼容设置：字段都是确定值，不再有 `Option`。
///
/// 对应原版 `getAnthropicCompat`：没有按 URL 探测，只有「默认值 + 显式覆盖」。
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedAnthropicCompat {
    /// 是否支持「随到随流」的工具参数（不支持时要用细粒度 beta 头）。
    pub supports_eager_tool_input_streaming: bool,
    /// 是否支持 1 小时长缓存（决定 `cache_control.ttl` 是否发 "1h"）。
    pub supports_long_cache_retention: bool,
    /// 是否发送会话亲和头 `x-session-affinity`。
    pub send_session_affinity_headers: bool,
    /// 是否允许在工具定义上打 `cache_control`。
    pub supports_cache_control_on_tools: bool,
    /// 是否支持 `temperature`（新版 Opus 不支持）。
    pub supports_temperature: bool,
    /// 是否强制使用自适应思考（`{type:"adaptive"}` + effort）。
    pub force_adaptive_thinking: bool,
    /// 是否允许回传空签名思考块（少数兼容服务商）。
    pub allow_empty_signature: bool,
    /// 是否支持严格模式工具（要求模型严格遵守参数 schema）。
    pub supports_strict_tools: bool,
    /// 服务端拒绝回退白名单。
    pub allowed_fallback_models: Vec<AnthropicAllowedFallbackModel>,
}

impl Default for ResolvedAnthropicCompat {
    fn default() -> Self {
        // 默认值对齐原版 `getAnthropicCompat` 的 ?? 兜底。
        Self {
            supports_eager_tool_input_streaming: true,
            supports_long_cache_retention: true,
            send_session_affinity_headers: false,
            supports_cache_control_on_tools: true,
            supports_temperature: true,
            force_adaptive_thinking: false,
            allow_empty_signature: false,
            supports_strict_tools: false,
            allowed_fallback_models: Vec::new(),
        }
    }
}

/// 解析兼容设置：默认值打底，再用 `model.compat` 里显式的 Anthropic 配置覆盖。
#[must_use]
pub fn resolve_anthropic_compat(model: &Model) -> ResolvedAnthropicCompat {
    let Some(ModelCompat::AnthropicMessages(compat)) = &model.compat else {
        return ResolvedAnthropicCompat::default();
    };
    let compat: &AnthropicMessagesCompat = compat;
    let defaults = ResolvedAnthropicCompat::default();
    ResolvedAnthropicCompat {
        supports_eager_tool_input_streaming: compat
            .supports_eager_tool_input_streaming
            .unwrap_or(defaults.supports_eager_tool_input_streaming),
        supports_long_cache_retention: compat
            .supports_long_cache_retention
            .unwrap_or(defaults.supports_long_cache_retention),
        send_session_affinity_headers: compat
            .send_session_affinity_headers
            .unwrap_or(defaults.send_session_affinity_headers),
        supports_cache_control_on_tools: compat
            .supports_cache_control_on_tools
            .unwrap_or(defaults.supports_cache_control_on_tools),
        supports_temperature: compat
            .supports_temperature
            .unwrap_or(defaults.supports_temperature),
        force_adaptive_thinking: compat
            .force_adaptive_thinking
            .unwrap_or(defaults.force_adaptive_thinking),
        allow_empty_signature: compat
            .allow_empty_signature
            .unwrap_or(defaults.allow_empty_signature),
        supports_strict_tools: compat
            .supports_strict_tools
            .unwrap_or(defaults.supports_strict_tools),
        allowed_fallback_models: compat.allowed_fallback_models.clone().unwrap_or_default(),
    }
}

// -----------------------------------------------------------------------------
// 请求 DTO   要 POST 出去的请求体
// -----------------------------------------------------------------------------

/// `POST /v1/messages` 的请求体。
#[derive(Clone, Debug, Serialize)]
pub struct MessagesRequest {
    /// 模型 id，例如 "claude-sonnet-4-5"。
    pub model: String,
    /// 对话消息。注意：system 不在里面，它是独立字段。
    pub messages: Vec<MessageParam>,
    /// 本次最多生成多少 token（Anthropic 必填）。
    pub max_tokens: u64,
    /// 是否流式返回。
    pub stream: bool,
    /// 系统提示数组；每个元素是 `{type:"text", text, cache_control}`。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<RequestBlock>>,
    /// 采样温度；与扩展思考不兼容，思考开启时不发。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// 思考参数（三模式）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    /// 自适应思考的 effort 配置。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
    /// 可用工具定义。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
}

/// 一条消息的角色。Anthropic 只有 user / assistant 两种（工具结果属于 user）。
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum MessageRole {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
}

/// 一条消息。
#[derive(Clone, Debug, Serialize)]
pub struct MessageParam {
    pub role: MessageRole,
    pub content: MessageContent,
}

/// 消息正文：纯字符串，或内容块数组。
///
/// C++ 对照：`std::variant<std::string, std::vector<RequestBlock>>`。
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<RequestBlock>),
}

/// 请求方向的内容块。用 `type` 字段区分形状。
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum RequestBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "image")]
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    /// 被安全策略屏蔽的思考：原文不可见，只回传不透明数据。
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: MessageContent,
        is_error: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// 图片块的数据来源：base64 内嵌。
#[derive(Clone, Debug, Serialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub kind: SourceKind,
    pub media_type: String,
    pub data: String,
}

/// 图片来源类型。当前只有 base64。
#[derive(Clone, Copy, Debug, Serialize)]
pub enum SourceKind {
    #[serde(rename = "base64")]
    Base64,
}

/// 提示缓存标记。
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: CacheKind,
    /// 长缓存时为 "1h"。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

/// 缓存标记类型。当前只有 ephemeral（临时缓存）。
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum CacheKind {
    #[serde(rename = "ephemeral")]
    Ephemeral,
}

/// 思考参数：三种模式。
///
/// C++ 对照：Rust 的枚举直接把「哪个字段有哪些字段」编码进类型；
/// TS 里那是三个不同的 interface 组成的联合类型。
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum ThinkingConfig {
    /// 自适应：由模型自己决定思考多少，可配 effort。
    #[serde(rename = "adaptive")]
    Adaptive { display: ThinkingDisplay },
    /// 按预算思考（老模型）。
    #[serde(rename = "enabled")]
    Enabled {
        budget_tokens: u64,
        display: ThinkingDisplay,
    },
    /// 显式关闭思考。
    #[serde(rename = "disabled")]
    Disabled,
}

/// 思考摘要的展示方式。
#[derive(Clone, Copy, Debug, Serialize)]
pub enum ThinkingDisplay {
    #[serde(rename = "summarized")]
    Summarized,
    #[serde(rename = "omitted")]
    Omitted,
}

/// 自适应思考的 effort 配置。
#[derive(Clone, Debug, Serialize)]
pub struct OutputConfig {
    pub effort: String,
}

/// Anthropic 的工具定义。
#[derive(Clone, Debug, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    pub description: String,
    /// 是否让服务端尽早流式返回工具参数。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eager_input_streaming: Option<bool>,
    /// 是否要求严格遵守参数 schema。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    pub input_schema: Value,
    /// 提示缓存标记（只打在最后一个工具上）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

// -----------------------------------------------------------------------------
// 请求构造
// -----------------------------------------------------------------------------

/// 把 pi 的模型和上下文翻译成 Anthropic 请求体。
#[must_use]
pub fn build_request(
    model: &Model,
    context: &Context,
    options: &RequestOptions,
) -> MessagesRequest {
    let compat = resolve_anthropic_compat(model);
    // 思考开关：`Some(非 off)` 才开启；None 与 off 都走「关闭」分支。
    let thinking_enabled = matches!(
        options.reasoning_effort,
        Some(level) if level != ModelThinkingLevel::Off
    );
    let cache_control = resolve_cache_control(&compat, options.cache_retention);

    let messages = convert_messages(model, context, cache_control.as_ref());

    let system = context.system_prompt.as_ref().map(|prompt| {
        vec![RequestBlock::Text {
            text: prompt.clone(),
            cache_control: cache_control.clone(),
        }]
    });

    // 温度与扩展思考不兼容；厂商不支持时也不发。
    let temperature = options
        .temperature
        .filter(|_| !thinking_enabled && compat.supports_temperature);

    let (thinking, output_config) = resolve_thinking(model, &compat, options, thinking_enabled);

    let tools = context.tools.as_ref().and_then(|tools| {
        if tools.is_empty() {
            return None;
        }
        // 只有厂商支持时才把缓存标记打到工具上。
        let tool_cache_control = if compat.supports_cache_control_on_tools {
            cache_control.as_ref()
        } else {
            None
        };
        Some(convert_tools(
            tools,
            compat.supports_eager_tool_input_streaming,
            compat.supports_strict_tools,
            tool_cache_control,
        ))
    });

    MessagesRequest {
        model: model.id.clone(),
        messages,
        max_tokens: options.max_tokens.unwrap_or(model.max_tokens),
        stream: true,
        system,
        temperature,
        thinking,
        output_config,
        tools,
    }
}

/// 计算提示缓存标记。
///
/// 与原版差异：不读 `PI_CACHE_RETENTION` 环境变量，只认调用方传入的 `cache_retention`。
fn resolve_cache_control(
    compat: &ResolvedAnthropicCompat,
    cache_retention: Option<CacheRetention>,
) -> Option<CacheControl> {
    let retention = cache_retention.unwrap_or_default();
    if retention == CacheRetention::None {
        return None;
    }
    let ttl = (retention == CacheRetention::Long && compat.supports_long_cache_retention)
        .then(|| "1h".to_owned());
    Some(CacheControl {
        kind: CacheKind::Ephemeral,
        ttl,
    })
}

/// 按 Anthropic 约定把 `cache_control` 打在最后一条 user 消息的最后一个块上。
fn apply_last_message_cache_control(messages: &mut [MessageParam], cache_control: &CacheControl) {
    let Some(last) = messages.last_mut() else {
        return;
    };
    if last.role != MessageRole::User {
        return;
    }
    match &mut last.content {
        // 纯字符串要包成单元素数组才能挂标记。
        MessageContent::Text(text) => {
            let text = std::mem::take(text);
            last.content = MessageContent::Blocks(vec![RequestBlock::Text {
                text,
                cache_control: Some(cache_control.clone()),
            }]);
        }
        MessageContent::Blocks(blocks) => {
            if let Some(block) = blocks.last_mut() {
                match block {
                    RequestBlock::Text {
                        cache_control: slot,
                        ..
                    }
                    | RequestBlock::Image {
                        cache_control: slot,
                        ..
                    }
                    | RequestBlock::ToolResult {
                        cache_control: slot,
                        ..
                    } => {
                        *slot = Some(cache_control.clone());
                    }
                    RequestBlock::Thinking { .. }
                    | RequestBlock::RedactedThinking { .. }
                    | RequestBlock::ToolUse { .. } => {}
                }
            }
        }
    }
}

/// 计算要发送的思考参数，返回（thinking, output_config）。
///
/// 与原版差异：原版由 `streamSimple` 按思考级别算出 budget 并同时调整 max_tokens；
/// 这里 `RequestOptions` 没有 thinkingBudgets，所以预算固定用 1024。
fn resolve_thinking(
    model: &Model,
    compat: &ResolvedAnthropicCompat,
    options: &RequestOptions,
    thinking_enabled: bool,
) -> (Option<ThinkingConfig>, Option<OutputConfig>) {
    if !model.reasoning {
        return (None, None);
    }
    if thinking_enabled {
        let display = ThinkingDisplay::Summarized;
        if compat.force_adaptive_thinking {
            let effort = options
                .reasoning_effort
                .map(|level| map_thinking_level_to_effort(model, level));
            return (
                Some(ThinkingConfig::Adaptive { display }),
                effort.map(|effort| OutputConfig { effort }),
            );
        }
        return (
            Some(ThinkingConfig::Enabled {
                budget_tokens: 1024,
                display,
            }),
            None,
        );
    }
    // 关闭思考：只有模型没有显式声明 off 不支持时才发 disabled。
    let off_is_disabled = model
        .thinking_level_map
        .as_ref()
        .is_some_and(|map| matches!(map.get(&ModelThinkingLevel::Off), Some(None)));
    if off_is_disabled {
        return (None, None);
    }
    (Some(ThinkingConfig::Disabled), None)
}

/// 思考级别对应的自适应 effort：模型显式映射优先，否则按级别分档。
fn map_thinking_level_to_effort(model: &Model, level: ModelThinkingLevel) -> String {
    if let Some(Some(value)) = model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.get(&level))
    {
        return value.clone();
    }
    match level {
        ModelThinkingLevel::Minimal | ModelThinkingLevel::Low => "low",
        ModelThinkingLevel::Medium => "medium",
        ModelThinkingLevel::High
        | ModelThinkingLevel::Xhigh
        | ModelThinkingLevel::Max
        | ModelThinkingLevel::Off => "high",
    }
    .to_owned()
}

/// 把对话上下文翻译成 Anthropic 消息数组。
///
/// 与原版差异：不做 `transformMessages`（图片降级、孤儿工具调用补全等），
/// 但保留它的净效果之一：工具调用 id 会被规范化。
/// ser 过滤空文本、assistant 过滤空块、连续 tool_result 合并成一条 user、最后打缓存标记
#[must_use]
pub fn convert_messages(
    model: &Model,
    context: &Context,
    cache_control: Option<&CacheControl>,
) -> Vec<MessageParam> {
    let compat = resolve_anthropic_compat(model);
    let mut params: Vec<MessageParam> = Vec::new();
    let mut i = 0;
    while i < context.messages.len() {
        match &context.messages[i] {
            ConversationMessage::User(user) => {
                match &user.content {
                    UserMessageContent::Text(text) => {
                        if !text.trim().is_empty() {
                            params.push(MessageParam {
                                role: MessageRole::User,
                                content: MessageContent::Text(text.clone()),
                            });
                        }
                    }
                    UserMessageContent::Blocks(blocks) => {
                        let converted = convert_user_blocks(blocks);
                        if !converted.is_empty() {
                            params.push(MessageParam {
                                role: MessageRole::User,
                                content: MessageContent::Blocks(converted),
                            });
                        }
                    }
                }
                i += 1;
            }
            ConversationMessage::Assistant(assistant) => {
                let blocks = convert_assistant(assistant, &compat);
                if !blocks.is_empty() {
                    params.push(MessageParam {
                        role: MessageRole::Assistant,
                        content: MessageContent::Blocks(blocks),
                    });
                }
                i += 1;
            }
            ConversationMessage::ToolResult(_) => {
                // 连续的多个工具结果合并成一条 user 消息（z.ai 的 Anthropic 端点要求）。
                let mut blocks = Vec::new();
                while i < context.messages.len() {
                    let ConversationMessage::ToolResult(result) = &context.messages[i] else {
                        break;
                    };
                    blocks.push(convert_tool_result(result));
                    i += 1;
                }
                params.push(MessageParam {
                    role: MessageRole::User,
                    content: MessageContent::Blocks(blocks),
                });
            }
        }
    }
    // 最后一条 user 消息的最后一个块挂上缓存标记（缓存整段对话历史）。
    if let Some(cache_control) = cache_control {
        apply_last_message_cache_control(&mut params, cache_control);
    }
    params
}

/// user 消息的内容块：过滤掉空白文本块。
fn convert_user_blocks(blocks: &[UserContent]) -> Vec<RequestBlock> {
    let mut converted = Vec::new();
    for block in blocks {
        match block {
            UserContent::Text(text) => {
                if text.text.trim().is_empty() {
                    continue;
                }
                converted.push(RequestBlock::Text {
                    text: text.text.clone(),
                    cache_control: None,
                });
            }
            UserContent::Image(image) => converted.push(image_block(image)),
        }
    }
    converted
}

/// 图片内容块（base64 内嵌）。
fn image_block(image: &ImageContent) -> RequestBlock {
    RequestBlock::Image {
        source: ImageSource {
            kind: SourceKind::Base64,
            media_type: image.mime_type.clone(),
            data: image.data.clone(),
        },
        cache_control: None,
    }
}

/// assistant 消息的内容块。
fn convert_assistant(
    message: &AssistantMessage,
    compat: &ResolvedAnthropicCompat,
) -> Vec<RequestBlock> {
    let mut blocks = Vec::new();
    for block in &message.content {
        match block {
            AssistantContent::Text(content) => {
                if content.text.trim().is_empty() {
                    continue;
                }
                blocks.push(RequestBlock::Text {
                    text: content.text.clone(),
                    cache_control: None,
                });
            }
            AssistantContent::Thinking(content) => {
                convert_thinking_block(content, compat, &mut blocks);
            }
            AssistantContent::ToolCall(call) => blocks.push(RequestBlock::ToolUse {
                id: normalize_tool_call_id(&call.id),
                name: call.name.clone(),
                input: Value::Object(call.arguments.clone()),
            }),
        }
    }
    blocks
}

/// 把思考块翻译成 Anthropic 能接受的形式。
fn convert_thinking_block(
    content: &ThinkingContent,
    compat: &ResolvedAnthropicCompat,
    blocks: &mut Vec<RequestBlock>,
) {
    // 被屏蔽的思考：把不透明数据原样回传。
    if content.redacted == Some(true) {
        blocks.push(RequestBlock::RedactedThinking {
            data: content.thinking_signature.clone().unwrap_or_default(),
        });
        return;
    }
    let signature = content.thinking_signature.clone().unwrap_or_default();
    let has_signature = !signature.trim().is_empty();
    if content.thinking.trim().is_empty() && !has_signature {
        return;
    }
    if has_signature {
        blocks.push(RequestBlock::Thinking {
            thinking: content.thinking.clone(),
            signature,
        });
    } else if compat.allow_empty_signature {
        // 少数兼容服务商接受空签名。
        blocks.push(RequestBlock::Thinking {
            thinking: content.thinking.clone(),
            signature: String::new(),
        });
    } else {
        // 没有签名（例如流被中断）时降级成普通文本，Anthropic 才认。
        blocks.push(RequestBlock::Text {
            text: content.thinking.clone(),
            cache_control: None,
        });
    }
}

/// 工具结果消息 -> tool_result 块。
fn convert_tool_result(message: &ToolResultMessage) -> RequestBlock {
    RequestBlock::ToolResult {
        tool_use_id: normalize_tool_call_id(&message.tool_call_id),
        content: convert_content_blocks(&message.content),
        is_error: message.is_error,
        // 缓存标记统一在整条消息的最后一个块上打，这里先留空。
        cache_control: None,
    }
}

/// 工具结果正文：没有图片时拼成字符串，有图片时用块数组。
///
/// 与原版一致：只有图片时补一个 `(see attached image)` 占位文本块。
fn convert_content_blocks(content: &[ToolResultContent]) -> MessageContent {
    let has_images = content
        .iter()
        .any(|block| matches!(block, ToolResultContent::Image(_)));
    if !has_images {
        let text = content
            .iter()
            .map(|block| match block {
                ToolResultContent::Text(text) => text.text.as_str(),
                ToolResultContent::Image(_) => "",
            })
            .collect::<Vec<_>>()
            .join("\n");
        return MessageContent::Text(text);
    }

    let mut blocks: Vec<RequestBlock> = Vec::new();
    for block in content {
        match block {
            ToolResultContent::Text(text) => blocks.push(RequestBlock::Text {
                text: text.text.clone(),
                cache_control: None,
            }),
            ToolResultContent::Image(image) => blocks.push(image_block(image)),
        }
    }
    if !blocks
        .iter()
        .any(|block| matches!(block, RequestBlock::Text { .. }))
    {
        blocks.insert(
            0,
            RequestBlock::Text {
                text: "(see attached image)".to_owned(),
                cache_control: None,
            },
        );
    }
    MessageContent::Blocks(blocks)
}

/// 把 pi 的工具定义翻译成 Anthropic 工具。 只给最后一个工具打缓存标记
fn convert_tools(
    tools: &[Tool],
    supports_eager_tool_input_streaming: bool,
    supports_strict_tools: bool,
    cache_control: Option<&CacheControl>,
) -> Vec<AnthropicTool> {
    let last_index = tools.len().saturating_sub(1);
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let strict = resolve_strict_sampling(tool, supports_strict_tools);
            AnthropicTool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                eager_input_streaming: supports_eager_tool_input_streaming.then_some(true),
                strict: strict.then_some(true),
                input_schema: build_input_schema(&tool.parameters, strict),
                cache_control: (index == last_index)
                    .then(|| cache_control.cloned())
                    .flatten(),
            }
        })
        .collect()
}

/// 工具是否要求严格采样：厂商支持 strict，且工具显式要求 require。
fn resolve_strict_sampling(tool: &Tool, supports_strict_tools: bool) -> bool {
    if !supports_strict_tools {
        return false;
    }
    matches!(
        tool.constrained_sampling,
        Some(ConstrainedSamplingConfig::JsonSchema {
            strict: JsonSchemaStrict::Require
        })
    )
}

/// 构造工具的 `input_schema`。
///
/// 非严格模式只发 `{type, properties, required}`；
/// 严格模式先摊平原 schema，再用这三个字段覆盖。
fn build_input_schema(parameters: &Value, strict: bool) -> Value {
    let schema = parameters.as_object();
    let properties = schema
        .and_then(|schema| schema.get("properties"))
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let required = schema
        .and_then(|schema| schema.get("required"))
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()));

    let mut legacy = Map::new();
    legacy.insert("type".to_owned(), Value::String("object".to_owned()));
    legacy.insert("properties".to_owned(), properties);
    legacy.insert("required".to_owned(), required);

    if strict {
        let mut merged = schema.cloned().unwrap_or_default();
        for (key, value) in legacy {
            merged.insert(key, value);
        }
        return Value::Object(merged);
    }
    Value::Object(legacy)
}

/// 把工具调用 id 规范成 Anthropic 允许的字符集与长度（非字母数字变下划线，截断 64）。
fn normalize_tool_call_id(id: &str) -> String {
    id.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}

// -----------------------------------------------------------------------------
// 响应 DTO
// -----------------------------------------------------------------------------

/// 流式响应的一个事件。
///
/// 用顶层 `type` 字段区分形状，对应原版 `RawMessageStreamEvent`。
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
pub enum RawMessageStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartMessage },
    #[serde(rename = "message_delta")]
    MessageDelta {
        #[serde(default)]
        delta: MessageDeltaPayload,
        #[serde(default)]
        usage: Option<AnthropicUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        #[serde(default)]
        index: u32,
        content_block: StartBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        #[serde(default)]
        index: u32,
        delta: DeltaPayload,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop {
        #[serde(default)]
        index: u32,
    },
}

/// `message_start` 里携带的初始消息。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct MessageStartMessage {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
}

/// Anthropic 的 token 用量。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct AnthropicUsage {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    /// 缓存写入总量（含 1 小时部分）。
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_creation: Option<CacheCreation>,
    #[serde(default)]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

/// 缓存写入明细。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct CacheCreation {
    /// 1 小时长缓存的写入 token 数。
    #[serde(default)]
    pub ephemeral_1h_input_tokens: Option<u64>,
}

/// 输出 token 明细。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct OutputTokensDetails {
    /// 其中用于思考的 token（是 output_tokens 的子集）。
    #[serde(default)]
    pub thinking_tokens: Option<u64>,
}

/// `message_delta` 里的 delta。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct MessageDeltaPayload {
    #[serde(default)]
    pub stop_reason: Option<String>,
    /// 拒绝作答时的补充说明。
    #[serde(default)]
    pub stop_details: Option<RefusalStopDetails>,
}

/// 拒绝作答的详情。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct RefusalStopDetails {
    #[serde(default)]
    pub explanation: Option<String>,
}

/// `content_block_start` 里的内容块。
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
pub enum StartBlock {
    #[serde(rename = "text")]
    Text {
        #[serde(default)]
        text: Option<String>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        thinking: Option<String>,
        #[serde(default)]
        signature: Option<String>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Option<Value>,
    },
    /// 未知内容块类型（API 可能新增）；跳过，与原版 if-else 链的兜底一致。
    #[serde(other)]
    Other,
}

/// `content_block_delta` 里的增量。
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
pub enum DeltaPayload {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
    /// 未知增量类型；跳过。
    #[serde(other)]
    Other,
}

/// 把 Anthropic 的 stop reason 映射成 pi 的 `StopReason`。
///
/// 第二个返回值是出错时的错误信息；未知原因返回 `Err`（原版是抛异常）。
pub fn map_stop_reason(
    reason: &str,
    explanation: Option<&str>,
) -> Result<(StopReason, Option<String>), String> {
    match reason {
        "end_turn" => Ok((StopReason::Stop, None)),
        "max_tokens" => Ok((StopReason::Length, None)),
        "tool_use" => Ok((StopReason::ToolUse, None)),
        "refusal" => Ok((
            StopReason::Error,
            Some(
                explanation
                    .map(str::to_owned)
                    .unwrap_or_else(|| "The model refused to complete the request".to_owned()),
            ),
        )),
        // pause_turn 表示服务端还要继续，对客户端来说按正常结束处理。
        "pause_turn" => Ok((StopReason::Stop, None)),
        // 我们不发送 stop_sequence，所以理论上不会出现。
        "stop_sequence" => Ok((StopReason::Stop, None)),
        "sensitive" => Ok((
            StopReason::Error,
            Some("Provider stopped with: sensitive".to_owned()),
        )),
        other => Err(format!("Unhandled stop reason: {other}")),
    }
}

// -----------------------------------------------------------------------------
// 流式聚合：把一串事件拼装成完整的 AssistantMessage
// -----------------------------------------------------------------------------

/// 累积中的一个内容块。
///
/// 每个块都记住 Anthropic 的顶层 `index`，因为 delta/stop 事件靠它定位块。
#[derive(Clone, Debug)]
enum Block {
    Text {
        index: u32,
        text: String,
    },
    Thinking {
        index: u32,
        text: String,
        signature: String,
        /// 是否为被安全策略屏蔽的思考；回传时要还原成 `redacted_thinking`。
        redacted: bool,
    },
    ToolCall {
        index: u32,
        id: String,
        name: String,
        arguments: Map<String, Value>,
        /// 流式到达的 JSON 片段，收尾时解析成 `arguments`。
        partial_json: String,
    },
}

impl Block {
    fn index(&self) -> u32 {
        match self {
            Self::Text { index, .. }
            | Self::Thinking { index, .. }
            | Self::ToolCall { index, .. } => *index,
        }
    }
}

/// 把 Anthropic 事件序列聚合成完整助手消息与流式事件。
pub struct AnthropicStream {
    model: Model,
    compat: ResolvedAnthropicCompat,
    /// 计算费用用的模型：服务端回退到其它模型时会换掉价格表。
    usage_model: Model,
    response_id: Option<String>,
    response_model: Option<String>,
    usage: Usage,
    stop_reason: StopReason,
    raw_stop_reason: Option<String>,
    error_message: Option<String>,
    blocks: Vec<Block>,
    saw_message_start: bool,
    saw_message_stop: bool,
    timestamp: u64,
}

impl AnthropicStream {
    /// 为指定模型创建一个聚合器。
    #[must_use]
    pub fn new(model: Model) -> Self {
        let compat = resolve_anthropic_compat(&model);
        Self {
            usage_model: model.clone(),
            model,
            compat,
            response_id: None,
            response_model: None,
            usage: zero_usage(),
            stop_reason: StopReason::Pending,
            raw_stop_reason: None,
            error_message: None,
            blocks: Vec::new(),
            saw_message_start: false,
            saw_message_stop: false,
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

    /// 处理一个事件，返回它产生的中间事件。
    pub fn handle_event(&mut self, event: &RawMessageStreamEvent) -> Vec<AssistantMessageEvent> {
        match event {
            RawMessageStreamEvent::MessageStart { message } => {
                self.handle_message_start(message);
                Vec::new()
            }
            RawMessageStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => self.start_block(*index, content_block),
            RawMessageStreamEvent::ContentBlockDelta { index, delta } => {
                self.apply_delta(*index, delta)
            }
            RawMessageStreamEvent::ContentBlockStop { index } => self.stop_block(*index),
            RawMessageStreamEvent::MessageDelta { delta, usage } => {
                self.handle_message_delta(delta, usage.as_ref());
                Vec::new()
            }
            RawMessageStreamEvent::MessageStop => {
                self.saw_message_stop = true;
                Vec::new()
            }
        }
    }

    /// `message_start`：取响应 id、实际模型、初始用量。
    fn handle_message_start(&mut self, message: &MessageStartMessage) {
        self.saw_message_start = true;
        self.response_id = message.id.clone();
        // 服务端可能回退到别的模型；记录实际模型，并按它的价格表算费。
        if let Some(response_model) = &message.model {
            if response_model != &self.model.id {
                self.response_model = Some(response_model.clone());
                let fallback = self.compat.allowed_fallback_models.iter().find(|fallback| {
                    fallback.provider == self.model.provider && &fallback.model == response_model
                });
                if let Some(fallback) = fallback {
                    self.usage_model.id.clone_from(response_model);
                    self.usage_model.cost = fallback.cost.clone();
                }
            }
        }

        let usage = message.usage.clone().unwrap_or_default();
        self.usage.input = usage.input_tokens.unwrap_or(0);
        self.usage.output = usage.output_tokens.unwrap_or(0);
        self.usage.cache_read = usage.cache_read_input_tokens.unwrap_or(0);
        self.usage.cache_write = usage.cache_creation_input_tokens.unwrap_or(0);
        self.usage.cache_write_1h = Some(
            usage
                .cache_creation
                .as_ref()
                .and_then(|creation| creation.ephemeral_1h_input_tokens)
                .unwrap_or(0),
        );
        self.recalc_usage();
    }

    /// `content_block_start`：新建一个内容块并发对应 `*_start` 事件。
    fn start_block(
        &mut self,
        index: u32,
        content_block: &StartBlock,
    ) -> Vec<AssistantMessageEvent> {
        let block = match content_block {
            StartBlock::Text { text } => Block::Text {
                index,
                text: text.clone().unwrap_or_default(),
            },
            StartBlock::Thinking {
                thinking,
                signature,
            } => Block::Thinking {
                index,
                text: thinking.clone().unwrap_or_default(),
                signature: signature.clone().unwrap_or_default(),
                redacted: false,
            },
            StartBlock::RedactedThinking { data } => Block::Thinking {
                index,
                text: "[Reasoning redacted]".to_owned(),
                signature: data.clone(),
                redacted: true,
            },
            StartBlock::ToolUse { id, name, input } => Block::ToolCall {
                index,
                id: id.clone(),
                name: name.clone(),
                arguments: input
                    .as_ref()
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default(),
                partial_json: String::new(),
            },
            // 未知块类型：不建立块，后续针对它的 delta/stop 自然被忽略。
            StartBlock::Other => return Vec::new(),
        };
        self.blocks.push(block);
        let content_index = (self.blocks.len() - 1) as u32;
        let partial = self.snapshot();
        match content_block {
            StartBlock::Text { .. } => vec![AssistantMessageEvent::TextStart {
                content_index,
                partial,
            }],
            StartBlock::Thinking { .. } | StartBlock::RedactedThinking { .. } => {
                vec![AssistantMessageEvent::ThinkingStart {
                    content_index,
                    partial,
                }]
            }
            StartBlock::ToolUse { .. } => vec![AssistantMessageEvent::ToolcallStart {
                content_index,
                partial,
            }],
            StartBlock::Other => Vec::new(),
        }
    }

    /// `content_block_delta`：按顶层 index 找到块并累加。
    fn apply_delta(&mut self, index: u32, delta: &DeltaPayload) -> Vec<AssistantMessageEvent> {
        let Some(position) = self.block_position(index) else {
            return Vec::new();
        };
        let content_index = position as u32;
        match delta {
            DeltaPayload::TextDelta { text } => {
                let applied = if let Block::Text { text: buffer, .. } = &mut self.blocks[position] {
                    buffer.push_str(text);
                    true
                } else {
                    false
                };
                if !applied {
                    return Vec::new();
                }
                vec![AssistantMessageEvent::TextDelta {
                    content_index,
                    delta: text.clone(),
                    partial: self.snapshot(),
                }]
            }
            DeltaPayload::ThinkingDelta { thinking } => {
                let applied =
                    if let Block::Thinking { text: buffer, .. } = &mut self.blocks[position] {
                        buffer.push_str(thinking);
                        true
                    } else {
                        false
                    };
                if !applied {
                    return Vec::new();
                }
                vec![AssistantMessageEvent::ThinkingDelta {
                    content_index,
                    delta: thinking.clone(),
                    partial: self.snapshot(),
                }]
            }
            DeltaPayload::InputJsonDelta { partial_json } => {
                let applied = if let Block::ToolCall {
                    arguments,
                    partial_json: buffer,
                    ..
                } = &mut self.blocks[position]
                {
                    buffer.push_str(partial_json);
                    *arguments = parse_arguments(buffer);
                    true
                } else {
                    false
                };
                if !applied {
                    return Vec::new();
                }
                vec![AssistantMessageEvent::ToolcallDelta {
                    content_index,
                    delta: partial_json.clone(),
                    partial: self.snapshot(),
                }]
            }
            DeltaPayload::SignatureDelta { signature } => {
                // 签名只做累加，不产生事件。
                if let Block::Thinking {
                    signature: buffer, ..
                } = &mut self.blocks[position]
                {
                    buffer.push_str(signature);
                }
                Vec::new()
            }
            DeltaPayload::Other => Vec::new(),
        }
    }

    /// `content_block_stop`：发对应 `*_end` 事件；工具调用在此定格参数。
    fn stop_block(&mut self, index: u32) -> Vec<AssistantMessageEvent> {
        let Some(position) = self.block_position(index) else {
            return Vec::new();
        };
        let content_index = position as u32;
        // 工具调用收尾：把累积的 JSON 片段解析成最终参数对象。
        if let Block::ToolCall {
            arguments,
            partial_json,
            ..
        } = &mut self.blocks[position]
        {
            *arguments = parse_arguments(partial_json);
        }
        match &self.blocks[position] {
            Block::Text { text, .. } => vec![AssistantMessageEvent::TextEnd {
                content_index,
                content: text.clone(),
                partial: self.snapshot(),
            }],
            Block::Thinking { text, .. } => vec![AssistantMessageEvent::ThinkingEnd {
                content_index,
                content: text.clone(),
                partial: self.snapshot(),
            }],
            Block::ToolCall {
                id,
                name,
                arguments,
                ..
            } => vec![AssistantMessageEvent::ToolcallEnd {
                content_index,
                tool_call: ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                    thought_signature: None,
                    namespace: None,
                },
                partial: self.snapshot(),
            }],
        }
    }

    /// `message_delta`：更新停止原因与用量。
    fn handle_message_delta(
        &mut self,
        delta: &MessageDeltaPayload,
        usage: Option<&AnthropicUsage>,
    ) {
        if let Some(reason) = &delta.stop_reason {
            self.raw_stop_reason = Some(reason.clone());
            let explanation = delta
                .stop_details
                .as_ref()
                .and_then(|details| details.explanation.as_deref());
            match map_stop_reason(reason, explanation) {
                Ok((stop_reason, error_message)) => {
                    self.stop_reason = stop_reason;
                    if let Some(message) = error_message {
                        self.error_message = Some(message);
                    }
                }
                Err(message) => self.set_error(message),
            }
        }
        // 只覆盖非空字段：代理可能省略 input_tokens，不能把它冲成 0。
        if let Some(usage) = usage {
            if let Some(value) = usage.input_tokens {
                self.usage.input = value;
            }
            if let Some(value) = usage.output_tokens {
                self.usage.output = value;
            }
            if let Some(value) = usage.cache_read_input_tokens {
                self.usage.cache_read = value;
            }
            if let Some(value) = usage.cache_creation_input_tokens {
                self.usage.cache_write = value;
            }
            if let Some(thinking) = usage
                .output_tokens_details
                .as_ref()
                .and_then(|details| details.thinking_tokens)
            {
                self.usage.reasoning = Some(thinking);
            }
        }
        self.recalc_usage();
    }

    /// 按各分量重算总量与费用（Anthropic 不给 total_tokens）。
    fn recalc_usage(&mut self) {
        self.usage.total_tokens =
            self.usage.input + self.usage.output + self.usage.cache_read + self.usage.cache_write;
        let cost = calculate_cost(&self.usage_model, &self.usage);
        self.usage.cost = cost;
    }

    /// 找到顶层 index 对应的块在 `blocks` 中的位置。
    fn block_position(&self, index: u32) -> Option<usize> {
        self.blocks.iter().position(|block| block.index() == index)
    }

    /// 标记流出错；`finish` 会据此发出 `error` 事件。
    pub fn set_error(&mut self, message: impl Into<String>) {
        self.error_message = Some(message.into());
        self.stop_reason = StopReason::Error;
    }

    /// 收尾：做完整性校验，然后发 `done` 或 `error`。
    pub fn finish(&mut self) -> Vec<AssistantMessageEvent> {
        // 已经有明确错误时不再覆盖它的错误信息。
        if self.stop_reason != StopReason::Error && self.saw_message_start && !self.saw_message_stop
        {
            self.set_error("Anthropic stream ended before message_stop");
        }
        if self.stop_reason == StopReason::Pending {
            self.set_error("Anthropic stream ended without a stop reason");
        }
        if self.stop_reason == StopReason::Error {
            return vec![AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: self.snapshot(),
            }];
        }
        vec![AssistantMessageEvent::Done {
            reason: self.stop_reason,
            message: self.snapshot(),
        }]
    }

    /// 当前累积状态快照（克隆成一条完整助手消息）。
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
        Block::Text { text, .. } => AssistantContent::Text(TextContent {
            text: text.clone(),
            text_signature: None,
        }),
        Block::Thinking {
            text,
            signature,
            redacted,
            ..
        } => AssistantContent::Thinking(ThinkingContent {
            thinking: text.clone(),
            thinking_signature: (!signature.is_empty()).then(|| signature.clone()),
            redacted: redacted.then_some(true),
        }),
        Block::ToolCall {
            id,
            name,
            arguments,
            ..
        } => AssistantContent::ToolCall(ToolCall {
            id: id.clone(),
            name: name.clone(),
            arguments: arguments.clone(),
            thought_signature: None,
            namespace: None,
        }),
    }
}

/// 解析流式累积的工具参数 JSON；解析失败时按空对象处理。
fn parse_arguments(arguments_json: &str) -> Map<String, Value> {
    match serde_json::from_str::<Value>(arguments_json) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    }
}

// -----------------------------------------------------------------------------
// SSE 解码与惰性事件流
// -----------------------------------------------------------------------------

/// 一个解码完成的 SSE 事件。
#[derive(Clone, Debug)]
struct SseEvent {
    /// `event:` 行的值；没有该行时为 `None`。
    event: Option<String>,
    /// 多行 `data:` 用 `\n` 拼接后的内容。
    data: String,
}

/// SSE 解码器的累积状态。
///
/// C++ 对照：一个小的状态机结构体，把「事件名 + 数据行」攒到空行为止。
#[derive(Clone, Debug, Default)]
struct SseState {
    event: Option<String>,
    data: Vec<String>,
}

/// 冲出一个事件并清空状态；没有任何内容时返回 `None`。
fn flush_sse_event(state: &mut SseState) -> Option<SseEvent> {
    if state.event.is_none() && state.data.is_empty() {
        return None;
    }
    let event = SseEvent {
        event: state.event.take(),
        data: state.data.join("\n"),
    };
    state.data.clear();
    Some(event)
}

/// 解码一行 SSE；只有空行才会冲出一个事件。
///
/// 规则与 SSE 规范一致：`:` 开头是注释；按第一个 `:` 切字段名，值去掉一个前导空格。
fn decode_sse_line(line: &str, state: &mut SseState) -> Option<SseEvent> {
    if line.is_empty() {
        return flush_sse_event(state);
    }
    if line.starts_with(':') {
        return None;
    }
    let (field, value) = match line.find(':') {
        Some(position) => {
            let value = &line[position + 1..];
            (&line[..position], value.strip_prefix(' ').unwrap_or(value))
        }
        None => (line, ""),
    };
    if field == "event" {
        state.event = Some(value.to_owned());
    } else if field == "data" {
        state.data.push(value.to_owned());
    }
    None
}

/// 惰性事件流：一边从 reader 读 SSE，一边产出助手消息事件。
///
/// C++ 对照：类似一个自制的流式迭代器类，把「读网络 + 解码 SSE + 状态机」封在一起。
pub struct AnthropicEventStream {
    reader: BufReader<Box<dyn Read>>,
    stream: AnthropicStream,
    /// 已解析出来、还没发出的事件队列（一个 SSE 事件可能产生多个事件）。
    pending: VecDeque<AssistantMessageEvent>,
    state: SseState,
    finished: bool,
}

impl AnthropicEventStream {
    /// 用模型和响应体 reader 创建事件流；首个事件是 `start`。
    #[must_use]
    pub fn new(model: Model, reader: Box<dyn Read>) -> Self {
        let stream = AnthropicStream::new(model);
        let mut pending = VecDeque::new();
        pending.push_back(stream.start_event());
        Self {
            reader: BufReader::new(reader),
            stream,
            pending,
            state: SseState::default(),
            finished: false,
        }
    }

    /// 发收尾事件（`done`/`error`），标记流结束；重复调用只生效一次。
    fn finish_stream(&mut self) {
        if self.finished {
            return;
        }
        self.pending.extend(self.stream.finish());
        self.finished = true;
    }

    /// 处理一个解码好的 SSE 事件：过滤事件名，解析载荷，喂给聚合器。 error 特判 → 无名跳过 → 白名单外跳过 → 解析失败报错
    fn dispatch_sse(&mut self, sse: &SseEvent) -> Vec<AssistantMessageEvent> {
        // 显式的 error 事件：data 原文就是错误消息，原版直接以它抛错。
        if sse.event.as_deref() == Some("error") {
            self.stream.set_error(sse.data.clone());
            self.finish_stream();
            return Vec::new();
        }
        let Some(name) = sse.event.as_deref() else {
            return Vec::new();
        };
        if !ANTHROPIC_MESSAGE_EVENTS.contains(&name) {
            return Vec::new();
        }
        match serde_json::from_str::<RawMessageStreamEvent>(&sse.data) {
            Ok(event) => self.stream.handle_event(&event),
            Err(error) => {
                self.stream.set_error(format!(
                    "Could not parse Anthropic SSE event {name}: {error}; data={}",
                    sse.data
                ));
                self.finish_stream();
                Vec::new()
            }
        }
    }
}

impl Iterator for AnthropicEventStream {
    type Item = AssistantMessageEvent;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // 先发队列里已有的事件。
            if let Some(event) = self.pending.pop_front() {
                return Some(event);
            }
            if self.finished {
                return None;
            }
            // 队列空了才读下一行，实现「按需读取」。
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    // 读到结尾：先冲出最后一个没被空行终止的事件，再收尾。
                    if let Some(sse) = flush_sse_event(&mut self.state) {
                        let events = self.dispatch_sse(&sse);
                        self.pending.extend(events);
                    }
                    self.finish_stream();
                }
                Ok(_) => {
                    let line = line.trim_end_matches(['\r', '\n']);
                    if let Some(sse) = decode_sse_line(line, &mut self.state) {
                        let events = self.dispatch_sse(&sse);
                        self.pending.extend(events);
                    }
                }
                Err(error) => {
                    self.stream.set_error(format!("读取响应流失败: {error}"));
                    self.finish_stream();
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Provider：把「构造请求 → 传输 → 聚合」串起来
// -----------------------------------------------------------------------------

/// 基于 Anthropic Messages API 的 Provider。
pub struct AnthropicMessagesProvider {
    transport: Box<dyn HttpTransport>,
    api_key: String,
}

impl AnthropicMessagesProvider {
    /// 用传输层和 API Key 创建 Provider。
    #[must_use]
    pub fn new(transport: Box<dyn HttpTransport>, api_key: impl Into<String>) -> Self {
        Self {
            transport,
            api_key: api_key.into(),
        }
    }

    /// 构造发给 `/v1/messages` 的 HTTP 请求。
    fn build_http_request(
        &self,
        model: &Model,
        context: &Context,
        options: &RequestOptions,
    ) -> Result<HttpRequest, HttpError> {
        let body = serde_json::to_string(&build_request(model, context, options))
            .map_err(|error| HttpError::new(format!("序列化请求失败: {error}")))?;
        let url = format!("{}/v1/messages", model.base_url.trim_end_matches('/'));
        let compat = resolve_anthropic_compat(model);

        let mut headers = vec![
            ("Content-Type".to_owned(), "application/json".to_owned()),
            ("x-api-key".to_owned(), self.api_key.clone()),
            ("anthropic-version".to_owned(), "2023-06-01".to_owned()),
            ("accept".to_owned(), "application/json".to_owned()),
        ];

        let mut beta_features: Vec<&str> = Vec::new();
        let has_tools = context
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty());
        // 不支持随到随流时必须显式声明细粒度工具流式 beta。
        if has_tools && !compat.supports_eager_tool_input_streaming {
            beta_features.push(FINE_GRAINED_TOOL_STREAMING_BETA);
        }
        // 默认开启交错思考；自适应思考模型自带该能力，跳过。
        if !compat.force_adaptive_thinking {
            beta_features.push(INTERLEAVED_THINKING_BETA);
        }
        if !beta_features.is_empty() {
            headers.push(("anthropic-beta".to_owned(), beta_features.join(",")));
        }

        if let Some(session_id) = &options.session_id {
            if compat.send_session_affinity_headers {
                headers.push(("x-session-affinity".to_owned(), session_id.clone()));
            }
        }

        Ok(HttpRequest { url, headers, body })
    }

    /// 发请求并返回惰性事件流；失败时返回只含一个 error 事件的流。
    fn run(
        &self,
        model: &Model,
        context: &Context,
        options: &RequestOptions,
    ) -> Box<dyn Iterator<Item = AssistantMessageEvent>> {
        let request = match self.build_http_request(model, context, options) {
            Ok(request) => request,
            Err(error) => return Box::new(std::iter::once(error_event(model, &error.message))),
        };
        match self.transport.post(&request) {
            Ok(reader) => Box::new(AnthropicEventStream::new(model.clone(), reader)),
            Err(error) => Box::new(std::iter::once(error_event(model, &error.message))),
        }
    }
}

impl Provider for AnthropicMessagesProvider {
    fn id(&self) -> &str {
        "anthropic-messages"
    }

    fn name(&self) -> &str {
        "Anthropic Messages API"
    }

    fn get_models(&self) -> &[Model] {
        &[]
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &RequestOptions,
    ) -> Box<dyn Iterator<Item = AssistantMessageEvent>> {
        self.run(model, context, options)
    }
}

/// 把一段完整的 SSE 响应体聚合成事件序列。
#[must_use]
pub fn aggregate_sse(model: &Model, body: &str) -> Vec<AssistantMessageEvent> {
    let reader = Box::new(std::io::Cursor::new(body.to_owned()));
    AnthropicEventStream::new(model.clone(), reader).collect()
}

/// 返回一个全 0 的用量，作为流开始时的初始值。
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
