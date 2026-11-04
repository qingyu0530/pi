//! Amazon Bedrock Converse Stream 协议。
//!
//! 和前三个协议的差别：
//! - 请求体是 Converse 的 JSON（messages/system/inferenceConfig/toolConfig）；
//! - 响应不是 SSE 文本，而是 **AWS event-stream 二进制分帧**：每帧 = 预奏
//!   （总长/头长/CRC）+ 头部（`:event-type` 等元数据）+ JSON 载荷 + 尾部 CRC。
//!
//! 认证只实现 **Bearer Token**（`AWS_BEARER_TOKEN_BEDROCK` / `--api-key`）；
//! SigV4 签名需要凭据派生与时间戳签名，暂未实现（文档见模块尾部说明）。
//!
//! 这是精简版：不处理加密思考回放（redactedContent）、adaptive thinking、
//! GovCloud 分支、请求元数据。

use std::collections::HashMap;
use std::io::Read;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::api::http::{HttpError, HttpRequest, HttpTransport};
use crate::content::{AssistantContent, ToolResultContent};
use crate::context::Context;
use crate::event::AssistantMessageEvent;
use crate::message::{
    AssistantMessage, AssistantRole, ConversationMessage, StopReason, Usage, UsageCost,
};
use crate::model::{Model, ModelCompat, calculate_cost};
use crate::provider::{Provider, RequestOptions};

// -----------------------------------------------------------------------------
// 请求方向
// -----------------------------------------------------------------------------

/// `POST /model/{modelId}/converse-stream` 的请求体。
#[derive(Clone, Debug, Serialize)]
pub struct ConverseStreamRequest {
    /// 模型 id（或推理配置 ARN）。
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// 对话消息。工具结果必须合并进 user 消息（Bedrock 的硬性要求）。
    pub messages: Vec<BedrockMessage>,
    /// 系统提示（Converse 用 `system` 数组，不是消息）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<BedrockSystemBlock>>,
    /// 推理配置（maxTokens / temperature）。
    #[serde(rename = "inferenceConfig", skip_serializing_if = "Option::is_none")]
    pub inference_config: Option<InferenceConfig>,
    /// 工具配置。
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<BedrockToolConfig>,
    /// 附加字段（Claude 的 thinkingConfig 放这里）。
    #[serde(
        rename = "additionalModelRequestFields",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_model_request_fields: Option<Value>,
}

/// 一条 Converse 消息：角色 + 内容块数组。
#[derive(Clone, Debug, Serialize)]
pub struct BedrockMessage {
    pub role: &'static str,
    pub content: Vec<BedrockContentBlock>,
}

/// 系统提示块。
#[derive(Clone, Debug, Serialize)]
pub struct BedrockSystemBlock {
    pub text: String,
}

/// 推理配置。
#[derive(Clone, Debug, Serialize)]
pub struct InferenceConfig {
    #[serde(rename = "maxTokens", skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

/// 工具配置。
#[derive(Clone, Debug, Serialize)]
pub struct BedrockToolConfig {
    pub tools: Vec<BedrockTool>,
}

/// 一个工具定义。
#[derive(Clone, Debug, Serialize)]
pub struct BedrockTool {
    #[serde(rename = "toolSpec")]
    pub tool_spec: BedrockToolSpec,
}

#[derive(Clone, Debug, Serialize)]
pub struct BedrockToolSpec {
    pub name: String,
    pub description: String,
    /// 参数 schema 包在 `inputSchema.json` 下。
    #[serde(rename = "inputSchema")]
    pub input_schema: BedrockToolSchema,
}

#[derive(Clone, Debug, Serialize)]
pub struct BedrockToolSchema {
    pub json: Value,
}

/// Converse 的内容块。各变体字段名不同，用 untagged 枚举序列化。
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum BedrockContentBlock {
    /// 文本。
    Text { text: String },
    /// 图片（Base64 源）。
    Image { image: BedrockImage },
    /// 助手的工具调用（回放历史）。
    ToolUse {
        #[serde(rename = "toolUse")]
        tool_use: BedrockToolUse,
    },
    /// 工具结果（在 user 消息里）。
    ToolResult {
        #[serde(rename = "toolResult")]
        tool_result: BedrockToolResult,
    },
    /// 思考内容（回放历史）。
    Reasoning {
        #[serde(rename = "reasoningContent")]
        reasoning: BedrockReasoningContent,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct BedrockImage {
    pub format: &'static str,
    pub source: BedrockImageSource,
}

#[derive(Clone, Debug, Serialize)]
pub struct BedrockImageSource {
    /// Base64 编码的图片字节。
    pub bytes: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct BedrockToolUse {
    #[serde(rename = "toolUseId")]
    pub tool_use_id: String,
    pub name: String,
    /// 参数对象。
    pub input: Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct BedrockToolResult {
    #[serde(rename = "toolUseId")]
    pub tool_use_id: String,
    pub content: Vec<BedrockToolResultContent>,
    /// `success` 或 `error`。
    pub status: &'static str,
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum BedrockToolResultContent {
    Text { text: String },
    Image { image: BedrockImage },
}

#[derive(Clone, Debug, Serialize)]
pub struct BedrockReasoningContent {
    #[serde(rename = "reasoningText")]
    pub reasoning_text: BedrockReasoningText,
}

#[derive(Clone, Debug, Serialize)]
pub struct BedrockReasoningText {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// 是否是 Bedrock 上的 Claude 模型。
fn is_claude_model(model: &Model) -> bool {
    model.id.contains("claude") || model.name.to_lowercase().contains("claude")
}

/// 解析后的兼容设置（精简版：只有 strict 工具一个开关）。
#[derive(Clone, Debug)]
pub struct ResolvedBedrockCompat {
    pub supports_strict_mode: bool,
}

/// 从模型解析兼容设置。
#[must_use]
pub fn resolve_bedrock_compat(model: &Model) -> ResolvedBedrockCompat {
    let mut supports_strict_mode = false;
    if let Some(ModelCompat::BedrockConverseStream(config)) = &model.compat {
        if let Some(value) = config.supports_strict_mode {
            supports_strict_mode = value;
        }
    }
    ResolvedBedrockCompat {
        supports_strict_mode,
    }
}

/// 默认思考预算（原版同值；xhigh/max 收敛到 high）。
fn thinking_budget(level: &crate::model::ModelThinkingLevel) -> Option<u64> {
    use crate::model::ModelThinkingLevel;
    match level {
        ModelThinkingLevel::Off => None,
        ModelThinkingLevel::Minimal => Some(1024),
        ModelThinkingLevel::Low => Some(2048),
        ModelThinkingLevel::Medium => Some(8192),
        ModelThinkingLevel::High | ModelThinkingLevel::Xhigh | ModelThinkingLevel::Max => {
            Some(16384)
        }
    }
}

/// 构造发给 `/model/{modelId}/converse-stream` 的请求。
#[must_use]
pub fn build_request(
    model: &Model,
    context: &Context,
    options: &RequestOptions,
) -> ConverseStreamRequest {
    let _compat = resolve_bedrock_compat(model);

    // 思考配置：只有 Claude 模型且指定了思考级别时发。
    let additional = if model.reasoning && is_claude_model(model) {
        options.reasoning_effort.and_then(|level| {
            // 自定义预算优先（xhigh/max 同样收敛到 high 档）。
            let custom = options
                .thinking_budgets
                .as_ref()
                .and_then(|budgets| match level {
                    crate::model::ModelThinkingLevel::Minimal => budgets.minimal,
                    crate::model::ModelThinkingLevel::Low => budgets.low,
                    crate::model::ModelThinkingLevel::Medium => budgets.medium,
                    _ => budgets.high,
                });
            let budget = custom.or_else(|| thinking_budget(&level))?;
            Some(serde_json::json!({
                "thinking": { "type": "enabled", "budget_tokens": budget }
            }))
        })
    } else {
        None
    };

    // maxTokens：显式选项优先；Claude 模型默认带模型上限，其它模型不带。
    let max_tokens = options
        .max_tokens
        .or_else(|| is_claude_model(model).then_some(model.max_tokens));

    ConverseStreamRequest {
        model_id: model.id.clone(),
        messages: convert_messages(context),
        system: context.system_prompt.as_ref().map(|prompt| {
            vec![BedrockSystemBlock {
                text: prompt.clone(),
            }]
        }),
        inference_config: Some(InferenceConfig {
            max_tokens,
            temperature: options.temperature,
        }),
        tool_config: context.tools.as_ref().map(|tools| BedrockToolConfig {
            tools: tools
                .iter()
                .map(|tool| BedrockTool {
                    tool_spec: BedrockToolSpec {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        input_schema: BedrockToolSchema {
                            json: tool.parameters.clone(),
                        },
                    },
                })
                .collect(),
        }),
        additional_model_request_fields: additional,
    }
}

/// 把对话上下文翻译成 Converse 消息。工具结果合并进 user 消息（Bedrock 要求）。
#[must_use]
pub fn convert_messages(context: &Context) -> Vec<BedrockMessage> {
    let mut result = Vec::new();

    let messages: Vec<&ConversationMessage> = context.messages.iter().collect();
    let mut index = 0;
    while index < messages.len() {
        match messages[index] {
            ConversationMessage::User(user) => {
                let content = user_content_blocks(&user.content);
                result.push(BedrockMessage {
                    role: "user",
                    content,
                });
                index += 1;
            }
            ConversationMessage::Assistant(assistant) => {
                let content: Vec<BedrockContentBlock> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Text(text) => {
                            non_blank(&text.text).map(|text| BedrockContentBlock::Text { text })
                        }
                        AssistantContent::ToolCall(call) => Some(BedrockContentBlock::ToolUse {
                            tool_use: BedrockToolUse {
                                tool_use_id: call.id.clone(),
                                name: call.name.clone(),
                                input: Value::Object(call.arguments.clone()),
                            },
                        }),
                        // 思考块回放（无签名时跳过，避免 Bedrock 拒绝）。
                        AssistantContent::Thinking(thinking) => {
                            let text = non_blank(&thinking.thinking)?;
                            Some(BedrockContentBlock::Reasoning {
                                reasoning: BedrockReasoningContent {
                                    reasoning_text: BedrockReasoningText {
                                        text,
                                        signature: thinking
                                            .thinking_signature
                                            .clone()
                                            .filter(|s| !s.trim().is_empty()),
                                    },
                                },
                            })
                        }
                    })
                    .collect();
                if !content.is_empty() {
                    result.push(BedrockMessage {
                        role: "assistant",
                        content,
                    });
                }
                index += 1;
            }
            ConversationMessage::ToolResult(_) => {
                // 连续的工具结果合并成一条 user 消息。
                let mut blocks = Vec::new();
                while let Some(ConversationMessage::ToolResult(result)) = messages.get(index) {
                    blocks.push(BedrockContentBlock::ToolResult {
                        tool_result: BedrockToolResult {
                            tool_use_id: result.tool_call_id.clone(),
                            content: tool_result_content(&result.content),
                            status: if result.is_error { "error" } else { "success" },
                        },
                    });
                    index += 1;
                }
                if !blocks.is_empty() {
                    result.push(BedrockMessage {
                        role: "user",
                        content: blocks,
                    });
                }
            }
        }
    }
    result
}

/// 用户消息的内容块。
fn user_content_blocks(content: &crate::message::UserMessageContent) -> Vec<BedrockContentBlock> {
    let blocks: Vec<BedrockContentBlock> = match content {
        crate::message::UserMessageContent::Text(text) => vec![BedrockContentBlock::Text {
            text: if text.trim().is_empty() {
                "<empty>".to_owned()
            } else {
                text.clone()
            },
        }],
        crate::message::UserMessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                crate::content::UserContent::Text(text) => {
                    non_blank(&text.text).map(|text| BedrockContentBlock::Text { text })
                }
                crate::content::UserContent::Image(image) => {
                    image_block(&image.mime_type, &image.data)
                        .map(|image| BedrockContentBlock::Image { image })
                }
            })
            .collect(),
    };
    // 空内容兜底（Bedrock 拒绝空 content 数组）。
    if blocks.is_empty() {
        vec![BedrockContentBlock::Text {
            text: "<empty>".to_owned(),
        }]
    } else {
        blocks
    }
}

/// 工具结果的内容块。
fn tool_result_content(content: &[ToolResultContent]) -> Vec<BedrockToolResultContent> {
    content
        .iter()
        .filter_map(|block| match block {
            ToolResultContent::Text(text) => Some(BedrockToolResultContent::Text {
                text: text.text.clone(),
            }),
            ToolResultContent::Image(image) => image_block(&image.mime_type, &image.data)
                .map(|image| BedrockToolResultContent::Image { image }),
        })
        .collect()
}

/// 构造图片块；不支持的格式返回 `None`（跳过）。
fn image_block(mime_type: &str, data: &str) -> Option<BedrockImage> {
    let format = match mime_type {
        "image/jpeg" | "image/jpg" => "jpeg",
        "image/png" => "png",
        "image/gif" => "gif",
        "image/webp" => "webp",
        _ => return None,
    };
    Some(BedrockImage {
        format,
        source: BedrockImageSource {
            bytes: data.to_owned(),
        },
    })
}

/// 空白文本返回 `None`（Bedrock 跳过空白块）。
fn non_blank(text: &str) -> Option<String> {
    if text.trim().is_empty() {
        None
    } else {
        Some(text.to_owned())
    }
}

// -----------------------------------------------------------------------------
// AWS event-stream 二进制分帧解析
// -----------------------------------------------------------------------------
// 帧结构（大端）：
//   [总长 u32][头长 u32][预奏 CRC u32][头部字节][载荷字节][消息 CRC u32]
// 头部是一串 [名字长 u8][名字][值类型 u8][值字节]；元数据放 `:message-type`、
// `:event-type` 两个字符串头里，载荷是事件的 JSON。

/// 一帧解析结果：事件类型 + JSON 载荷。
#[derive(Clone, Debug)]
pub struct EventStreamFrame {
    pub event_type: String,
    pub payload: Value,
}

/// 从 reader 逐帧读取的迭代器（保持惰性，配合真流式）。
pub struct EventFrames<R: Read> {
    reader: R,
    done: bool,
}

impl<R: Read> EventFrames<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            done: false,
        }
    }
}

impl<R: Read> Iterator for EventFrames<R> {
    type Item = Result<EventStreamFrame, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        // 预奏 12 字节。读到 0 字节表示流正常结束。
        let mut prelude = [0u8; 12];
        match read_exact_or_eof(&mut self.reader, &mut prelude) {
            Ok(false) => {
                self.done = true;
                return None;
            }
            Ok(true) => {}
            Err(error) => {
                self.done = true;
                return Some(Err(format!("读取事件帧失败: {error}")));
            }
        }
        match parse_frame(&prelude, &mut self.reader) {
            Ok(frame) => Some(Ok(frame)),
            Err(error) => {
                self.done = true;
                Some(Err(error))
            }
        }
    }
}

/// 读满 buffer；返回 false 表示正好 EOF（干净收尾）。
fn read_exact_or_eof(reader: &mut impl Read, buffer: &mut [u8]) -> std::io::Result<bool> {
    let mut filled = 0;
    while filled < buffer.len() {
        let n = reader.read(&mut buffer[filled..])?;
        if n == 0 {
            if filled == 0 {
                return Ok(false);
            }
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "事件流在帧中间截断",
            ));
        }
        filled += n;
    }
    Ok(true)
}

/// 读并解析一帧（预奏已读到 `prelude`）。
fn parse_frame(prelude: &[u8; 12], reader: &mut impl Read) -> Result<EventStreamFrame, String> {
    let total_len = u32::from_be_bytes([prelude[0], prelude[1], prelude[2], prelude[3]]) as usize;
    let headers_len = u32::from_be_bytes([prelude[4], prelude[5], prelude[6], prelude[7]]) as usize;
    let prelude_crc = u32::from_be_bytes([prelude[8], prelude[9], prelude[10], prelude[11]]);

    // 预奏 CRC 覆盖前 8 字节。
    if crc32(&prelude[..8]) != prelude_crc {
        return Err("事件帧预奏 CRC 校验失败".to_owned());
    }
    if total_len < 16 || headers_len + 12 > total_len {
        return Err(format!(
            "事件帧长度非法: total={total_len} headers={headers_len}"
        ));
    }

    let payload_len = total_len - 12 - headers_len - 4;
    let mut headers = vec![0u8; headers_len];
    let mut payload = vec![0u8; payload_len];
    let mut message_crc_bytes = [0u8; 4];
    reader
        .read_exact(&mut headers)
        .map_err(|error| format!("读取事件头失败: {error}"))?;
    reader
        .read_exact(&mut payload)
        .map_err(|error| format!("读取事件载荷失败: {error}"))?;
    reader
        .read_exact(&mut message_crc_bytes)
        .map_err(|error| format!("读取事件 CRC 失败: {error}"))?;

    // 消息 CRC 覆盖除尾部 4 字节外的整帧。
    let mut whole = prelude.to_vec();
    whole.extend_from_slice(&headers);
    whole.extend_from_slice(&payload);
    if crc32(&whole) != u32::from_be_bytes(message_crc_bytes) {
        return Err("事件帧消息 CRC 校验失败".to_owned());
    }

    let event_type = parse_event_type(&headers)?;
    let payload_json = if payload.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&payload)
            .map_err(|error| format!("解析事件载荷 JSON 失败: {error}"))?
    };
    Ok(EventStreamFrame {
        event_type,
        payload: payload_json,
    })
}

/// 从头部字节里取 `:event-type` 字符串值。
fn parse_event_type(headers: &[u8]) -> Result<String, String> {
    let mut offset = 0;
    while offset < headers.len() {
        let name_len = headers[offset] as usize;
        offset += 1;
        if offset + name_len + 1 > headers.len() {
            return Err("事件头长度越界".to_owned());
        }
        let name = std::str::from_utf8(&headers[offset..offset + name_len])
            .map_err(|_| "事件头名字不是 UTF-8".to_owned())?
            .to_owned();
        offset += name_len;
        let value_type = headers[offset];
        offset += 1;
        let value_len = match value_type {
            // bool true / false：无载荷。
            0 | 1 => 0,
            // byte / short / int / long / timestamp：定长。
            2 => 1,
            3 => 2,
            4 | 8 => 4,
            5 => 8,
            // bytes / string：2 字节长度前缀。
            6 | 7 => {
                if offset + 2 > headers.len() {
                    return Err("事件头变长值越界".to_owned());
                }
                let len = u16::from_be_bytes([headers[offset], headers[offset + 1]]) as usize;
                offset += 2;
                len
            }
            // uuid。
            9 => 16,
            other => return Err(format!("未知事件头值类型: {other}")),
        };
        if offset + value_len > headers.len() {
            return Err("事件头值长度越界".to_owned());
        }
        if name == ":event-type" {
            return headers[offset..offset + value_len]
                .to_vec()
                .try_into()
                .map_err(|_| "事件类型不是 UTF-8".to_owned());
        }
        offset += value_len;
    }
    Err("事件头缺少 :event-type".to_owned())
}

/// CRC32（IEEE 802.3 反射多项式），逐位计算，无查表。
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

// -----------------------------------------------------------------------------
// 流式聚合
// -----------------------------------------------------------------------------

/// 累积中的一个内容块。
#[derive(Clone, Debug)]
enum Block {
    Text(String),
    Thinking {
        text: String,
        signature: String,
    },
    ToolCall {
        tool_use_id: String,
        name: String,
        arguments_json: String,
    },
}

/// 把一串 Converse Stream 事件聚合成完整助手消息与流式事件。
pub struct BedrockStream {
    model: Model,
    usage: Usage,
    stop_reason: StopReason,
    raw_stop_reason: Option<String>,
    error_message: Option<String>,
    /// 内容块列表，顺序即最终顺序。
    blocks: Vec<Block>,
    /// contentBlockIndex -> 块下标（文本/思考块常在 delta 时才创建）。
    slots: HashMap<u32, usize>,
    /// 已收尾（contentBlockStop）的块。
    closed: Vec<bool>,
    timestamp: u64,
}

impl BedrockStream {
    /// 为指定模型创建聚合器。
    #[must_use]
    pub fn new(model: Model) -> Self {
        Self {
            model,
            usage: zero_usage(),
            stop_reason: StopReason::Pending,
            raw_stop_reason: None,
            error_message: None,
            blocks: Vec::new(),
            slots: HashMap::new(),
            closed: Vec::new(),
            timestamp: now_millis(),
        }
    }

    /// 流开始的 `start` 事件。
    #[must_use]
    pub fn start_event(&self) -> AssistantMessageEvent {
        AssistantMessageEvent::Start {
            partial: self.snapshot(),
        }
    }

    /// 处理一帧事件，返回产生的中间事件。
    pub fn handle_frame(&mut self, frame: &EventStreamFrame) -> Vec<AssistantMessageEvent> {
        let mut events = Vec::new();
        match frame.event_type.as_str() {
            "contentBlockStart" => self.handle_block_start(&frame.payload, &mut events),
            "contentBlockDelta" => self.handle_block_delta(&frame.payload, &mut events),
            "contentBlockStop" => self.handle_block_stop(&frame.payload, &mut events),
            "messageStop" => {
                self.raw_stop_reason = frame.payload["stopReason"].as_str().map(str::to_owned);
                let (reason, error) = map_stop_reason(self.raw_stop_reason.as_deref());
                self.stop_reason = reason;
                if let Some(error) = error {
                    self.error_message = Some(error);
                }
            }
            "metadata" => self.handle_metadata(&frame.payload),
            // 异常事件：转成错误信息。
            "internalServerException"
            | "modelStreamErrorException"
            | "validationException"
            | "throttlingException"
            | "serviceUnavailableException" => {
                let message = frame.payload["message"]
                    .as_str()
                    .unwrap_or("Bedrock 服务端错误")
                    .to_owned();
                self.error_message = Some(message);
            }
            // messageStart / metrics / 未知事件：忽略。
            _ => {}
        }
        events
    }

    /// contentBlockStart：工具调用块在这里声明（文本/思考块不发 start）。
    fn handle_block_start(&mut self, payload: &Value, events: &mut Vec<AssistantMessageEvent>) {
        let Some(index) = payload["contentBlockIndex"].as_u64() else {
            return;
        };
        let index = index as u32;
        let Some(tool_use) = payload["start"]["toolUse"].as_object() else {
            return;
        };
        let block_index = self.blocks.len();
        self.blocks.push(Block::ToolCall {
            tool_use_id: tool_use
                .get("toolUseId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            name: tool_use
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            arguments_json: String::new(),
        });
        self.closed.push(false);
        self.slots.insert(index, block_index);
        events.push(AssistantMessageEvent::ToolcallStart {
            content_index: block_index as u32,
            partial: self.snapshot(),
        });
    }

    /// contentBlockDelta：文本/思考的增量（块按需创建），工具参数增量。
    fn handle_block_delta(&mut self, payload: &Value, events: &mut Vec<AssistantMessageEvent>) {
        let Some(block_index_key) = payload["contentBlockIndex"].as_u64() else {
            return;
        };
        let index = block_index_key as u32;
        let delta = &payload["delta"];

        if let Some(text) = delta["text"].as_str() {
            let slot = self.ensure_slot(index, Kind::Text, events);
            if let Block::Text(buffer) = &mut self.blocks[slot] {
                buffer.push_str(text);
                events.push(AssistantMessageEvent::TextDelta {
                    content_index: slot as u32,
                    delta: text.to_owned(),
                    partial: self.snapshot(),
                });
            }
        } else if let Some(input) = delta["toolUse"]["input"].as_str() {
            if let Some(&slot) = self.slots.get(&index) {
                if let Block::ToolCall { arguments_json, .. } = &mut self.blocks[slot] {
                    arguments_json.push_str(input);
                    events.push(AssistantMessageEvent::ToolcallDelta {
                        content_index: slot as u32,
                        delta: input.to_owned(),
                        partial: self.snapshot(),
                    });
                }
            }
        } else if let Some(reasoning) = delta["reasoningContent"].as_object() {
            let slot = self.ensure_slot(index, Kind::Thinking, events);
            // 先取出增量（借用 payload），再更新块，最后才做快照，避免借用冲突。
            let delta_text = reasoning
                .get("text")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let delta_signature = reasoning
                .get("signature")
                .and_then(Value::as_str)
                .map(str::to_owned);
            if let Block::Thinking { text, signature } = &mut self.blocks[slot] {
                if let Some(delta_text) = &delta_text {
                    text.push_str(delta_text);
                }
                if let Some(delta_signature) = &delta_signature {
                    signature.push_str(delta_signature);
                }
            }
            if let Some(delta_text) = delta_text {
                let partial = self.snapshot();
                events.push(AssistantMessageEvent::ThinkingDelta {
                    content_index: slot as u32,
                    delta: delta_text,
                    partial,
                });
            }
        }
    }

    /// contentBlockStop：收尾对应块并发 end 事件。
    fn handle_block_stop(&mut self, payload: &Value, events: &mut Vec<AssistantMessageEvent>) {
        let Some(block_index_key) = payload["contentBlockIndex"].as_u64() else {
            return;
        };
        let index = block_index_key as u32;
        let Some(&slot) = self.slots.get(&index) else {
            return;
        };
        match &self.blocks[slot] {
            Block::Text(text) => events.push(AssistantMessageEvent::TextEnd {
                content_index: slot as u32,
                content: text.clone(),
                partial: self.snapshot(),
            }),
            Block::Thinking { text, .. } => {
                events.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: slot as u32,
                    content: text.clone(),
                    partial: self.snapshot(),
                });
            }
            Block::ToolCall { .. } => {
                let (tool_use_id, name, arguments) = match &self.blocks[slot] {
                    Block::ToolCall {
                        tool_use_id,
                        name,
                        arguments_json,
                    } => (
                        tool_use_id.clone(),
                        name.clone(),
                        parse_arguments(arguments_json),
                    ),
                    _ => unreachable!("刚匹配过 ToolCall"),
                };
                events.push(AssistantMessageEvent::ToolcallEnd {
                    content_index: slot as u32,
                    tool_call: crate::content::ToolCall {
                        id: tool_use_id,
                        name,
                        arguments,
                        thought_signature: None,
                        namespace: None,
                    },
                    partial: self.snapshot(),
                });
            }
        }
        if let Some(flag) = self.closed.get_mut(slot) {
            *flag = true;
        }
        self.slots.remove(&index);
    }

    /// metadata：token 用量 + 费用。
    fn handle_metadata(&mut self, payload: &Value) {
        let Some(usage) = payload["usage"].as_object() else {
            return;
        };
        let input = usage
            .get("inputTokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let output = usage
            .get("outputTokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let cache_read = usage
            .get("cacheReadInputTokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let cache_write = usage
            .get("cacheWriteInputTokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let total = usage
            .get("totalTokens")
            .and_then(Value::as_u64)
            .unwrap_or(input + output);
        self.usage.input = input;
        self.usage.output = output;
        self.usage.cache_read = cache_read;
        self.usage.cache_write = cache_write;
        self.usage.total_tokens = total;
        self.usage.cost = calculate_cost(&self.model, &self.usage);
    }

    /// 收尾：为未关闭的块发 end 事件，最后发 `done` 或 `error`。
    pub fn finish(&mut self) -> Vec<AssistantMessageEvent> {
        let mut events = Vec::new();
        for slot in 0..self.blocks.len() {
            if self.closed.get(slot).copied().unwrap_or(false) {
                continue;
            }
            match &self.blocks[slot] {
                Block::Text(text) => events.push(AssistantMessageEvent::TextEnd {
                    content_index: slot as u32,
                    content: text.clone(),
                    partial: self.snapshot(),
                }),
                Block::Thinking { text, .. } => events.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: slot as u32,
                    content: text.clone(),
                    partial: self.snapshot(),
                }),
                Block::ToolCall { .. } => {
                    let (tool_use_id, name, arguments) = match &self.blocks[slot] {
                        Block::ToolCall {
                            tool_use_id,
                            name,
                            arguments_json,
                        } => (
                            tool_use_id.clone(),
                            name.clone(),
                            parse_arguments(arguments_json),
                        ),
                        _ => unreachable!(),
                    };
                    events.push(AssistantMessageEvent::ToolcallEnd {
                        content_index: slot as u32,
                        tool_call: crate::content::ToolCall {
                            id: tool_use_id,
                            name,
                            arguments,
                            thought_signature: None,
                            namespace: None,
                        },
                        partial: self.snapshot(),
                    });
                }
            }
            if let Some(flag) = self.closed.get_mut(slot) {
                *flag = true;
            }
        }

        if self.stop_reason == StopReason::Pending {
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

    /// 按需创建文本/思考块（Bedrock 对它们不发 contentBlockStart）。
    fn ensure_slot(
        &mut self,
        index: u32,
        kind: Kind,
        events: &mut Vec<AssistantMessageEvent>,
    ) -> usize {
        if let Some(&slot) = self.slots.get(&index) {
            return slot;
        }
        let slot = self.blocks.len();
        self.blocks.push(match kind {
            Kind::Text => Block::Text(String::new()),
            Kind::Thinking => Block::Thinking {
                text: String::new(),
                signature: String::new(),
            },
        });
        self.closed.push(false);
        self.slots.insert(index, slot);
        let event = match kind {
            Kind::Text => AssistantMessageEvent::TextStart {
                content_index: slot as u32,
                partial: self.snapshot(),
            },
            Kind::Thinking => AssistantMessageEvent::ThinkingStart {
                content_index: slot as u32,
                partial: self.snapshot(),
            },
        };
        events.push(event);
        slot
    }

    /// 当前累积状态快照。
    #[must_use]
    pub fn snapshot(&self) -> AssistantMessage {
        let content: Vec<AssistantContent> = self
            .blocks
            .iter()
            .map(|block| match block {
                Block::Text(text) => AssistantContent::Text(crate::content::TextContent {
                    text: text.clone(),
                    text_signature: None,
                }),
                Block::Thinking { text, signature } => {
                    AssistantContent::Thinking(crate::content::ThinkingContent {
                        thinking: text.clone(),
                        thinking_signature: if signature.is_empty() {
                            None
                        } else {
                            Some(signature.clone())
                        },
                        redacted: None,
                    })
                }
                Block::ToolCall {
                    tool_use_id,
                    name,
                    arguments_json,
                } => AssistantContent::ToolCall(crate::content::ToolCall {
                    id: tool_use_id.clone(),
                    name: name.clone(),
                    arguments: parse_arguments(arguments_json),
                    thought_signature: None,
                    namespace: None,
                }),
            })
            .collect();

        AssistantMessage {
            role: AssistantRole::Assistant,
            content,
            api: self.model.api.clone(),
            provider: self.model.provider.clone(),
            model: self.model.id.clone(),
            response_model: None,
            response_id: None,
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

/// 按需建块的种类。
#[derive(Clone, Copy)]
enum Kind {
    Text,
    Thinking,
}

/// Bedrock 停止原因映射。
fn map_stop_reason(reason: Option<&str>) -> (StopReason, Option<String>) {
    match reason {
        Some("end_turn") | Some("stop_sequence") => (StopReason::Stop, None),
        Some("max_tokens") | Some("model_context_window_exceeded") => (StopReason::Length, None),
        Some("tool_use") => (StopReason::ToolUse, None),
        Some(other) => (
            StopReason::Error,
            Some(format!("Provider stopped with: {other}")),
        ),
        None => (StopReason::Error, None),
    }
}

/// 解析工具调用参数 JSON 字符串；失败返回空对象。
fn parse_arguments(arguments_json: &str) -> Map<String, Value> {
    serde_json::from_str(arguments_json).unwrap_or_default()
}

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

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

// -----------------------------------------------------------------------------
// Provider
// -----------------------------------------------------------------------------

/// 基于 Bedrock Converse Stream 的 Provider（Bearer Token 认证）。
pub struct BedrockProvider {
    transport: Box<dyn HttpTransport>,
    bearer_token: String,
}

impl BedrockProvider {
    /// 用传输层和 Bearer Token 创建 Provider。
    ///
    /// Token 来自 `--api-key` 或 `AWS_BEARER_TOKEN_BEDROCK`。
    /// SigV4（AWS 凭据签名）暂未实现。
    #[must_use]
    pub fn new(transport: Box<dyn HttpTransport>, bearer_token: impl Into<String>) -> Self {
        Self {
            transport,
            bearer_token: bearer_token.into(),
        }
    }

    /// 构造 HTTP 请求。
    fn build_http_request(
        &self,
        model: &Model,
        context: &Context,
        options: &RequestOptions,
    ) -> Result<HttpRequest, HttpError> {
        let body = serde_json::to_string(&build_request(model, context, options))
            .map_err(|error| HttpError::new(format!("序列化请求失败: {error}")))?;
        // 模型 id 里的 `/`（推理配置 ARN）会破坏路径，做百分号编码；`:` 合法保留。
        let encoded_model = model.id.replace('/', "%2F");
        let url = format!(
            "{}/model/{encoded_model}/converse-stream",
            model.base_url.trim_end_matches('/')
        );
        let headers = vec![
            ("Content-Type".to_owned(), "application/json".to_owned()),
            (
                "Authorization".to_owned(),
                format!("Bearer {}", self.bearer_token),
            ),
            (
                "Accept".to_owned(),
                "application/vnd.amazon.eventstream".to_owned(),
            ),
        ];
        Ok(HttpRequest { url, headers, body })
    }

    /// 发请求并返回惰性事件流；失败时返回单个 error 事件的流。
    fn run(
        &self,
        model: &Model,
        context: &Context,
        options: &RequestOptions,
    ) -> Box<dyn Iterator<Item = AssistantMessageEvent>> {
        let request = match self.build_http_request(model, context, options) {
            Ok(request) => request,
            Err(error) => {
                return Box::new(std::iter::once(error_event(model, &error.message)));
            }
        };
        match self.transport.post(&request) {
            Ok(reader) => Box::new(BedrockEventStream::new(model.clone(), reader)),
            Err(error) => Box::new(std::iter::once(error_event(model, &error.message))),
        }
    }
}

impl Provider for BedrockProvider {
    fn id(&self) -> &str {
        "bedrock-converse-stream"
    }

    fn name(&self) -> &str {
        "Amazon Bedrock Converse Stream"
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

/// 惰性事件流：边读二进制帧边聚合。
pub struct BedrockEventStream {
    frames: EventFrames<Box<dyn Read>>,
    stream: BedrockStream,
    pending: std::collections::VecDeque<AssistantMessageEvent>,
    finished: bool,
}

impl BedrockEventStream {
    /// 用模型和响应体 reader 创建事件流；首个事件是 `start`。
    #[must_use]
    pub fn new(model: Model, reader: Box<dyn Read>) -> Self {
        let stream = BedrockStream::new(model);
        let mut pending = std::collections::VecDeque::new();
        pending.push_back(stream.start_event());
        Self {
            frames: EventFrames::new(reader),
            stream,
            pending,
            finished: false,
        }
    }
}

impl Iterator for BedrockEventStream {
    type Item = AssistantMessageEvent;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(event);
            }
            if self.finished {
                return None;
            }
            match self.frames.next() {
                None => {
                    let events = self.stream.finish();
                    self.pending.extend(events);
                    self.finished = true;
                }
                Some(Ok(frame)) => {
                    let events = self.stream.handle_frame(&frame);
                    self.pending.extend(events);
                }
                Some(Err(error)) => {
                    self.stream.error_message = Some(error);
                    self.stream.stop_reason = StopReason::Error;
                    let events = self.stream.finish();
                    self.pending.extend(events);
                    self.finished = true;
                }
            }
        }
    }
}

/// 构造错误事件。
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
