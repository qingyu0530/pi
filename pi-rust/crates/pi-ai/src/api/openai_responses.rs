//! OpenAI Responses API 协议（GPT-5 系列）。
//!
//! 和 Chat Completions 的差别：消息不再是 `messages` 数组，而是 `input`
//! 项列表（message / function_call / function_call_output 各自成项）；
//! 流式事件也不是 chunk，而是按 `output_index` 定位的一组 `response.*` 事件。
//!
//! 这是精简版：不处理文本签名、加密思考回放、自定义工具输入、服务分级计价。

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::api::http::{HttpError, HttpRequest, HttpTransport};
use crate::content::{
    AssistantContent, TextContent, ThinkingContent, ToolCall, ToolResultContent, UserContent,
};
use crate::context::Context;
use crate::event::AssistantMessageEvent;
use crate::message::{
    AssistantMessage, AssistantRole, ConversationMessage, StopReason, Usage, UsageCost,
};
use crate::model::{Model, ModelCompat, ModelThinkingLevel, calculate_cost};
use crate::provider::{Provider, RequestOptions};

// -----------------------------------------------------------------------------
// 请求方向
// -----------------------------------------------------------------------------

/// `POST /responses` 的请求体。
#[derive(Clone, Debug, Serialize)]
pub struct ResponsesRequest {
    /// 要使用的模型 id。
    pub model: String,
    /// 输入项列表：角色消息、助手消息、工具调用、工具结果。
    pub input: Vec<ResponsesInputItem>,
    /// 是否流式返回。
    pub stream: bool,
    /// 最多生成多少 token（Responses API 的字段名）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    /// 采样温度。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// 推理配置（effort + summary）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponsesReasoning>,
    /// 随回复返回的内容类型（如加密思考）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    /// 工具定义。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,
    /// 不存储本次请求。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
}

/// 推理配置。
#[derive(Clone, Debug, Serialize)]
pub struct ResponsesReasoning {
    /// 推理强度。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// 思考摘要模式。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// 一个输入项。四种形状字段不同，用 untagged 枚举各自序列化。
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum ResponsesInputItem {
    /// 系统提示（system 或 developer 角色）。
    System { role: &'static str, content: String },
    /// 用户消息（字符串正文）。
    User { role: &'static str, content: String },
    /// 用户消息（内容块数组，可含图片）。
    UserBlocks {
        role: &'static str,
        content: Vec<ResponsesInputContent>,
    },
    /// 助手文本消息（回放历史）。
    Assistant {
        #[serde(rename = "type")]
        kind: &'static str,
        role: &'static str,
        content: Vec<ResponsesOutputText>,
        status: &'static str,
    },
    /// 工具调用（回放历史）。
    FunctionCall {
        #[serde(rename = "type")]
        kind: &'static str,
        #[serde(rename = "call_id")]
        call_id: String,
        name: String,
        arguments: String,
    },
    /// 工具结果。
    FunctionCallOutput {
        #[serde(rename = "type")]
        kind: &'static str,
        #[serde(rename = "call_id")]
        call_id: String,
        output: String,
    },
}

/// 用户消息的内容块。
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum ResponsesInputContent {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_image")]
    InputImage {
        image_url: String,
        detail: &'static str,
    },
}

/// 助手消息的文本块（回放历史时用）。
#[derive(Clone, Debug, Serialize)]
pub struct ResponsesOutputText {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
    pub annotations: Vec<Value>,
}

/// 工具定义（Responses API 的 function 工具是扁平结构）。
#[derive(Clone, Debug, Serialize)]
pub struct ResponsesTool {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub name: String,
    pub description: String,
    pub parameters: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// 解析后的兼容设置（精简版）。
#[derive(Clone, Debug)]
pub struct ResolvedResponsesCompat {
    pub supports_developer_role: bool,
    pub supports_strict_mode: bool,
}

/// 从模型解析兼容设置：默认值 + 显式覆盖。
#[must_use]
pub fn resolve_responses_compat(model: &Model) -> ResolvedResponsesCompat {
    let mut compat = ResolvedResponsesCompat {
        supports_developer_role: true,
        supports_strict_mode: true,
    };
    if let Some(ModelCompat::OpenaiResponses(config)) = &model.compat {
        if let Some(value) = config.supports_developer_role {
            compat.supports_developer_role = value;
        }
        if let Some(value) = config.supports_strict_mode {
            compat.supports_strict_mode = value;
        }
    }
    compat
}

/// Responses API 的最小输出 token 数。
const MIN_OUTPUT_TOKENS: u64 = 16;

/// 构造发给 `/responses` 的请求。
#[must_use]
pub fn build_request(
    model: &Model,
    context: &Context,
    options: &RequestOptions,
) -> ResponsesRequest {
    let compat = resolve_responses_compat(model);
    let input = convert_responses_messages(model, context, &compat);
    let tools = context.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|tool| ResponsesTool {
                kind: "function",
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
                strict: compat.supports_strict_mode.then_some(false),
            })
            .collect()
    });

    // 推理配置：有 effort 就带摘要；off 未被禁用时发 off 取值。
    let reasoning = if model.reasoning {
        if let Some(level) = options.reasoning_effort {
            let effort = mapped_effort(model, level);
            Some(ResponsesReasoning {
                effort: Some(effort),
                summary: Some("auto".to_owned()),
            })
        } else {
            let off_disabled = model
                .thinking_level_map
                .as_ref()
                .is_some_and(|map| matches!(map.get(&ModelThinkingLevel::Off), Some(None)));
            if off_disabled {
                None
            } else {
                Some(ResponsesReasoning {
                    effort: Some(off_effort(model)),
                    summary: None,
                })
            }
        }
    } else {
        None
    };

    ResponsesRequest {
        model: model.id.clone(),
        input,
        stream: true,
        // 上限至少 16（原版 OPENAI_RESPONSES_MIN_OUTPUT_TOKENS）。
        max_output_tokens: options
            .max_tokens
            .or(Some(model.max_tokens))
            .map(|value| value.max(MIN_OUTPUT_TOKENS)),
        temperature: options.temperature,
        include: reasoning
            .is_some()
            .then(|| vec!["reasoning.encrypted_content".to_owned()]),
        reasoning,
        tools,
        store: Some(false),
    }
}

/// 把对话上下文翻译成 Responses API 的输入项列表。
#[must_use]
pub fn convert_responses_messages(
    model: &Model,
    context: &Context,
    compat: &ResolvedResponsesCompat,
) -> Vec<ResponsesInputItem> {
    let mut items = Vec::new();

    // 系统提示：推理模型用 developer 角色（默认支持）。
    if let Some(prompt) = &context.system_prompt {
        let role = if model.reasoning && compat.supports_developer_role {
            "developer"
        } else {
            "system"
        };
        items.push(ResponsesInputItem::System {
            role,
            content: prompt.clone(),
        });
    }

    for message in &context.messages {
        match message {
            ConversationMessage::User(user) => match &user.content {
                crate::message::UserMessageContent::Text(text) => {
                    items.push(ResponsesInputItem::User {
                        role: "user",
                        content: text.clone(),
                    });
                }
                crate::message::UserMessageContent::Blocks(blocks) => {
                    let content: Vec<ResponsesInputContent> = blocks
                        .iter()
                        .map(|block| match block {
                            UserContent::Text(text) => ResponsesInputContent::InputText {
                                text: text.text.clone(),
                            },
                            UserContent::Image(image) => ResponsesInputContent::InputImage {
                                image_url: format!(
                                    "data:{};base64,{}",
                                    image.mime_type, image.data
                                ),
                                detail: "auto",
                            },
                        })
                        .collect();
                    if !content.is_empty() {
                        items.push(ResponsesInputItem::UserBlocks {
                            role: "user",
                            content,
                        });
                    }
                }
            },
            ConversationMessage::Assistant(assistant) => {
                for block in &assistant.content {
                    match block {
                        AssistantContent::Text(text) => {
                            items.push(ResponsesInputItem::Assistant {
                                kind: "message",
                                role: "assistant",
                                content: vec![ResponsesOutputText {
                                    kind: "output_text",
                                    text: text.text.clone(),
                                    annotations: Vec::new(),
                                }],
                                status: "completed",
                            });
                        }
                        // 思考块没有签名可回放，精简版跳过。
                        AssistantContent::Thinking(_) => {}
                        AssistantContent::ToolCall(call) => {
                            items.push(ResponsesInputItem::FunctionCall {
                                kind: "function_call",
                                call_id: call.id.clone(),
                                name: call.name.clone(),
                                arguments: Value::Object(call.arguments.clone()).to_string(),
                            });
                        }
                    }
                }
            }
            ConversationMessage::ToolResult(result) => {
                let output: String = result
                    .content
                    .iter()
                    .map(|block| match block {
                        ToolResultContent::Text(text) => text.text.clone(),
                        ToolResultContent::Image(image) => {
                            format!("[image: {}]", image.mime_type)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                items.push(ResponsesInputItem::FunctionCallOutput {
                    kind: "function_call_output",
                    call_id: result.tool_call_id.clone(),
                    output,
                });
            }
        }
    }
    items
}

/// 取某个思考级别对应的服务端取值：模型显式映射优先，否则用级别名字。
fn mapped_effort(model: &Model, level: ModelThinkingLevel) -> String {
    model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.get(&level))
        .and_then(|value| value.clone())
        .unwrap_or_else(|| level_name(level).to_owned())
}

/// off 级别对应的取值；没有映射时用 "none"。
fn off_effort(model: &Model) -> String {
    model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.get(&ModelThinkingLevel::Off))
        .and_then(|value| value.clone())
        .unwrap_or_else(|| "none".to_owned())
}

/// 思考级别对应的字符串。
fn level_name(level: ModelThinkingLevel) -> &'static str {
    match level {
        ModelThinkingLevel::Off => "off",
        ModelThinkingLevel::Minimal => "minimal",
        ModelThinkingLevel::Low => "low",
        ModelThinkingLevel::Medium => "medium",
        ModelThinkingLevel::High => "high",
        ModelThinkingLevel::Xhigh => "xhigh",
        ModelThinkingLevel::Max => "max",
    }
}

// -----------------------------------------------------------------------------
// 响应方向：流式事件
// -----------------------------------------------------------------------------

/// 流式事件。用 `#[serde(other)]` 兜底忽略不关心的事件。
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesStreamEvent {
    /// 一个输出项开始（message / reasoning / function_call）。
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: u32,
        item: ResponseItem,
    },
    /// 一个输出项结束。
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        output_index: u32,
        item: ResponseItem,
    },
    /// 文本增量。
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        output_index: u32,
        #[serde(default)]
        delta: String,
    },
    /// 思考摘要增量。
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        output_index: u32,
        #[serde(default)]
        delta: String,
    },
    /// 思考正文增量。
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        output_index: u32,
        #[serde(default)]
        delta: String,
    },
    /// 工具调用参数增量。
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        output_index: u32,
        #[serde(default)]
        delta: String,
    },
    /// 整个响应完成，带最终输出与用量。
    #[serde(rename = "response.completed")]
    Completed { response: ResponseObject },
    /// 响应未完成（如超长截断）。
    #[serde(rename = "response.incomplete")]
    Incomplete { response: ResponseObject },
    /// 响应失败。
    #[serde(rename = "response.failed")]
    Failed { response: ResponseObject },
    /// 其余事件（created / in_progress / part 级事件等）忽略。
    #[serde(other)]
    Other,
}

/// 一个输出项。字段全可选，兼容不同事件携带的形状。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ResponseItem {
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
    #[serde(default)]
    pub content: Option<Vec<ResponseContentPart>>,
    #[serde(default)]
    pub summary: Option<Vec<ResponseSummaryPart>>,
}

/// 消息内容块（output_text / refusal）。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ResponseContentPart {
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

/// 思考摘要块。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ResponseSummaryPart {
    #[serde(default)]
    pub text: Option<String>,
}

/// 最终响应对象。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ResponseObject {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
    #[serde(default)]
    pub output: Vec<ResponseItem>,
}

/// Responses API 的用量。
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ResponsesUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub input_tokens_details: Option<ResponsesInputDetails>,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub output_tokens_details: Option<ResponsesOutputDetails>,
    #[serde(default)]
    pub total_tokens: u64,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ResponsesInputDetails {
    #[serde(default)]
    pub cached_tokens: u64,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ResponsesOutputDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
}

/// 累积中的一个内容块（与 ChatCompletionStream 同一套思路）。
#[derive(Clone, Debug)]
enum Block {
    Text(String),
    Thinking(String),
    ToolCall {
        call_id: String,
        name: String,
        arguments_json: String,
    },
}

/// 把一串 `response.*` 事件聚合成完整助手消息与流式事件。
pub struct ResponsesStream {
    model: Model,
    response_id: Option<String>,
    response_model: Option<String>,
    usage: Usage,
    stop_reason: StopReason,
    error_message: Option<String>,
    /// 内容块列表，顺序即最终顺序。
    blocks: Vec<Block>,
    /// output_index -> 块下标。
    slots: HashMap<u32, usize>,
    /// 已结束（收到 output_item.done）的块下标集合。
    closed: Vec<bool>,
    timestamp: u64,
}

impl ResponsesStream {
    /// 为指定模型创建聚合器。
    #[must_use]
    pub fn new(model: Model) -> Self {
        Self {
            model,
            response_id: None,
            response_model: None,
            usage: zero_usage(),
            stop_reason: StopReason::Pending,
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

    /// 处理一个流式事件，返回产生的中间事件。
    pub fn handle_event(&mut self, event: &ResponsesStreamEvent) -> Vec<AssistantMessageEvent> {
        let mut events = Vec::new();
        match event {
            ResponsesStreamEvent::OutputItemAdded { output_index, item } => {
                self.create_slot(*output_index, item, &mut events);
            }
            ResponsesStreamEvent::OutputItemDone { output_index, item } => {
                self.close_slot(*output_index, item, &mut events);
            }
            ResponsesStreamEvent::OutputTextDelta {
                output_index,
                delta,
            } => {
                if let Some(&index) = self.slots.get(output_index) {
                    if let Block::Text(text) = &mut self.blocks[index] {
                        text.push_str(delta);
                        events.push(AssistantMessageEvent::TextDelta {
                            content_index: index as u32,
                            delta: delta.clone(),
                            partial: self.snapshot(),
                        });
                    }
                }
            }
            ResponsesStreamEvent::ReasoningSummaryTextDelta {
                output_index,
                delta,
            }
            | ResponsesStreamEvent::ReasoningTextDelta {
                output_index,
                delta,
            } => {
                if let Some(&index) = self.slots.get(output_index) {
                    if let Block::Thinking(thinking) = &mut self.blocks[index] {
                        thinking.push_str(delta);
                        events.push(AssistantMessageEvent::ThinkingDelta {
                            content_index: index as u32,
                            delta: delta.clone(),
                            partial: self.snapshot(),
                        });
                    }
                }
            }
            ResponsesStreamEvent::FunctionCallArgumentsDelta {
                output_index,
                delta,
            } => {
                if let Some(&index) = self.slots.get(output_index) {
                    if let Block::ToolCall { arguments_json, .. } = &mut self.blocks[index] {
                        arguments_json.push_str(delta);
                        events.push(AssistantMessageEvent::ToolcallDelta {
                            content_index: index as u32,
                            delta: delta.clone(),
                            partial: self.snapshot(),
                        });
                    }
                }
            }
            ResponsesStreamEvent::Completed { response }
            | ResponsesStreamEvent::Incomplete { response } => {
                self.apply_final_response(response, &mut events);
            }
            ResponsesStreamEvent::Failed { response } => {
                self.error_message = Some("响应失败".to_owned());
                if let Some(id) = &response.id {
                    if self.response_id.is_none() {
                        self.response_id = Some(id.clone());
                    }
                }
                events.push(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: self.snapshot(),
                });
            }
            ResponsesStreamEvent::Other => {}
        }
        events
    }

    /// 收尾：为未关闭的块发 end 事件，最后发 `done` 或 `error`。
    pub fn finish(&mut self) -> Vec<AssistantMessageEvent> {
        let mut events = Vec::new();
        for index in 0..self.blocks.len() {
            if self.closed.get(index).copied().unwrap_or(false) {
                continue;
            }
            match &self.blocks[index] {
                Block::Text(text) => events.push(AssistantMessageEvent::TextEnd {
                    content_index: index as u32,
                    content: text.clone(),
                    partial: self.snapshot(),
                }),
                Block::Thinking(thinking) => events.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: index as u32,
                    content: thinking.clone(),
                    partial: self.snapshot(),
                }),
                Block::ToolCall {
                    call_id,
                    name,
                    arguments_json,
                } => {
                    events.push(AssistantMessageEvent::ToolcallEnd {
                        content_index: index as u32,
                        tool_call: ToolCall {
                            id: call_id.clone(),
                            name: name.clone(),
                            arguments: parse_arguments(arguments_json),
                            thought_signature: None,
                            namespace: None,
                        },
                        partial: self.snapshot(),
                    });
                }
            }
            if let Some(flag) = self.closed.get_mut(index) {
                *flag = true;
            }
        }

        // 推断结束原因：没收到 completed 就按是否有工具调用推断。
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

    /// 为输出项创建对应的块和 start 事件。
    fn create_slot(
        &mut self,
        output_index: u32,
        item: &ResponseItem,
        events: &mut Vec<AssistantMessageEvent>,
    ) {
        let index = self.blocks.len();
        let block = match item.kind.as_str() {
            "reasoning" => Block::Thinking(String::new()),
            "function_call" => Block::ToolCall {
                call_id: item.call_id.clone().unwrap_or_default(),
                name: item.name.clone().unwrap_or_default(),
                arguments_json: String::new(),
            },
            // message 及其它都当文本。
            _ => Block::Text(String::new()),
        };
        self.blocks.push(block);
        self.closed.push(false);
        self.slots.insert(output_index, index);

        let event = match &self.blocks[index] {
            Block::Thinking(_) => AssistantMessageEvent::ThinkingStart {
                content_index: index as u32,
                partial: self.snapshot(),
            },
            Block::ToolCall { .. } => AssistantMessageEvent::ToolcallStart {
                content_index: index as u32,
                partial: self.snapshot(),
            },
            Block::Text(_) => AssistantMessageEvent::TextStart {
                content_index: index as u32,
                partial: self.snapshot(),
            },
        };
        events.push(event);
    }

    /// 关闭输出项：用完整项内容收尾并发 end 事件。
    fn close_slot(
        &mut self,
        output_index: u32,
        item: &ResponseItem,
        events: &mut Vec<AssistantMessageEvent>,
    ) {
        let Some(&index) = self.slots.get(&output_index) else {
            return;
        };
        match &mut self.blocks[index] {
            Block::Text(text) => {
                // done 事件带完整文本时以它为准（覆盖增量累加的结果）。
                let full_text = item
                    .content
                    .as_ref()
                    .map(|parts| {
                        parts
                            .iter()
                            .map(|part| part.text.clone().unwrap_or_default())
                            .collect::<String>()
                    })
                    .unwrap_or_default();
                if !full_text.is_empty() {
                    *text = full_text;
                }
                events.push(AssistantMessageEvent::TextEnd {
                    content_index: index as u32,
                    content: text.clone(),
                    partial: self.snapshot(),
                });
            }
            Block::Thinking(thinking) => {
                let summary = item
                    .summary
                    .as_ref()
                    .map(|parts| {
                        parts
                            .iter()
                            .filter_map(|part| part.text.clone())
                            .collect::<Vec<_>>()
                            .join("\n\n")
                    })
                    .unwrap_or_default();
                if !summary.is_empty() {
                    *thinking = summary;
                }
                events.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: index as u32,
                    content: thinking.clone(),
                    partial: self.snapshot(),
                });
            }
            Block::ToolCall { arguments_json, .. } => {
                if let Some(arguments) = &item.arguments {
                    *arguments_json = arguments.clone();
                }
            }
        }

        // 工具调用块收尾要读 call_id/name 和参数，单独处理避免可变借用冲突。
        if let Block::ToolCall {
            call_id,
            name,
            arguments_json,
        } = &self.blocks[index]
        {
            let (call_id, name, arguments) = (
                call_id.clone(),
                name.clone(),
                parse_arguments(arguments_json),
            );
            events.push(AssistantMessageEvent::ToolcallEnd {
                content_index: index as u32,
                tool_call: ToolCall {
                    id: call_id,
                    name,
                    arguments,
                    thought_signature: None,
                    namespace: None,
                },
                partial: self.snapshot(),
            });
        }
        if let Some(flag) = self.closed.get_mut(index) {
            *flag = true;
        }
        self.slots.remove(&output_index);
    }

    /// 应用最终响应：用量、结束原因、响应 id。
    fn apply_final_response(
        &mut self,
        response: &ResponseObject,
        events: &mut Vec<AssistantMessageEvent>,
    ) {
        if self.response_id.is_none() {
            if let Some(id) = &response.id {
                self.response_id = Some(id.clone());
            }
        }
        if self.response_model.is_none() {
            if let Some(model) = &response.model {
                self.response_model = Some(model.clone());
            }
        }
        if let Some(usage) = &response.usage {
            self.usage = usage_from_responses(usage, &self.model);
        }
        // 结束原因：有工具调用就是 ToolUse，否则 Stop。
        self.stop_reason = if self
            .blocks
            .iter()
            .any(|block| matches!(block, Block::ToolCall { .. }))
        {
            StopReason::ToolUse
        } else {
            StopReason::Stop
        };
        events.extend(self.finish());
    }

    /// 当前累积状态的快照。
    #[must_use]
    pub fn snapshot(&self) -> AssistantMessage {
        let content: Vec<AssistantContent> = self
            .blocks
            .iter()
            .map(|block| match block {
                Block::Text(text) => AssistantContent::Text(TextContent {
                    text: text.clone(),
                    text_signature: None,
                }),
                Block::Thinking(thinking) => AssistantContent::Thinking(ThinkingContent {
                    thinking: thinking.clone(),
                    thinking_signature: None,
                    redacted: None,
                }),
                Block::ToolCall {
                    call_id,
                    name,
                    arguments_json,
                } => AssistantContent::ToolCall(ToolCall {
                    id: call_id.clone(),
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
            response_model: self.response_model.clone(),
            response_id: self.response_id.clone(),
            diagnostics: None,
            usage: self.usage.clone(),
            stop_reason: self.stop_reason,
            deferred: None,
            error_message: self.error_message.clone(),
            raw_stop_reason: None,
            end_turn: None,
            timestamp: self.timestamp,
        }
    }
}

/// 把 Responses 的用量转成 pi 的 Usage（并计算费用）。
fn usage_from_responses(usage: &ResponsesUsage, model: &Model) -> Usage {
    let cached = usage
        .input_tokens_details
        .as_ref()
        .map_or(0, |details| details.cached_tokens);
    let reasoning = usage
        .output_tokens_details
        .as_ref()
        .and_then(|details| details.reasoning_tokens)
        .unwrap_or(0);
    let mut result = zero_usage();
    result.input = usage.input_tokens.saturating_sub(cached);
    result.cache_read = cached;
    result.output = usage.output_tokens;
    result.reasoning = Some(reasoning);
    result.total_tokens = usage.total_tokens;
    result.cost = calculate_cost(model, &result);
    result
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

/// 基于 OpenAI Responses API 的 Provider。
pub struct OpenAiResponsesProvider {
    transport: Box<dyn HttpTransport>,
    api_key: String,
}

impl OpenAiResponsesProvider {
    /// 用传输层和 API Key 创建 Provider。
    #[must_use]
    pub fn new(transport: Box<dyn HttpTransport>, api_key: impl Into<String>) -> Self {
        Self {
            transport,
            api_key: api_key.into(),
        }
    }

    /// 构造发给 `/responses` 的 HTTP 请求。
    fn build_http_request(
        &self,
        model: &Model,
        context: &Context,
        options: &RequestOptions,
    ) -> Result<HttpRequest, HttpError> {
        let body = serde_json::to_string(&build_request(model, context, options))
            .map_err(|error| HttpError::new(format!("序列化请求失败: {error}")))?;
        let url = format!("{}/responses", model.base_url.trim_end_matches('/'));
        let headers = vec![
            ("Content-Type".to_owned(), "application/json".to_owned()),
            (
                "Authorization".to_owned(),
                format!("Bearer {}", self.api_key),
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
            Ok(reader) => Box::new(ResponsesEventStream::new(model.clone(), reader)),
            Err(error) => Box::new(std::iter::once(error_event(model, &error.message))),
        }
    }
}

impl Provider for OpenAiResponsesProvider {
    fn id(&self) -> &str {
        "openai-responses"
    }

    fn name(&self) -> &str {
        "OpenAI Responses API"
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

/// 惰性事件流：边读 SSE 边聚合（与 ChatCompletionEventStream 同一套设计）。
pub struct ResponsesEventStream {
    reader: BufReader<Box<dyn Read>>,
    stream: ResponsesStream,
    pending: std::collections::VecDeque<AssistantMessageEvent>,
    finished: bool,
}

impl ResponsesEventStream {
    /// 用模型和响应体 reader 创建事件流；首个事件是 `start`。
    #[must_use]
    pub fn new(model: Model, reader: Box<dyn Read>) -> Self {
        let stream = ResponsesStream::new(model);
        let mut pending = std::collections::VecDeque::new();
        pending.push_back(stream.start_event());
        Self {
            reader: BufReader::new(reader),
            stream,
            pending,
            finished: false,
        }
    }
}

impl Iterator for ResponsesEventStream {
    type Item = AssistantMessageEvent;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(event);
            }
            if self.finished {
                return None;
            }
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    let events = self.stream.finish();
                    self.pending.extend(events);
                    self.finished = true;
                }
                Ok(_) => {
                    let Some(payload) = parse_sse_data(line.trim_end()) else {
                        continue;
                    };
                    // Responses 流没有 [DONE] 标记，但兼容一下。
                    if payload == "[DONE]" {
                        let events = self.stream.finish();
                        self.pending.extend(events);
                        self.finished = true;
                        continue;
                    }
                    if let Ok(event) = serde_json::from_str::<ResponsesStreamEvent>(payload) {
                        let events = self.stream.handle_event(&event);
                        self.pending.extend(events);
                    }
                }
                Err(error) => {
                    self.stream.error_message = Some(format!("读取响应流失败: {error}"));
                    self.stream.stop_reason = StopReason::Error;
                    let events = self.stream.finish();
                    self.pending.extend(events);
                    self.finished = true;
                }
            }
        }
    }
}

/// 从 SSE 行取出 `data:` 后面的内容。
fn parse_sse_data(line: &str) -> Option<&str> {
    line.strip_prefix("data:")
        .map(str::trim)
        .filter(|payload| !payload.is_empty())
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
