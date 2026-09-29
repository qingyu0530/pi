use std::cell::RefCell;
use std::io::Read;
use std::rc::Rc;

use pi_ai::api::anthropic_messages::{
    AnthropicMessagesProvider, AnthropicStream, aggregate_sse, build_request, convert_messages,
    map_stop_reason,
};
use pi_ai::api::http::{HttpError, HttpRequest, HttpTransport};
use pi_ai::{
    AnthropicAllowedFallbackModel, AnthropicMessagesCompat, AssistantContent, AssistantMessage,
    AssistantMessageEvent, AssistantRole, CacheRetention, Context, ConversationMessage,
    ImageContent, InputType, Model, ModelCompat, ModelCost, ModelCostRates, ModelThinkingLevel,
    Provider, RequestOptions, StopReason, TextContent, ThinkingContent, Tool, ToolCall,
    ToolResultContent, ToolResultMessage, ToolResultRole, Usage, UsageCost, UserContent,
    UserMessage, UserMessageContent, UserRole,
};
use serde_json::json;

fn rates() -> ModelCostRates {
    ModelCostRates {
        input: 0.0,
        output: 0.0,
        cache_read: 0.0,
        cache_write: 0.0,
    }
}

fn model() -> Model {
    Model {
        id: "claude-sonnet-4-5".to_owned(),
        name: "Claude Sonnet 4.5".to_owned(),
        api: "anthropic-messages".to_owned(),
        provider: "anthropic".to_owned(),
        base_url: "https://api.anthropic.com".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![InputType::Text, InputType::Image],
        cost: ModelCost {
            rates: rates(),
            tiers: None,
        },
        context_window: 200_000,
        max_tokens: 1_024,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn reasoning_model() -> Model {
    Model {
        reasoning: true,
        max_tokens: 4_096,
        ..model()
    }
}

fn with_compat(model: Model, compat: AnthropicMessagesCompat) -> Model {
    Model {
        compat: Some(ModelCompat::AnthropicMessages(Box::new(compat))),
        ..model
    }
}

fn user_text(text: &str) -> UserMessage {
    UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Text(text.to_owned()),
        timestamp: 1,
    }
}

fn assistant(content: Vec<AssistantContent>) -> AssistantMessage {
    AssistantMessage {
        role: AssistantRole::Assistant,
        content,
        api: "anthropic-messages".to_owned(),
        provider: "anthropic".to_owned(),
        model: "claude-sonnet-4-5".to_owned(),
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
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 2,
    }
}

fn context(system_prompt: Option<&str>, messages: Vec<ConversationMessage>) -> Context {
    Context {
        system_prompt: system_prompt.map(str::to_owned),
        messages,
        tools: None,
    }
}

/// 记录请求并返回固定响应体的假传输层。
struct FakeTransport {
    body: String,
    request: Rc<RefCell<Option<HttpRequest>>>,
}

impl FakeTransport {
    fn new(body: &str) -> Self {
        Self {
            body: body.to_owned(),
            request: Rc::new(RefCell::new(None)),
        }
    }
}

impl HttpTransport for FakeTransport {
    fn post(&self, request: &HttpRequest) -> Result<Box<dyn Read>, HttpError> {
        *self.request.borrow_mut() = Some(request.clone());
        Ok(Box::new(std::io::Cursor::new(self.body.clone())))
    }
}

/// 每次只返回少量字节的 reader，用来验证跨读取边界的 SSE 解码。
struct ChunkedReader {
    data: Vec<u8>,
    position: usize,
    step: usize,
}

impl Read for ChunkedReader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if self.position >= self.data.len() {
            return Ok(0);
        }
        let remaining = &self.data[self.position..];
        let count = remaining.len().min(self.step).min(buffer.len());
        buffer[..count].copy_from_slice(&remaining[..count]);
        self.position += count;
        Ok(count)
    }
}

/// 分块返回响应体的传输层。
struct ChunkedTransport {
    body: String,
    step: usize,
    request: Rc<RefCell<Option<HttpRequest>>>,
}

impl HttpTransport for ChunkedTransport {
    fn post(&self, request: &HttpRequest) -> Result<Box<dyn Read>, HttpError> {
        *self.request.borrow_mut() = Some(request.clone());
        Ok(Box::new(ChunkedReader {
            data: self.body.clone().into_bytes(),
            position: 0,
            step: self.step,
        }))
    }
}

/// 取流结束时的 done 消息（克隆出来，避免借用临时 Vec）。
fn done_message(events: &[AssistantMessageEvent]) -> AssistantMessage {
    match events.last() {
        Some(AssistantMessageEvent::Done { message, .. }) => message.clone(),
        other => panic!("期望以 done 结束，实际: {other:?}"),
    }
}

/// 取流结束时的 error 消息（克隆出来）。
fn error_message(events: &[AssistantMessageEvent]) -> AssistantMessage {
    match events.last() {
        Some(AssistantMessageEvent::Error { error, .. }) => error.clone(),
        other => panic!("期望以 error 结束，实际: {other:?}"),
    }
}

fn text_sse() -> String {
    [
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4-5","usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":2,"cache_creation_input_tokens":3,"cache_creation":{"ephemeral_1h_input_tokens":1}}}}"#,
        "",
        "event: content_block_start",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        "",
        "event: content_block_delta",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你好"}}"#,
        "",
        "event: content_block_stop",
        r#"data: {"type":"content_block_stop","index":0}"#,
        "",
        "event: message_delta",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7,"output_tokens_details":{"thinking_tokens":2}}}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
        "",
    ]
    .join("\n")
}

// -----------------------------------------------------------------------------
// 请求构造
// -----------------------------------------------------------------------------

#[test]
fn request_has_required_fields_and_system_array() {
    let request = build_request(
        &model(),
        &context(Some("你是助手"), vec![user_text("你好").into()]),
        &RequestOptions::default(),
    );
    let value = serde_json::to_value(request).unwrap();

    assert_eq!(value["model"], "claude-sonnet-4-5");
    assert_eq!(value["stream"], true);
    assert_eq!(value["max_tokens"], 1_024);
    assert_eq!(
        value["system"],
        json!([{
            "type": "text",
            "text": "你是助手",
            "cache_control": { "type": "ephemeral" }
        }])
    );
}

#[test]
fn options_temperature_is_sent_when_thinking_is_off() {
    let options = RequestOptions {
        temperature: Some(0.5),
        max_tokens: Some(64),
        reasoning_effort: None,
        session_id: None,
        cache_retention: None,
    };
    let value = serde_json::to_value(build_request(
        &model(),
        &context(None, vec![user_text("hi").into()]),
        &options,
    ))
    .unwrap();

    assert_eq!(value["temperature"], 0.5);
    assert_eq!(value["max_tokens"], 64);
}

#[test]
fn long_cache_retention_adds_one_hour_ttl() {
    let options = RequestOptions {
        cache_retention: Some(CacheRetention::Long),
        ..RequestOptions::default()
    };
    let value = serde_json::to_value(build_request(
        &model(),
        &context(Some("系统"), vec![user_text("hi").into()]),
        &options,
    ))
    .unwrap();

    assert_eq!(value["system"][0]["cache_control"]["ttl"], "1h");
}

#[test]
fn no_cache_retention_omits_cache_control() {
    let options = RequestOptions {
        cache_retention: Some(CacheRetention::None),
        ..RequestOptions::default()
    };
    let value = serde_json::to_value(build_request(
        &model(),
        &context(Some("系统"), vec![user_text("hi").into()]),
        &options,
    ))
    .unwrap();

    assert!(value["system"][0].get("cache_control").is_none());
    assert!(value["messages"][0]["content"].is_string());
}

#[test]
fn user_blocks_become_text_and_image_blocks() {
    let message = UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Blocks(vec![
            UserContent::Text(TextContent {
                text: "看这张图".to_owned(),
                text_signature: None,
            }),
            UserContent::Image(ImageContent {
                data: "AAAA".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ]),
        timestamp: 1,
    };

    let messages = convert_messages(&model(), &context(None, vec![message.into()]), None);

    assert_eq!(
        serde_json::to_value(messages).unwrap(),
        json!([{
            "role": "user",
            "content": [
                { "type": "text", "text": "看这张图" },
                { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "AAAA" } }
            ]
        }])
    );
}

#[test]
fn assistant_thinking_becomes_thinking_or_text_by_signature() {
    let message = assistant(vec![
        AssistantContent::Thinking(ThinkingContent {
            thinking: "有签名".to_owned(),
            thinking_signature: Some("sig-1".to_owned()),
            redacted: None,
        }),
        AssistantContent::Thinking(ThinkingContent {
            thinking: "没签名".to_owned(),
            thinking_signature: None,
            redacted: None,
        }),
        AssistantContent::ToolCall(ToolCall {
            id: "call/1".to_owned(),
            name: "read".to_owned(),
            arguments: json!({ "path": "a.txt" }).as_object().unwrap().clone(),
            thought_signature: None,
            namespace: None,
        }),
    ]);

    let messages = convert_messages(&model(), &context(None, vec![message.into()]), None);

    assert_eq!(
        serde_json::to_value(messages).unwrap(),
        json!([{
            "role": "assistant",
            "content": [
                { "type": "thinking", "thinking": "有签名", "signature": "sig-1" },
                { "type": "text", "text": "没签名" },
                // id 被规范化：非字母数字变下划线。
                { "type": "tool_use", "id": "call_1", "name": "read", "input": { "path": "a.txt" } }
            ]
        }])
    );
}

#[test]
fn redacted_thinking_is_passed_back_as_opaque_data() {
    let message = assistant(vec![AssistantContent::Thinking(ThinkingContent {
        thinking: "[Reasoning redacted]".to_owned(),
        thinking_signature: Some("opaque".to_owned()),
        redacted: Some(true),
    })]);

    let messages = convert_messages(&model(), &context(None, vec![message.into()]), None);

    assert_eq!(
        serde_json::to_value(messages).unwrap(),
        json!([{ "role": "assistant", "content": [{ "type": "redacted_thinking", "data": "opaque" }] }])
    );
}

#[test]
fn consecutive_tool_results_merge_into_one_user_message() {
    let result = |id: &str, text: &str| -> ConversationMessage {
        ToolResultMessage {
            role: ToolResultRole::ToolResult,
            tool_call_id: id.to_owned(),
            tool_name: "read".to_owned(),
            content: vec![ToolResultContent::Text(TextContent {
                text: text.to_owned(),
                text_signature: None,
            })],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 3,
        }
        .into()
    };

    let messages = convert_messages(
        &model(),
        &context(None, vec![result("a/1", "内容A"), result("b/2", "内容B")]),
        None,
    );

    assert_eq!(
        serde_json::to_value(messages).unwrap(),
        json!([{
            "role": "user",
            "content": [
                { "type": "tool_result", "tool_use_id": "a_1", "content": "内容A", "is_error": false },
                { "type": "tool_result", "tool_use_id": "b_2", "content": "内容B", "is_error": false }
            ]
        }])
    );
}

#[test]
fn last_user_message_gets_cache_control_on_last_block() {
    let messages = convert_messages(
        &model(),
        &context(None, vec![user_text("hi").into()]),
        Some(&pi_ai::api::anthropic_messages::CacheControl {
            kind: pi_ai::api::anthropic_messages::CacheKind::Ephemeral,
            ttl: None,
        }),
    );

    assert_eq!(
        serde_json::to_value(messages).unwrap(),
        json!([{
            "role": "user",
            "content": [{ "type": "text", "text": "hi", "cache_control": { "type": "ephemeral" } }]
        }])
    );
}

#[test]
fn tools_are_converted_with_schema_and_cache_control() {
    let context = Context {
        system_prompt: None,
        messages: vec![user_text("hi").into()],
        tools: Some(vec![
            Tool {
                name: "read".to_owned(),
                description: "读取文件".to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"],
                    "additionalProperties": false
                }),
                constrained_sampling: None,
            },
            Tool {
                name: "write".to_owned(),
                description: "写文件".to_owned(),
                parameters: json!({ "type": "object" }),
                constrained_sampling: None,
            },
        ]),
    };

    let value = serde_json::to_value(build_request(
        &model(),
        &context,
        &RequestOptions::default(),
    ))
    .unwrap();

    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0]["name"], "read");
    assert_eq!(tools[0]["eager_input_streaming"], true);
    // 非严格模式只发 type/properties/required。
    assert_eq!(
        tools[0]["input_schema"],
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        })
    );
    assert!(tools[0].get("strict").is_none());
    // cache_control 只打在最后一个工具上。
    assert!(tools[0].get("cache_control").is_none());
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn strict_tools_merge_original_schema() {
    let model = with_compat(
        model(),
        AnthropicMessagesCompat {
            supports_strict_tools: Some(true),
            ..AnthropicMessagesCompat::default()
        },
    );
    let context = Context {
        system_prompt: None,
        messages: vec![user_text("hi").into()],
        tools: Some(vec![Tool {
            name: "read".to_owned(),
            description: "读取文件".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"],
                "additionalProperties": false
            }),
            constrained_sampling: Some(pi_ai::ConstrainedSamplingConfig::JsonSchema {
                strict: pi_ai::JsonSchemaStrict::Require,
            }),
        }]),
    };

    let value =
        serde_json::to_value(build_request(&model, &context, &RequestOptions::default())).unwrap();

    assert_eq!(value["tools"][0]["strict"], true);
    // 严格模式摊平原 schema，额外字段保留。
    assert_eq!(
        value["tools"][0]["input_schema"]["additionalProperties"],
        false
    );
    assert_eq!(value["tools"][0]["input_schema"]["type"], "object");
}

#[test]
fn non_reasoning_model_never_sends_thinking() {
    let options = RequestOptions {
        reasoning_effort: Some(ModelThinkingLevel::High),
        ..RequestOptions::default()
    };
    let value = serde_json::to_value(build_request(
        &model(),
        &context(None, vec![user_text("hi").into()]),
        &options,
    ))
    .unwrap();

    assert!(value.get("thinking").is_none());
}

#[test]
fn reasoning_model_thinking_modes() {
    // 未指定级别：显式关闭。
    let off = serde_json::to_value(build_request(
        &reasoning_model(),
        &context(None, vec![user_text("hi").into()]),
        &RequestOptions::default(),
    ))
    .unwrap();
    assert_eq!(off["thinking"], json!({ "type": "disabled" }));

    // 指定级别：按预算思考，temperature 不发。
    let options = RequestOptions {
        reasoning_effort: Some(ModelThinkingLevel::High),
        temperature: Some(0.7),
        ..RequestOptions::default()
    };
    let enabled = serde_json::to_value(build_request(
        &reasoning_model(),
        &context(None, vec![user_text("hi").into()]),
        &options,
    ))
    .unwrap();
    assert_eq!(
        enabled["thinking"],
        json!({ "type": "enabled", "budget_tokens": 1024, "display": "summarized" })
    );
    assert!(enabled.get("temperature").is_none());
}

#[test]
fn adaptive_thinking_sends_effort() {
    let model = with_compat(
        reasoning_model(),
        AnthropicMessagesCompat {
            force_adaptive_thinking: Some(true),
            ..AnthropicMessagesCompat::default()
        },
    );
    let options = RequestOptions {
        reasoning_effort: Some(ModelThinkingLevel::Low),
        ..RequestOptions::default()
    };
    let value = serde_json::to_value(build_request(
        &model,
        &context(None, vec![user_text("hi").into()]),
        &options,
    ))
    .unwrap();

    assert_eq!(
        value["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(value["output_config"], json!({ "effort": "low" }));
}

#[test]
fn off_level_marked_unsupported_omits_thinking() {
    let mut model = reasoning_model();
    model.thinking_level_map = Some([(ModelThinkingLevel::Off, None)].into_iter().collect());

    let value = serde_json::to_value(build_request(
        &model,
        &context(None, vec![user_text("hi").into()]),
        &RequestOptions::default(),
    ))
    .unwrap();

    assert!(value.get("thinking").is_none());
}

// -----------------------------------------------------------------------------
// 流式聚合
// -----------------------------------------------------------------------------

#[test]
fn text_stream_produces_start_delta_end_done() {
    let events = aggregate_sse(&model(), &text_sse());

    assert_eq!(events.len(), 5);
    assert!(matches!(events[0], AssistantMessageEvent::Start { .. }));
    assert!(matches!(
        events[1],
        AssistantMessageEvent::TextStart {
            content_index: 0,
            ..
        }
    ));
    assert!(matches!(
        &events[2],
        AssistantMessageEvent::TextDelta { delta, .. } if delta == "你好"
    ));
    assert!(matches!(
        &events[3],
        AssistantMessageEvent::TextEnd { content, .. } if content == "你好"
    ));

    let message = done_message(&events);
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.response_id.as_deref(), Some("msg_1"));
    assert_eq!(message.response_model, None);
    assert_eq!(message.usage.input, 10);
    assert_eq!(message.usage.output, 7);
    assert_eq!(message.usage.cache_read, 2);
    assert_eq!(message.usage.cache_write, 3);
    assert_eq!(message.usage.cache_write_1h, Some(1));
    assert_eq!(message.usage.reasoning, Some(2));
    assert_eq!(message.usage.total_tokens, 22);
    assert_eq!(
        message.content,
        vec![AssistantContent::Text(TextContent {
            text: "你好".to_owned(),
            text_signature: None,
        })]
    );
}

#[test]
fn tool_use_stream_accumulates_arguments() {
    let body = [
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_2","model":"claude-sonnet-4-5","usage":{"input_tokens":5,"output_tokens":0}}}"#,
        "",
        "event: content_block_start",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"read","input":{}}}"#,
        "",
        "event: content_block_delta",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}"#,
        "",
        "event: content_block_delta",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"a.txt\"}"}}"#,
        "",
        "event: content_block_stop",
        r#"data: {"type":"content_block_stop","index":0}"#,
        "",
        "event: message_delta",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
        "",
    ]
    .join("\n");

    let events = aggregate_sse(&model(), &body);

    let tool_call = match events.iter().find_map(|event| match event {
        AssistantMessageEvent::ToolcallEnd { tool_call, .. } => Some(tool_call.clone()),
        _ => None,
    }) {
        Some(tool_call) => tool_call,
        None => panic!("没有找到 toolcall_end 事件: {events:?}"),
    };
    assert_eq!(tool_call.id, "toolu_1");
    assert_eq!(tool_call.name, "read");
    assert_eq!(
        tool_call.arguments,
        json!({ "path": "a.txt" }).as_object().unwrap().clone()
    );

    let message = done_message(&events);
    assert_eq!(message.stop_reason, StopReason::ToolUse);
}

#[test]
fn thinking_signature_delta_is_captured() {
    let body = [
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_3","model":"claude-sonnet-4-5","usage":{"input_tokens":1,"output_tokens":0}}}"#,
        "",
        "event: content_block_start",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
        "",
        "event: content_block_delta",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"推理中"}}"#,
        "",
        "event: content_block_delta",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-xyz"}}"#,
        "",
        "event: content_block_stop",
        r#"data: {"type":"content_block_stop","index":0}"#,
        "",
        "event: message_delta",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
        "",
    ]
    .join("\n");

    let message = done_message(&aggregate_sse(&model(), &body));

    assert_eq!(
        message.content,
        vec![AssistantContent::Thinking(ThinkingContent {
            thinking: "推理中".to_owned(),
            thinking_signature: Some("sig-xyz".to_owned()),
            redacted: None,
        })]
    );
}

#[test]
fn redacted_thinking_block_becomes_placeholder() {
    let body = [
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_4","model":"claude-sonnet-4-5","usage":{"input_tokens":1,"output_tokens":0}}}"#,
        "",
        "event: content_block_start",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}"#,
        "",
        "event: content_block_stop",
        r#"data: {"type":"content_block_stop","index":0}"#,
        "",
        "event: message_delta",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
        "",
    ]
    .join("\n");

    let message = done_message(&aggregate_sse(&model(), &body));

    assert_eq!(
        message.content,
        vec![AssistantContent::Thinking(ThinkingContent {
            thinking: "[Reasoning redacted]".to_owned(),
            thinking_signature: Some("opaque".to_owned()),
            redacted: Some(true),
        })]
    );

    // 往返：被屏蔽的思考回传时必须还原成 redacted_thinking，而不是普通 thinking。
    let messages = convert_messages(&model(), &context(None, vec![message.into()]), None);
    assert_eq!(
        serde_json::to_value(messages).unwrap(),
        json!([{
            "role": "assistant",
            "content": [{ "type": "redacted_thinking", "data": "opaque" }]
        }])
    );
}

#[test]
fn unknown_content_block_is_skipped() {
    let body = [
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_5","model":"claude-sonnet-4-5","usage":{"input_tokens":1,"output_tokens":0}}}"#,
        "",
        "event: content_block_start",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"x"}}"#,
        "",
        "event: content_block_start",
        r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":"ok"}}"#,
        "",
        "event: content_block_stop",
        r#"data: {"type":"content_block_stop","index":0}"#,
        "",
        "event: content_block_stop",
        r#"data: {"type":"content_block_stop","index":1}"#,
        "",
        "event: message_delta",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
        "",
    ]
    .join("\n");

    let message = done_message(&aggregate_sse(&model(), &body));

    assert_eq!(
        message.content,
        vec![AssistantContent::Text(TextContent {
            text: "ok".to_owned(),
            text_signature: None,
        })]
    );
}

#[test]
fn comment_and_unknown_events_are_ignored() {
    let body = [
        ": 这是一行注释",
        "",
        "event: ping",
        r#"data: {"type":"ping"}"#,
        "",
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_6","model":"claude-sonnet-4-5","usage":{"input_tokens":1,"output_tokens":0}}}"#,
        "",
        "event: message_delta",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
        "",
    ]
    .join("\n");

    let events = aggregate_sse(&model(), &body);

    assert_eq!(events.len(), 2);
    let message = done_message(&events);
    assert_eq!(message.stop_reason, StopReason::Stop);
}

#[test]
fn sse_error_event_becomes_error_with_raw_data() {
    let body = [
        "event: error",
        r#"data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        "",
    ]
    .join("\n");

    let events = aggregate_sse(&model(), &body);

    let message = error_message(&events);
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some(r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#)
    );
}

#[test]
fn stream_without_message_stop_is_an_error() {
    let body = [
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_7","model":"claude-sonnet-4-5","usage":{"input_tokens":1,"output_tokens":0}}}"#,
        "",
    ]
    .join("\n");

    let message = error_message(&aggregate_sse(&model(), &body));
    assert_eq!(
        message.error_message.as_deref(),
        Some("Anthropic stream ended before message_stop")
    );
}

#[test]
fn stream_without_stop_reason_is_an_error() {
    let body = [
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_8","model":"claude-sonnet-4-5","usage":{"input_tokens":1,"output_tokens":0}}}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
        "",
    ]
    .join("\n");

    let message = error_message(&aggregate_sse(&model(), &body));
    assert_eq!(
        message.error_message.as_deref(),
        Some("Anthropic stream ended without a stop reason")
    );
}

#[test]
fn unknown_stop_reason_is_an_error() {
    let body = [
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_9","model":"claude-sonnet-4-5","usage":{"input_tokens":1,"output_tokens":0}}}"#,
        "",
        "event: message_delta",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"something_new"}}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
        "",
    ]
    .join("\n");

    let message = error_message(&aggregate_sse(&model(), &body));
    assert_eq!(
        message.error_message.as_deref(),
        Some("Unhandled stop reason: something_new")
    );
}

#[test]
fn refusal_stop_reason_carries_explanation() {
    let body = [
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_10","model":"claude-sonnet-4-5","usage":{"input_tokens":1,"output_tokens":0}}}"#,
        "",
        "event: message_delta",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"explanation":"不能回答"}}}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
        "",
    ]
    .join("\n");

    let message = error_message(&aggregate_sse(&model(), &body));
    assert_eq!(message.error_message.as_deref(), Some("不能回答"));
}

#[test]
fn response_model_fallback_switches_cost_table() {
    let model = with_compat(
        model(),
        AnthropicMessagesCompat {
            allowed_fallback_models: Some(vec![AnthropicAllowedFallbackModel {
                provider: "anthropic".to_owned(),
                model: "claude-haiku-x".to_owned(),
                cost: ModelCost {
                    rates: ModelCostRates {
                        input: 1_000_000.0,
                        output: 0.0,
                        cache_read: 0.0,
                        cache_write: 0.0,
                    },
                    tiers: None,
                },
            }]),
            ..AnthropicMessagesCompat::default()
        },
    );
    let body = [
        "event: message_start",
        r#"data: {"type":"message_start","message":{"id":"msg_11","model":"claude-haiku-x","usage":{"input_tokens":10,"output_tokens":0}}}"#,
        "",
        "event: message_delta",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
        "",
        "event: message_stop",
        r#"data: {"type":"message_stop"}"#,
        "",
    ]
    .join("\n");

    let message = done_message(&aggregate_sse(&model, &body));

    assert_eq!(message.response_model.as_deref(), Some("claude-haiku-x"));
    // 回退价格表：输入 1000000/百万 * 10 token = 10。
    assert!((message.usage.cost.total - 10.0).abs() < 1e-9);
}

// -----------------------------------------------------------------------------
// Provider 与传输
// -----------------------------------------------------------------------------

#[test]
fn provider_builds_request_url_headers_and_streams() {
    let transport = FakeTransport::new(&text_sse());
    let request_cell = Rc::clone(&transport.request);
    let provider = AnthropicMessagesProvider::new(Box::new(transport), "sk-ant-test");

    let options = RequestOptions {
        session_id: Some("sess-1".to_owned()),
        ..RequestOptions::default()
    };
    let events: Vec<_> = provider
        .stream(
            &model(),
            &context(Some("系统"), vec![user_text("hi").into()]),
            &options,
        )
        .collect();

    let request = request_cell.borrow().clone().unwrap();
    assert_eq!(request.url, "https://api.anthropic.com/v1/messages");

    let header = |name: &str| -> Option<String> {
        request
            .headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    };
    assert_eq!(header("x-api-key").as_deref(), Some("sk-ant-test"));
    assert_eq!(header("anthropic-version").as_deref(), Some("2023-06-01"));
    assert_eq!(header("accept").as_deref(), Some("application/json"));
    // 默认支持随到随流，所以只带交错思考 beta；默认不发会话亲和头。
    assert_eq!(
        header("anthropic-beta").as_deref(),
        Some("interleaved-thinking-2025-05-14")
    );
    assert!(header("x-session-affinity").is_none());

    let body: serde_json::Value = serde_json::from_str(&request.body).unwrap();
    assert_eq!(body["model"], "claude-sonnet-4-5");
    assert_eq!(body["stream"], true);

    assert_eq!(done_message(&events).stop_reason, StopReason::Stop);
}

#[test]
fn provider_adds_beta_and_affinity_headers_when_configured() {
    let model = with_compat(
        model(),
        AnthropicMessagesCompat {
            supports_eager_tool_input_streaming: Some(false),
            send_session_affinity_headers: Some(true),
            ..AnthropicMessagesCompat::default()
        },
    );
    let transport =
        FakeTransport::new("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
    let request_cell = Rc::clone(&transport.request);
    let provider = AnthropicMessagesProvider::new(Box::new(transport), "sk-ant-test");

    let context = Context {
        system_prompt: None,
        messages: vec![user_text("hi").into()],
        tools: Some(vec![Tool {
            name: "read".to_owned(),
            description: "读".to_owned(),
            parameters: json!({ "type": "object" }),
            constrained_sampling: None,
        }]),
    };
    let options = RequestOptions {
        session_id: Some("sess-2".to_owned()),
        ..RequestOptions::default()
    };
    let _: Vec<_> = provider.stream(&model, &context, &options).collect();

    let request = request_cell.borrow().clone().unwrap();
    let value = |name: &str| -> Option<String> {
        request
            .headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    };
    let beta = value("anthropic-beta").unwrap();
    assert!(beta.contains("fine-grained-tool-streaming-2025-05-14"));
    assert!(beta.contains("interleaved-thinking-2025-05-14"));
    assert_eq!(value("x-session-affinity").as_deref(), Some("sess-2"));
}

#[test]
fn provider_transport_failure_yields_single_error_event() {
    struct FailingTransport;
    impl HttpTransport for FailingTransport {
        fn post(&self, _request: &HttpRequest) -> Result<Box<dyn Read>, HttpError> {
            Err(HttpError::new("连接失败"))
        }
    }

    let provider = AnthropicMessagesProvider::new(Box::new(FailingTransport), "sk-ant-test");
    let events: Vec<_> = provider
        .stream(
            &model(),
            &context(None, vec![user_text("hi").into()]),
            &RequestOptions::default(),
        )
        .collect();

    assert_eq!(events.len(), 1);
    assert_eq!(
        error_message(&events).error_message.as_deref(),
        Some("连接失败")
    );
}

#[test]
fn streaming_reader_handles_chunk_boundaries() {
    let transport = ChunkedTransport {
        body: text_sse(),
        step: 7,
        request: Rc::new(RefCell::new(None)),
    };
    let provider = AnthropicMessagesProvider::new(Box::new(transport), "sk-ant-test");

    let events: Vec<_> = provider
        .stream(
            &model(),
            &context(None, vec![user_text("hi").into()]),
            &RequestOptions::default(),
        )
        .collect();

    assert_eq!(done_message(&events).usage.output, 7);
}

#[test]
fn aggregator_can_be_driven_directly() {
    let mut stream = AnthropicStream::new(model());
    let event: pi_ai::api::anthropic_messages::RawMessageStreamEvent =
        serde_json::from_str(r#"{"type":"message_stop"}"#).unwrap();
    let events = stream.handle_event(&event);
    assert!(events.is_empty());

    let events = stream.finish();
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], AssistantMessageEvent::Error { .. }));
}

#[test]
fn stop_reasons_map_to_pi_reasons() {
    assert_eq!(
        map_stop_reason("end_turn", None).unwrap().0,
        StopReason::Stop
    );
    assert_eq!(
        map_stop_reason("max_tokens", None).unwrap().0,
        StopReason::Length
    );
    assert_eq!(
        map_stop_reason("tool_use", None).unwrap().0,
        StopReason::ToolUse
    );
    assert_eq!(
        map_stop_reason("pause_turn", None).unwrap().0,
        StopReason::Stop
    );
    assert_eq!(
        map_stop_reason("stop_sequence", None).unwrap().0,
        StopReason::Stop
    );
    let (reason, message) = map_stop_reason("refusal", None).unwrap();
    assert_eq!(reason, StopReason::Error);
    assert_eq!(
        message.as_deref(),
        Some("The model refused to complete the request")
    );
    let (reason, message) = map_stop_reason("sensitive", None).unwrap();
    assert_eq!(reason, StopReason::Error);
    assert_eq!(message.as_deref(), Some("Provider stopped with: sensitive"));
    assert!(map_stop_reason("nope", None).is_err());
}
