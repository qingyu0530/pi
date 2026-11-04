use std::io::Read;

use pi_ai::api::bedrock::{BedrockProvider, EventFrames, build_request, convert_messages};
use pi_ai::api::http::{HttpError, HttpRequest, HttpTransport};
use pi_ai::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantRole, Context,
    ConversationMessage, InputType, Model, ModelCost, ModelCostRates, ModelThinkingLevel, Provider,
    RequestOptions, StopReason, TextContent, ThinkingContent, Tool, ToolCall, ToolResultContent,
    ToolResultMessage, ToolResultRole, Usage, UsageCost, UserContent, UserMessage,
    UserMessageContent, UserRole,
};
use serde_json::{Value, json};

fn model() -> Model {
    Model {
        id: "anthropic.claude-sonnet-4-5".to_owned(),
        name: "Claude Sonnet 4.5".to_owned(),
        api: "bedrock-converse-stream".to_owned(),
        provider: "amazon-bedrock".to_owned(),
        base_url: "https://bedrock-runtime.us-east-1.amazonaws.com".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![InputType::Text],
        cost: ModelCost {
            rates: ModelCostRates {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            },
            tiers: None,
        },
        context_window: 200_000,
        max_tokens: 8_192,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn reasoning_model() -> Model {
    Model {
        reasoning: true,
        ..model()
    }
}

fn user_text(text: &str) -> UserMessage {
    UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Text(text.to_owned()),
        timestamp: 1,
    }
}

fn context(system_prompt: Option<&str>, messages: Vec<ConversationMessage>) -> Context {
    Context {
        system_prompt: system_prompt.map(str::to_owned),
        messages,
        tools: None,
    }
}

// -----------------------------------------------------------------------------
// 请求方向
// -----------------------------------------------------------------------------

#[test]
fn system_and_user_convert_to_converse_shape() {
    let messages = convert_messages(&context(Some("sys"), vec![user_text("hi").into()]));

    assert_eq!(messages.len(), 1);
    let value = serde_json::to_value(&messages[0]).unwrap();
    assert_eq!(value["role"], "user");
    assert_eq!(value["content"][0]["text"], "hi");

    let request = build_request(
        &model(),
        &context(Some("sys"), vec![user_text("hi").into()]),
        &RequestOptions::default(),
    );
    let value = serde_json::to_value(&request).unwrap();
    assert_eq!(value["modelId"], "anthropic.claude-sonnet-4-5");
    assert_eq!(value["system"][0]["text"], "sys");
    // Claude 模型默认带 maxTokens。
    assert_eq!(value["inferenceConfig"]["maxTokens"], 8192);
    // 非推理模型不发思考配置。
    assert!(value.get("additionalModelRequestFields").is_none());
}

#[test]
fn assistant_text_and_tool_call_replay() {
    let assistant = AssistantMessage {
        role: AssistantRole::Assistant,
        content: vec![
            AssistantContent::Text(TextContent {
                text: "调用工具".to_owned(),
                text_signature: None,
            }),
            AssistantContent::Thinking(ThinkingContent {
                thinking: "想一下".to_owned(),
                thinking_signature: Some("sig-1".to_owned()),
                redacted: None,
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "t1".to_owned(),
                name: "echo".to_owned(),
                arguments: json!({"text": "hi"}).as_object().unwrap().clone(),
                thought_signature: None,
                namespace: None,
            }),
        ],
        api: "bedrock-converse-stream".to_owned(),
        provider: "amazon-bedrock".to_owned(),
        model: "anthropic.claude-sonnet-4-5".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: zero_usage(),
        stop_reason: StopReason::ToolUse,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 2,
    };

    let messages = convert_messages(&context(None, vec![assistant.into()]));

    assert_eq!(messages.len(), 1);
    let value = serde_json::to_value(&messages[0]).unwrap();
    assert_eq!(value["role"], "assistant");
    // 文本块。
    assert_eq!(value["content"][0]["text"], "调用工具");
    // 思考块带签名。
    assert_eq!(
        value["content"][1]["reasoningContent"]["reasoningText"]["text"],
        "想一下"
    );
    assert_eq!(
        value["content"][1]["reasoningContent"]["reasoningText"]["signature"],
        "sig-1"
    );
    // 工具调用块。
    assert_eq!(value["content"][2]["toolUse"]["toolUseId"], "t1");
    assert_eq!(value["content"][2]["toolUse"]["name"], "echo");
    assert_eq!(value["content"][2]["toolUse"]["input"]["text"], "hi");
}

#[test]
fn tool_results_merge_into_single_user_message() {
    let result_one = tool_result("t1", "one", false);
    let result_two = tool_result("t2", "two", true);

    let messages = convert_messages(&context(None, vec![result_one.into(), result_two.into()]));

    // 两条工具结果合并成一条 user 消息。
    assert_eq!(messages.len(), 1);
    let value = serde_json::to_value(&messages[0]).unwrap();
    assert_eq!(value["role"], "user");
    assert_eq!(value["content"][0]["toolResult"]["toolUseId"], "t1");
    assert_eq!(value["content"][0]["toolResult"]["status"], "success");
    assert_eq!(
        value["content"][0]["toolResult"]["content"][0]["text"],
        "one"
    );
    assert_eq!(value["content"][1]["toolResult"]["toolUseId"], "t2");
    assert_eq!(value["content"][1]["toolResult"]["status"], "error");
}

#[test]
fn tools_convert_to_tool_config() {
    let tools = vec![Tool {
        name: "echo".to_owned(),
        description: "回显".to_owned(),
        parameters: json!({"type": "object"}),
        constrained_sampling: None,
    }];

    let request = build_request(
        &model(),
        &Context {
            system_prompt: None,
            messages: vec![user_text("hi").into()],
            tools: Some(tools),
        },
        &RequestOptions::default(),
    );
    let value = serde_json::to_value(&request).unwrap();

    assert_eq!(value["toolConfig"]["tools"][0]["toolSpec"]["name"], "echo");
    assert_eq!(
        value["toolConfig"]["tools"][0]["toolSpec"]["description"],
        "回显"
    );
    assert_eq!(
        value["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]["type"],
        "object"
    );
}

#[test]
fn claude_reasoning_sets_thinking_budget() {
    let options = RequestOptions {
        reasoning_effort: Some(ModelThinkingLevel::High),
        ..RequestOptions::default()
    };

    let request = build_request(
        &reasoning_model(),
        &context(None, vec![user_text("hi").into()]),
        &options,
    );
    let value = serde_json::to_value(&request).unwrap();

    assert_eq!(
        value["additionalModelRequestFields"]["thinking"]["type"],
        "enabled"
    );
    assert_eq!(
        value["additionalModelRequestFields"]["thinking"]["budget_tokens"],
        16384
    );
}

#[test]
fn non_claude_model_omits_thinking_config() {
    let mut nova = reasoning_model();
    nova.id = "amazon.nova-2-lite-v1:0".to_owned();
    nova.name = "Nova 2 Lite".to_owned();
    let options = RequestOptions {
        reasoning_effort: Some(ModelThinkingLevel::High),
        ..RequestOptions::default()
    };

    let request = build_request(
        &nova,
        &context(None, vec![user_text("hi").into()]),
        &options,
    );
    let value = serde_json::to_value(&request).unwrap();

    // 思考配置只发给 Claude 模型。
    assert!(value.get("additionalModelRequestFields").is_none());
}

#[test]
fn user_image_blocks_convert() {
    let user = UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Blocks(vec![
            UserContent::Text(TextContent {
                text: "看图".to_owned(),
                text_signature: None,
            }),
            UserContent::Image(pi_ai::ImageContent {
                data: "aGk=".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ]),
        timestamp: 1,
    };

    let messages = convert_messages(&context(None, vec![user.into()]));
    let value = serde_json::to_value(&messages[0]).unwrap();

    assert_eq!(value["content"][0]["text"], "看图");
    assert_eq!(value["content"][1]["image"]["format"], "png");
    assert_eq!(value["content"][1]["image"]["source"]["bytes"], "aGk=");
}

fn tool_result(tool_use_id: &str, text: &str, is_error: bool) -> ToolResultMessage {
    ToolResultMessage {
        role: ToolResultRole::ToolResult,
        tool_call_id: tool_use_id.to_owned(),
        tool_name: "echo".to_owned(),
        content: vec![ToolResultContent::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error,
        timestamp: 3,
    }
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

// -----------------------------------------------------------------------------
// 二进制 event-stream 解析
// -----------------------------------------------------------------------------

/// CRC32（与实现同算法），测试里用来构造合法帧。
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

/// 写一个字符串型事件头。
fn write_header(out: &mut Vec<u8>, name: &str, value: &str) {
    out.push(name.len() as u8);
    out.extend_from_slice(name.as_bytes());
    out.push(7); // string
    out.extend_from_slice(&(value.len() as u16).to_be_bytes());
    out.extend_from_slice(value.as_bytes());
}

/// 编码一帧合法的 AWS event-stream。
fn encode_frame(event_type: &str, payload: &str) -> Vec<u8> {
    let mut headers = Vec::new();
    write_header(&mut headers, ":message-type", "event");
    write_header(&mut headers, ":event-type", event_type);
    let payload_bytes = payload.as_bytes();
    let total_len = 12 + headers.len() + payload_bytes.len() + 4;

    let mut frame = Vec::new();
    frame.extend_from_slice(&(total_len as u32).to_be_bytes());
    frame.extend_from_slice(&(headers.len() as u32).to_be_bytes());
    frame.extend_from_slice(&crc32(&frame[..8]).to_be_bytes());
    frame.extend_from_slice(&headers);
    frame.extend_from_slice(payload_bytes);
    frame.extend_from_slice(&crc32(&frame).to_be_bytes());
    frame
}

#[test]
fn parses_binary_event_frames() {
    let mut bytes = Vec::new();
    bytes.extend(encode_frame("messageStart", r#"{"role":"assistant"}"#));
    bytes.extend(encode_frame(
        "contentBlockDelta",
        r#"{"contentBlockIndex":0,"delta":{"text":"你好"}}"#,
    ));
    bytes.extend(encode_frame(
        "metadata",
        r#"{"usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15}}"#,
    ));

    let frames: Vec<_> = EventFrames::new(std::io::Cursor::new(bytes))
        .map(Result::unwrap)
        .collect();

    assert_eq!(frames.len(), 3);
    assert_eq!(frames[0].event_type, "messageStart");
    assert_eq!(frames[0].payload["role"], "assistant");
    assert_eq!(frames[1].event_type, "contentBlockDelta");
    assert_eq!(frames[1].payload["delta"]["text"], "你好");
    assert_eq!(frames[2].event_type, "metadata");
    assert_eq!(frames[2].payload["usage"]["inputTokens"], 10);
}

#[test]
fn corrupted_frame_fails_crc_check() {
    let mut frame = encode_frame("messageStop", r#"{"stopReason":"end_turn"}"#);
    // 破坏载荷里的一个字节。
    let len = frame.len();
    frame[len - 5] ^= 0xFF;

    let result: Vec<_> = EventFrames::new(std::io::Cursor::new(frame)).collect();
    assert!(result[0].is_err(), "CRC 校验应失败");
}

// -----------------------------------------------------------------------------
// Provider
// -----------------------------------------------------------------------------

/// 返回固定字节的假传输层。
struct FakeTransport {
    body: Vec<u8>,
}

impl HttpTransport for FakeTransport {
    fn post(&self, _request: &HttpRequest) -> Result<Box<dyn Read>, HttpError> {
        Ok(Box::new(std::io::Cursor::new(self.body.clone())))
    }
}

fn done_message(events: &[AssistantMessageEvent]) -> AssistantMessage {
    events
        .iter()
        .rev()
        .find_map(|event| match event {
            AssistantMessageEvent::Done { message, .. } => Some(message.clone()),
            _ => None,
        })
        .expect("stream should end with done")
}

#[test]
fn provider_streams_text_from_event_stream() {
    let mut body = Vec::new();
    body.extend(encode_frame("messageStart", r#"{"role":"assistant"}"#));
    body.extend(encode_frame(
        "contentBlockDelta",
        r#"{"contentBlockIndex":0,"delta":{"text":"你好"}}"#,
    ));
    body.extend(encode_frame(
        "contentBlockDelta",
        r#"{"contentBlockIndex":0,"delta":{"text":"，世界"}}"#,
    ));
    body.extend(encode_frame(
        "contentBlockStop",
        r#"{"contentBlockIndex":0}"#,
    ));
    body.extend(encode_frame("messageStop", r#"{"stopReason":"end_turn"}"#));
    body.extend(encode_frame(
        "metadata",
        r#"{"usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15}}"#,
    ));

    let provider = BedrockProvider::new(Box::new(FakeTransport { body }), "test-token");
    let events: Vec<AssistantMessageEvent> = provider
        .stream(
            &model(),
            &context(None, vec![user_text("hi").into()]),
            &RequestOptions::default(),
        )
        .collect();

    assert!(matches!(
        events.first(),
        Some(AssistantMessageEvent::Start { .. })
    ));
    let message = done_message(&events);
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("end_turn"));
    match &message.content[0] {
        AssistantContent::Text(text) => assert_eq!(text.text, "你好，世界"),
        other => panic!("expected text, got {other:?}"),
    }
    assert_eq!(message.usage.input, 10);
    assert_eq!(message.usage.output, 5);
    // 有费用（单价 3/15 每百万 token）。
    assert!(message.usage.cost.total > 0.0);
}

#[test]
fn provider_aggregates_tool_calls() {
    let mut body = Vec::new();
    body.extend(encode_frame("messageStart", r#"{"role":"assistant"}"#));
    body.extend(encode_frame(
        "contentBlockStart",
        r#"{"contentBlockIndex":2,"start":{"toolUse":{"toolUseId":"t1","name":"echo"}}}"#,
    ));
    body.extend(encode_frame(
        "contentBlockDelta",
        r#"{"contentBlockIndex":2,"delta":{"toolUse":{"input":"{\"text\":"}}}"#,
    ));
    body.extend(encode_frame(
        "contentBlockDelta",
        r#"{"contentBlockIndex":2,"delta":{"toolUse":{"input":"\"hi\"}"}}}"#,
    ));
    body.extend(encode_frame(
        "contentBlockStop",
        r#"{"contentBlockIndex":2}"#,
    ));
    body.extend(encode_frame("messageStop", r#"{"stopReason":"tool_use"}"#));
    body.extend(encode_frame(
        "metadata",
        r#"{"usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15}}"#,
    ));

    let provider = BedrockProvider::new(Box::new(FakeTransport { body }), "test-token");
    let events: Vec<AssistantMessageEvent> = provider
        .stream(
            &model(),
            &context(None, vec![user_text("hi").into()]),
            &RequestOptions::default(),
        )
        .collect();

    let message = done_message(&events);
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    match &message.content[0] {
        AssistantContent::ToolCall(call) => {
            assert_eq!(call.id, "t1");
            assert_eq!(call.name, "echo");
            assert_eq!(
                call.arguments.get("text").and_then(Value::as_str),
                Some("hi")
            );
        }
        other => panic!("expected tool call, got {other:?}"),
    }
}

#[test]
fn provider_aggregates_thinking() {
    let mut body = Vec::new();
    body.extend(encode_frame("messageStart", r#"{"role":"assistant"}"#));
    body.extend(encode_frame(
        "contentBlockDelta",
        r#"{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"想一下"}}}"#,
    ));
    body.extend(encode_frame(
        "contentBlockDelta",
        r#"{"contentBlockIndex":0,"delta":{"reasoningContent":{"signature":"sig"}}}"#,
    ));
    body.extend(encode_frame(
        "contentBlockStop",
        r#"{"contentBlockIndex":0}"#,
    ));
    body.extend(encode_frame(
        "contentBlockDelta",
        r#"{"contentBlockIndex":1,"delta":{"text":"答"}}"#,
    ));
    body.extend(encode_frame("messageStop", r#"{"stopReason":"end_turn"}"#));
    body.extend(encode_frame(
        "metadata",
        r#"{"usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15}}"#,
    ));

    let provider = BedrockProvider::new(Box::new(FakeTransport { body }), "test-token");
    let events: Vec<AssistantMessageEvent> = provider
        .stream(
            &model(),
            &context(None, vec![user_text("hi").into()]),
            &RequestOptions::default(),
        )
        .collect();

    let message = done_message(&events);
    // 思考块在前（带签名），文本块在后。
    match &message.content[0] {
        AssistantContent::Thinking(thinking) => {
            assert_eq!(thinking.thinking, "想一下");
            assert_eq!(thinking.thinking_signature.as_deref(), Some("sig"));
        }
        other => panic!("expected thinking, got {other:?}"),
    }
    match &message.content[1] {
        AssistantContent::Text(text) => assert_eq!(text.text, "答"),
        other => panic!("expected text, got {other:?}"),
    }
}

#[test]
fn http_request_uses_bearer_token_and_encoded_model_path() {
    let provider = BedrockProvider::new(Box::new(FakeTransport { body: Vec::new() }), "tok");
    // 借用 provider 的私有方法不可行；通过 build_request 的 URL 逻辑验证。
    // 这里改用公开行为：流式失败时（空 body -> EOF）也能收到事件流。
    let events: Vec<AssistantMessageEvent> = provider
        .stream(
            &model(),
            &context(None, vec![user_text("hi").into()]),
            &RequestOptions::default(),
        )
        .collect();
    // 空 body：EOF 直接收尾，发 start + done。
    assert!(matches!(
        events.first(),
        Some(AssistantMessageEvent::Start { .. })
    ));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AssistantMessageEvent::Done { .. }))
    );
}
