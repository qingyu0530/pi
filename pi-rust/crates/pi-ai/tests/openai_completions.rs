use pi_ai::api::http::{HttpError, HttpRequest, HttpTransport};
use pi_ai::api::openai_completions::{
    ChatCompletionChunk, ChatCompletionStream, ChatMessage, ChunkUsage, CompletionTokensDetails,
    OpenAiCompletionsProvider, PromptTokensDetails, build_request, convert_messages,
    map_stop_reason, parse_usage,
};
use pi_ai::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantRole, Context,
    ConversationMessage, ImageContent, InputType, MaxTokensField, Model, ModelCompat, ModelCost,
    ModelCostRates, ModelCostTier, OpenAICompletionsCompat, OpenRouterRouting, Provider,
    StopReason, TextContent, Tool, ToolCall, ToolResultContent, ToolResultMessage, ToolResultRole,
    Usage, UsageCost, UserContent, UserMessage, UserMessageContent, UserRole, VercelGatewayRouting,
    calculate_cost, detect_openai_completions_compat,
};
use serde_json::json;
use std::cell::RefCell;
use std::rc::Rc;

fn model() -> Model {
    Model {
        id: "gpt-4o-mini".to_owned(),
        name: "GPT-4o mini".to_owned(),
        api: "openai-completions".to_owned(),
        provider: "openai".to_owned(),
        base_url: "https://api.openai.com/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![InputType::Text],
        cost: ModelCost {
            rates: ModelCostRates {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            tiers: None,
        },
        context_window: 128_000,
        max_tokens: 1_024,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn usage() -> Usage {
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
        api: "openai-completions".to_owned(),
        provider: "openai".to_owned(),
        model: "gpt-4o-mini".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: usage(),
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

#[test]
fn request_has_model_stream_and_usage_option() {
    let request = build_request(&model(), &context(None, vec![user_text("hi").into()]));
    let value = serde_json::to_value(request).unwrap();

    assert_eq!(value["model"], "gpt-4o-mini");
    assert_eq!(value["stream"], true);
    assert_eq!(value["stream_options"]["include_usage"], true);
    assert_eq!(value["max_completion_tokens"], 1_024);
}

#[test]
fn system_and_user_text_messages_match_openai_shape() {
    let messages = convert_messages(
        &model(),
        &context(Some("你是助手"), vec![user_text("你好").into()]),
    );

    assert_eq!(
        serde_json::to_value(messages).unwrap(),
        json!([
            { "role": "system", "content": "你是助手" },
            { "role": "user", "content": "你好" },
        ])
    );
}

#[test]
fn user_blocks_become_text_and_image_parts() {
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

    let messages = convert_messages(&model(), &context(None, vec![message.into()]));

    assert_eq!(
        serde_json::to_value(messages).unwrap(),
        json!([{
            "role": "user",
            "content": [
                { "type": "text", "text": "看这张图" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } }
            ]
        }])
    );
}

#[test]
fn assistant_tool_call_arguments_are_a_json_string() {
    let arguments = json!({ "path": "a.txt" }).as_object().unwrap().clone();
    let message = assistant(vec![
        AssistantContent::Text(TextContent {
            text: "我来读文件".to_owned(),
            text_signature: None,
        }),
        AssistantContent::ToolCall(ToolCall {
            id: "call_1".to_owned(),
            name: "read".to_owned(),
            arguments,
            thought_signature: None,
            namespace: None,
        }),
    ]);

    let messages = convert_messages(&model(), &context(None, vec![message.into()]));

    assert_eq!(
        serde_json::to_value(messages).unwrap(),
        json!([{
            "role": "assistant",
            "content": "我来读文件",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": { "name": "read", "arguments": "{\"path\":\"a.txt\"}" }
            }]
        }])
    );
}

#[test]
fn empty_assistant_message_is_skipped() {
    let messages = vec![
        ConversationMessage::from(assistant(Vec::new())),
        ConversationMessage::from(user_text("hi")),
    ];

    let converted = convert_messages(&model(), &context(None, messages));

    assert_eq!(converted.len(), 1);
    assert!(matches!(converted[0], ChatMessage::User { .. }));
}

#[test]
fn tool_result_becomes_tool_message() {
    let result = ToolResultMessage {
        role: ToolResultRole::ToolResult,
        tool_call_id: "call_1".to_owned(),
        tool_name: "read".to_owned(),
        content: vec![ToolResultContent::Text(TextContent {
            text: "文件内容".to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 3,
    };

    let messages = convert_messages(&model(), &context(None, vec![result.into()]));

    assert_eq!(
        serde_json::to_value(messages).unwrap(),
        json!([{ "role": "tool", "tool_call_id": "call_1", "content": "文件内容" }])
    );
}

#[test]
fn tools_are_serialized_as_function_tools() {
    let context = Context {
        system_prompt: None,
        messages: Vec::new(),
        tools: Some(vec![Tool {
            name: "read".to_owned(),
            description: "读取文件".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
            constrained_sampling: None,
        }]),
    };

    let request = build_request(&model(), &context);
    let value = serde_json::to_value(request).unwrap();

    assert_eq!(
        value["tools"],
        json!([{
            "type": "function",
            "function": {
                "name": "read",
                "description": "读取文件",
                "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                },
                // 标准 OpenAI 支持 strict 模式，带上 strict: false。
                "strict": false
            }
        }])
    );
}

fn priced_model() -> Model {
    let mut model = model();
    model.cost = ModelCost {
        rates: ModelCostRates {
            input: 3.0,
            output: 15.0,
            cache_read: 0.3,
            cache_write: 3.75,
        },
        tiers: None,
    };
    model
}

#[test]
fn chunk_deserializes_text_delta() {
    let chunk: ChatCompletionChunk = serde_json::from_value(json!({
        "id": "chatcmpl-1",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "delta": { "role": "assistant", "content": "你好" },
            "finish_reason": null
        }]
    }))
    .unwrap();

    assert_eq!(chunk.id.as_deref(), Some("chatcmpl-1"));
    assert_eq!(chunk.model.as_deref(), Some("gpt-4o-mini"));
    let choice = &chunk.choices[0];
    assert_eq!(choice.finish_reason, None);
    let delta = choice.delta.as_ref().unwrap();
    assert_eq!(delta.role.as_deref(), Some("assistant"));
    assert_eq!(delta.content.as_deref(), Some("你好"));
}

#[test]
fn chunk_deserializes_tool_call_fragments() {
    let chunk: ChatCompletionChunk = serde_json::from_value(json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "read", "arguments": "{\"path\"" }
                }]
            }
        }]
    }))
    .unwrap();

    let call = &chunk.choices[0]
        .delta
        .as_ref()
        .unwrap()
        .tool_calls
        .as_ref()
        .unwrap()[0];
    assert_eq!(call.index, 0);
    assert_eq!(call.id.as_deref(), Some("call_1"));
    assert_eq!(
        call.function.as_ref().unwrap().name.as_deref(),
        Some("read")
    );
    assert_eq!(
        call.function.as_ref().unwrap().arguments.as_deref(),
        Some("{\"path\"")
    );
}

#[test]
fn chunk_deserializes_usage_details() {
    let chunk: ChatCompletionChunk = serde_json::from_value(json!({
        "choices": [],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": { "cached_tokens": 40, "cache_write_tokens": 10 },
            "completion_tokens_details": { "reasoning_tokens": 5 }
        }
    }))
    .unwrap();

    let usage = chunk.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 100);
    assert_eq!(usage.completion_tokens, 20);
    assert_eq!(
        usage.prompt_tokens_details.as_ref().unwrap().cached_tokens,
        Some(40)
    );
    assert_eq!(
        usage
            .completion_tokens_details
            .as_ref()
            .unwrap()
            .reasoning_tokens,
        Some(5)
    );
}

#[test]
fn map_stop_reason_maps_known_reasons() {
    assert_eq!(map_stop_reason(None), (StopReason::Stop, None));
    assert_eq!(map_stop_reason(Some("stop")), (StopReason::Stop, None));
    assert_eq!(map_stop_reason(Some("length")), (StopReason::Length, None));
    assert_eq!(
        map_stop_reason(Some("tool_calls")),
        (StopReason::ToolUse, None)
    );

    let (reason, error) = map_stop_reason(Some("content_filter"));
    assert_eq!(reason, StopReason::Error);
    assert!(error.unwrap().contains("content_filter"));
}

#[test]
fn parse_usage_computes_input_and_cost() {
    let raw = ChunkUsage {
        prompt_tokens: 1_000,
        completion_tokens: 100,
        prompt_tokens_details: Some(PromptTokensDetails {
            cached_tokens: Some(200),
            cache_write_tokens: Some(100),
        }),
        completion_tokens_details: Some(CompletionTokensDetails {
            reasoning_tokens: Some(30),
        }),
        ..ChunkUsage::default()
    };

    let usage = parse_usage(&raw, &priced_model());

    assert_eq!(usage.input, 700);
    assert_eq!(usage.output, 100);
    assert_eq!(usage.cache_read, 200);
    assert_eq!(usage.cache_write, 100);
    assert_eq!(usage.total_tokens, 1_100);
    assert_eq!(usage.reasoning, Some(30));
    assert!((usage.cost.input - 0.0021).abs() < 1e-12);
    assert!((usage.cost.output - 0.0015).abs() < 1e-12);
    assert!((usage.cost.cache_read - 0.000_06).abs() < 1e-12);
    assert!((usage.cost.cache_write - 0.000_375).abs() < 1e-12);
    assert!((usage.cost.total - (0.0021 + 0.0015 + 0.000_06 + 0.000_375)).abs() < 1e-12);
}

#[test]
fn parse_usage_falls_back_to_top_level_cache_fields() {
    let raw = ChunkUsage {
        prompt_tokens: 500,
        completion_tokens: 10,
        prompt_cache_hit_tokens: Some(50),
        ..ChunkUsage::default()
    };

    let usage = parse_usage(&raw, &model());

    assert_eq!(usage.cache_read, 50);
    assert_eq!(usage.input, 450);
}

#[test]
fn calculate_cost_applies_the_highest_matching_tier() {
    let mut tiered = priced_model();
    tiered.cost.tiers = Some(vec![ModelCostTier {
        rates: ModelCostRates {
            input: 6.0,
            output: 30.0,
            cache_read: 0.6,
            cache_write: 7.5,
        },
        input_tokens_above: 1_000,
    }]);

    let mut request_usage = usage();
    request_usage.input = 2_000;
    request_usage.output = 100;
    request_usage.total_tokens = 2_100;

    let cost = calculate_cost(&tiered, &request_usage);

    // 输入 2000 token 超过阈值 1000，整次请求按 6 美元/百万计。
    assert!((cost.input - 0.012).abs() < 1e-12);
    assert!((cost.output - 0.003).abs() < 1e-12);
}

// -----------------------------------------------------------------------------
// 流式聚合测试
// -----------------------------------------------------------------------------

fn chunk(value: serde_json::Value) -> ChatCompletionChunk {
    serde_json::from_value(value).unwrap()
}

fn done_message(events: &[AssistantMessageEvent]) -> AssistantMessage {
    match events.last() {
        Some(AssistantMessageEvent::Done { message, .. }) => message.clone(),
        other => panic!("expected done as last event, got {other:?}"),
    }
}

#[test]
fn stream_aggregates_text_deltas() {
    let mut stream = ChatCompletionStream::new(model());

    let first = stream.handle_chunk(&chunk(json!({
        "id": "cmpl_1",
        "model": "gpt-4o-mini",
        "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "你好" } }]
    })));
    assert!(matches!(first[0], AssistantMessageEvent::TextStart { .. }));
    assert!(matches!(first[1], AssistantMessageEvent::TextDelta { .. }));

    let _ = stream.handle_chunk(&chunk(json!({
        "choices": [{ "index": 0, "delta": { "content": "，世界" } }]
    })));
    let _ = stream.handle_chunk(&chunk(json!({
        "id": "cmpl_1",
        "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14 }
    })));

    let events = stream.finish();
    let message = done_message(&events);

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.response_id.as_deref(), Some("cmpl_1"));
    assert_eq!(message.usage.input, 10);
    assert_eq!(message.usage.output, 4);
    match &message.content[0] {
        AssistantContent::Text(text) => assert_eq!(text.text, "你好，世界"),
        other => panic!("expected text, got {other:?}"),
    }
}

#[test]
fn stream_aggregates_tool_call_arguments() {
    let mut stream = ChatCompletionStream::new(model());

    let _ = stream.handle_chunk(&chunk(json!({
        "choices": [{ "index": 0, "delta": { "tool_calls": [
            { "index": 0, "id": "call_1", "type": "function",
              "function": { "name": "read", "arguments": "{\"path\":" } }
        ] } }]
    })));
    let _ = stream.handle_chunk(&chunk(json!({
        "choices": [{ "index": 0, "delta": { "tool_calls": [
            { "index": 0, "function": { "arguments": "\"a.txt\"}" } }
        ] } }]
    })));
    let _ = stream.handle_chunk(&chunk(json!({
        "choices": [{ "index": 0, "delta": {}, "finish_reason": "tool_calls" }]
    })));

    let events = stream.finish();
    let message = done_message(&events);

    assert_eq!(message.stop_reason, StopReason::ToolUse);
    match &message.content[0] {
        AssistantContent::ToolCall(call) => {
            assert_eq!(call.id, "call_1");
            assert_eq!(call.name, "read");
            assert_eq!(call.arguments.get("path"), Some(&json!("a.txt")));
        }
        other => panic!("expected tool call, got {other:?}"),
    }
}

#[test]
fn stream_aggregates_reasoning_into_thinking_block() {
    let mut stream = ChatCompletionStream::new(model());

    let _ = stream.handle_chunk(&chunk(json!({
        "choices": [{ "index": 0, "delta": { "reasoning_content": "先想" } }]
    })));
    let _ = stream.handle_chunk(&chunk(json!({
        "choices": [{ "index": 0, "delta": { "reasoning_content": "再看" } }]
    })));

    let events = stream.finish();
    let message = done_message(&events);

    match &message.content[0] {
        AssistantContent::Thinking(thinking) => assert_eq!(thinking.thinking, "先想再看"),
        other => panic!("expected thinking, got {other:?}"),
    }
}

// -----------------------------------------------------------------------------
// Provider 组装测试（用假传输层，不联网）
// -----------------------------------------------------------------------------

/// 记录请求、返回固定响应体的假传输层。
#[derive(Clone)]
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

    fn request(&self) -> Option<HttpRequest> {
        self.request.borrow().clone()
    }
}

impl HttpTransport for FakeTransport {
    fn post(&self, request: &HttpRequest) -> Result<String, HttpError> {
        *self.request.borrow_mut() = Some(request.clone());
        Ok(self.body.clone())
    }
}

/// 总是失败的假传输层。
struct FailingTransport;

impl HttpTransport for FailingTransport {
    fn post(&self, _request: &HttpRequest) -> Result<String, HttpError> {
        Err(HttpError::new("connection refused"))
    }
}

#[test]
fn provider_streams_text_from_sse_body() {
    let body = concat!(
        "data: {\"id\":\"cmpl_1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"你好\"}}]}\n\n",
        "data: {\"id\":\"cmpl_1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"，世界\"}}]}\n\n",
        "data: {\"id\":\"cmpl_1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let provider = OpenAiCompletionsProvider::new(Box::new(FakeTransport::new(body)), "test-key");
    let context = Context {
        system_prompt: None,
        messages: vec![user_text("hi").into()],
        tools: None,
    };

    let events: Vec<AssistantMessageEvent> = provider.stream(&model(), &context).collect();

    assert!(matches!(
        events.first(),
        Some(AssistantMessageEvent::Start { .. })
    ));
    let message = done_message(&events);
    assert_eq!(message.stop_reason, StopReason::Stop);
    match &message.content[0] {
        AssistantContent::Text(text) => assert_eq!(text.text, "你好，世界"),
        other => panic!("expected text, got {other:?}"),
    }
}

#[test]
fn provider_builds_request_url_headers_and_body() {
    let transport = FakeTransport::new("data: [DONE]\n\n");
    let provider = OpenAiCompletionsProvider::new(Box::new(transport.clone()), "test-key");
    let context = Context {
        system_prompt: Some("sys".to_owned()),
        messages: vec![user_text("hi").into()],
        tools: None,
    };

    let _: Vec<AssistantMessageEvent> = provider.stream(&model(), &context).collect();

    let request = transport
        .request()
        .expect("transport should receive a request");
    assert_eq!(request.url, "https://api.openai.com/v1/chat/completions");
    assert!(
        request
            .headers
            .iter()
            .any(|(name, value)| name == "Authorization" && value == "Bearer test-key")
    );
    assert!(
        request
            .headers
            .iter()
            .any(|(name, value)| name == "Content-Type" && value == "application/json")
    );

    let body: serde_json::Value = serde_json::from_str(&request.body).unwrap();
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["model"], json!("gpt-4o-mini"));
    assert_eq!(body["messages"][0]["role"], json!("system"));
}

#[test]
fn provider_emits_error_event_on_transport_failure() {
    let provider = OpenAiCompletionsProvider::new(Box::new(FailingTransport), "test-key");
    let context = Context {
        system_prompt: None,
        messages: vec![user_text("hi").into()],
        tools: None,
    };

    let events: Vec<AssistantMessageEvent> = provider.stream(&model(), &context).collect();

    match events.last() {
        Some(AssistantMessageEvent::Error { error, .. }) => {
            assert!(
                error
                    .error_message
                    .as_deref()
                    .unwrap_or("")
                    .contains("connection refused")
            );
        }
        other => panic!("expected error event, got {other:?}"),
    }
}

fn tool_result(tool_name: &str) -> ToolResultMessage {
    ToolResultMessage {
        role: ToolResultRole::ToolResult,
        tool_call_id: "call_1".to_owned(),
        tool_name: tool_name.to_owned(),
        content: vec![ToolResultContent::Text(TextContent {
            text: "结果".to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 1,
    }
}

fn with_compat(model: &mut Model, compat: OpenAICompletionsCompat) {
    model.compat = Some(ModelCompat::OpenaiCompletions(Box::new(compat)));
}

#[test]
fn standard_provider_uses_max_completion_tokens_and_store() {
    let request = build_request(&model(), &context(None, vec![user_text("hi").into()]));
    let value = serde_json::to_value(request).unwrap();

    assert!(value.get("max_completion_tokens").is_some());
    assert!(value.get("max_tokens").is_none());
    // 标准 OpenAI 支持 store，发送 store: false。
    assert_eq!(value["store"], false);
}

#[test]
fn deepseek_provider_uses_max_tokens_and_omits_store() {
    let mut deepseek = model();
    deepseek.provider = "deepseek".to_owned();
    deepseek.base_url = "https://api.deepseek.com/v1".to_owned();

    let request = build_request(&deepseek, &context(None, vec![user_text("hi").into()]));
    let value = serde_json::to_value(request).unwrap();

    // DeepSeek 被探测为非标准：用 max_tokens，且不发 store。
    assert!(value.get("max_tokens").is_some());
    assert!(value.get("max_completion_tokens").is_none());
    assert!(value.get("store").is_none());
}

#[test]
fn developer_role_used_for_reasoning_model() {
    let mut reasoning = model();
    reasoning.reasoning = true;

    let messages = convert_messages(&reasoning, &context(Some("你是助手"), Vec::new()));
    let value = serde_json::to_value(messages).unwrap();
    assert_eq!(value[0]["role"], "developer");

    // 非推理模型仍用 system。
    let messages = convert_messages(&model(), &context(Some("你是助手"), Vec::new()));
    let value = serde_json::to_value(messages).unwrap();
    assert_eq!(value[0]["role"], "system");
}

#[test]
fn tool_result_name_added_when_required() {
    let mut required = model();
    with_compat(
        &mut required,
        OpenAICompletionsCompat {
            requires_tool_result_name: Some(true),
            ..OpenAICompletionsCompat::default()
        },
    );

    let messages = convert_messages(&required, &context(None, vec![tool_result("read").into()]));
    let value = serde_json::to_value(messages).unwrap();

    assert_eq!(value[0]["role"], "tool");
    assert_eq!(value[0]["name"], "read");
}

#[test]
fn assistant_message_inserted_after_tool_result_when_required() {
    let mut required = model();
    with_compat(
        &mut required,
        OpenAICompletionsCompat {
            requires_assistant_after_tool_result: Some(true),
            ..OpenAICompletionsCompat::default()
        },
    );

    let messages = convert_messages(
        &required,
        &context(
            None,
            vec![tool_result("read").into(), user_text("继续").into()],
        ),
    );
    let value = serde_json::to_value(messages).unwrap();

    assert_eq!(value[0]["role"], "tool");
    assert_eq!(value[1]["role"], "assistant");
    assert_eq!(value[2]["role"], "user");
}

#[test]
fn detect_compat_recognizes_known_providers() {
    let mut openrouter = model();
    openrouter.provider = "openrouter".to_owned();
    openrouter.base_url = "https://openrouter.ai/api/v1".to_owned();
    let compat = detect_openai_completions_compat(&openrouter);
    assert_eq!(compat.thinking_format, pi_ai::ThinkingFormat::Openrouter);
    assert_eq!(
        compat.session_affinity_format,
        pi_ai::SessionAffinityFormat::Openrouter
    );
    assert_eq!(compat.max_tokens_field, MaxTokensField::MaxCompletionTokens);

    let mut nvidia = model();
    nvidia.provider = "nvidia".to_owned();
    nvidia.base_url = "https://integrate.api.nvidia.com/v1".to_owned();
    let compat = detect_openai_completions_compat(&nvidia);
    assert!(!compat.supports_store);
    assert_eq!(compat.max_tokens_field, MaxTokensField::MaxTokens);
}

#[test]
fn openrouter_routing_serialized_into_provider_field() {
    let mut routed = model();
    routed.provider = "openrouter".to_owned();
    routed.base_url = "https://openrouter.ai/api/v1".to_owned();
    with_compat(
        &mut routed,
        OpenAICompletionsCompat {
            open_router_routing: Some(OpenRouterRouting {
                only: Some(vec!["anthropic".to_owned()]),
                ..OpenRouterRouting::default()
            }),
            ..OpenAICompletionsCompat::default()
        },
    );

    let request = build_request(&routed, &context(None, vec![user_text("hi").into()]));
    let value = serde_json::to_value(request).unwrap();

    assert_eq!(value["provider"]["only"], json!(["anthropic"]));
}

#[test]
fn vercel_gateway_routing_wrapped_in_provider_options() {
    let mut routed = model();
    with_compat(
        &mut routed,
        OpenAICompletionsCompat {
            vercel_gateway_routing: Some(VercelGatewayRouting {
                order: Some(vec!["anthropic".to_owned()]),
                ..VercelGatewayRouting::default()
            }),
            ..OpenAICompletionsCompat::default()
        },
    );

    let request = build_request(&routed, &context(None, vec![user_text("hi").into()]));
    let value = serde_json::to_value(request).unwrap();

    assert_eq!(
        value["providerOptions"]["gateway"]["order"],
        json!(["anthropic"])
    );
}

#[test]
fn assistant_reasoning_content_added_for_deepseek() {
    let mut deepseek = model();
    deepseek.provider = "deepseek".to_owned();
    deepseek.base_url = "https://api.deepseek.com/v1".to_owned();

    let message = assistant(vec![AssistantContent::Text(TextContent {
        text: "回答".to_owned(),
        text_signature: None,
    })]);
    let messages = convert_messages(&deepseek, &context(None, vec![message.into()]));
    let value = serde_json::to_value(messages).unwrap();

    assert_eq!(value[0]["role"], "assistant");
    assert_eq!(value[0]["reasoning_content"], "");
}
