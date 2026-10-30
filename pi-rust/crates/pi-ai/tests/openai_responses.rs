use std::io::Read;

use pi_ai::api::http::{HttpError, HttpRequest, HttpTransport};
use pi_ai::api::openai_responses::{
    build_request, convert_responses_messages, resolve_responses_compat,
};
use pi_ai::{
    AssistantContent, Context, ConversationMessage, InputType, Model, ModelCost, ModelCostRates,
    ModelThinkingLevel, OpenAiResponsesProvider, Provider, RequestOptions, StopReason, TextContent,
    Tool, ToolCall, ToolResultContent, ToolResultMessage, ToolResultRole, UserMessage,
    UserMessageContent, UserRole,
};
use serde_json::json;

fn model() -> Model {
    Model {
        id: "gpt-5".to_owned(),
        name: "GPT-5".to_owned(),
        api: "openai-responses".to_owned(),
        provider: "openai".to_owned(),
        base_url: "https://api.openai.com/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![InputType::Text],
        cost: ModelCost {
            rates: ModelCostRates {
                input: 1.25,
                output: 10.0,
                cache_read: 0.125,
                cache_write: 0.0,
            },
            tiers: None,
        },
        context_window: 400_000,
        max_tokens: 128_000,
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

fn zero_usage() -> pi_ai::Usage {
    pi_ai::Usage {
        input: 0,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 0,
        cost: pi_ai::UsageCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
    }
}

// -----------------------------------------------------------------------------
// 请求方向
// -----------------------------------------------------------------------------

#[test]
fn system_prompt_uses_system_role_for_non_reasoning_model() {
    let items = convert_responses_messages(
        &model(),
        &context(Some("sys"), vec![]),
        &resolve_responses_compat(&model()),
    );

    assert_eq!(items.len(), 1);
    let value = serde_json::to_value(&items[0]).unwrap();
    assert_eq!(value["role"], "system");
    assert_eq!(value["content"], "sys");
}

#[test]
fn system_prompt_uses_developer_role_for_reasoning_model() {
    let model = reasoning_model();
    let compat = resolve_responses_compat(&model);
    let items = convert_responses_messages(&model, &context(Some("sys"), vec![]), &compat);

    let value = serde_json::to_value(&items[0]).unwrap();
    assert_eq!(value["role"], "developer");
}

#[test]
fn user_text_becomes_user_item() {
    let items = convert_responses_messages(
        &model(),
        &context(None, vec![user_text("hi").into()]),
        &resolve_responses_compat(&model()),
    );

    assert_eq!(items.len(), 1);
    let value = serde_json::to_value(&items[0]).unwrap();
    assert_eq!(value["role"], "user");
    assert_eq!(value["content"], "hi");
}

#[test]
fn user_blocks_become_input_parts() {
    let user = UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Blocks(vec![
            pi_ai::UserContent::Text(TextContent {
                text: "看图".to_owned(),
                text_signature: None,
            }),
            pi_ai::UserContent::Image(pi_ai::ImageContent {
                data: "aGk=".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ]),
        timestamp: 1,
    };

    let items = convert_responses_messages(
        &model(),
        &context(None, vec![user.into()]),
        &resolve_responses_compat(&model()),
    );

    let value = serde_json::to_value(&items[0]).unwrap();
    assert_eq!(value["content"][0]["type"], "input_text");
    assert_eq!(value["content"][0]["text"], "看图");
    assert_eq!(value["content"][1]["type"], "input_image");
    assert_eq!(
        value["content"][1]["image_url"],
        "data:image/png;base64,aGk="
    );
}

#[test]
fn assistant_text_and_tool_call_replay() {
    let assistant = pi_ai::AssistantMessage {
        role: pi_ai::AssistantRole::Assistant,
        content: vec![
            AssistantContent::Text(TextContent {
                text: "调用工具".to_owned(),
                text_signature: None,
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "call_1".to_owned(),
                name: "echo".to_owned(),
                arguments: json!({"text": "hi"}).as_object().unwrap().clone(),
                thought_signature: None,
                namespace: None,
            }),
        ],
        api: "openai-responses".to_owned(),
        provider: "openai".to_owned(),
        model: "gpt-5".to_owned(),
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
    let result = ToolResultMessage {
        role: ToolResultRole::ToolResult,
        tool_call_id: "call_1".to_owned(),
        tool_name: "echo".to_owned(),
        content: vec![ToolResultContent::Text(TextContent {
            text: "hi".to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 3,
    };

    let items = convert_responses_messages(
        &model(),
        &context(None, vec![assistant.into(), result.into()]),
        &resolve_responses_compat(&model()),
    );

    // 助手文本 -> message 项；工具调用 -> function_call 项；工具结果 -> function_call_output 项。
    assert_eq!(items.len(), 3);
    let message = serde_json::to_value(&items[0]).unwrap();
    assert_eq!(message["type"], "message");
    assert_eq!(message["role"], "assistant");
    assert_eq!(message["content"][0]["type"], "output_text");
    assert_eq!(message["content"][0]["text"], "调用工具");

    let call = serde_json::to_value(&items[1]).unwrap();
    assert_eq!(call["type"], "function_call");
    assert_eq!(call["call_id"], "call_1");
    assert_eq!(call["name"], "echo");
    assert_eq!(call["arguments"], json!({"text": "hi"}).to_string());

    let output = serde_json::to_value(&items[2]).unwrap();
    assert_eq!(output["type"], "function_call_output");
    assert_eq!(output["call_id"], "call_1");
    assert_eq!(output["output"], "hi");
}

#[test]
fn request_has_model_stream_and_tools() {
    let model = model();
    let tools = vec![Tool {
        name: "echo".to_owned(),
        description: "回显".to_owned(),
        parameters: json!({"type": "object"}),
        constrained_sampling: None,
    }];

    let request = build_request(
        &model,
        &Context {
            system_prompt: None,
            messages: vec![user_text("hi").into()],
            tools: Some(tools),
        },
        &RequestOptions::default(),
    );
    let value = serde_json::to_value(&request).unwrap();

    assert_eq!(value["model"], "gpt-5");
    assert_eq!(value["stream"], true);
    assert_eq!(value["store"], false);
    // max_output_tokens 至少 16。
    assert!(value["max_output_tokens"].as_u64().unwrap() >= 16);
    // 非推理模型不发 reasoning/include。
    assert!(value.get("reasoning").is_none());
    assert!(value.get("include").is_none());
    // 工具是扁平结构。
    assert_eq!(value["tools"][0]["type"], "function");
    assert_eq!(value["tools"][0]["name"], "echo");
    assert_eq!(value["tools"][0]["strict"], false);
}

#[test]
fn reasoning_effort_sets_reasoning_param() {
    let model = reasoning_model();
    let options = RequestOptions {
        reasoning_effort: Some(ModelThinkingLevel::High),
        ..RequestOptions::default()
    };

    let request = build_request(
        &model,
        &context(None, vec![user_text("hi").into()]),
        &options,
    );
    let value = serde_json::to_value(&request).unwrap();

    assert_eq!(value["reasoning"]["effort"], "high");
    assert_eq!(value["reasoning"]["summary"], "auto");
    assert_eq!(value["include"][0], "reasoning.encrypted_content");
}

#[test]
fn reasoning_off_effort_when_no_reasoning_effort() {
    let model = reasoning_model();

    let request = build_request(
        &model,
        &context(None, vec![user_text("hi").into()]),
        &RequestOptions::default(),
    );
    let value = serde_json::to_value(&request).unwrap();

    assert_eq!(value["reasoning"]["effort"], "none");
    assert!(value["reasoning"].get("summary").is_none());
}

#[test]
fn reasoning_uses_model_mapping() {
    let model = Model {
        thinking_level_map: Some(std::collections::HashMap::from([(
            ModelThinkingLevel::High,
            Some("high-mapped".to_owned()),
        )])),
        ..reasoning_model()
    };
    let options = RequestOptions {
        reasoning_effort: Some(ModelThinkingLevel::High),
        ..RequestOptions::default()
    };

    let request = build_request(
        &model,
        &context(None, vec![user_text("hi").into()]),
        &options,
    );
    let value = serde_json::to_value(&request).unwrap();

    assert_eq!(value["reasoning"]["effort"], "high-mapped");
}

// -----------------------------------------------------------------------------
// 响应方向：流式聚合与 Provider
// -----------------------------------------------------------------------------

/// 返回固定响应体的假传输层。
struct FakeTransport {
    body: String,
}

impl HttpTransport for FakeTransport {
    fn post(&self, _request: &HttpRequest) -> Result<Box<dyn Read>, HttpError> {
        Ok(Box::new(std::io::Cursor::new(self.body.clone())))
    }
}

fn done_message(events: &[pi_ai::AssistantMessageEvent]) -> pi_ai::AssistantMessage {
    events
        .iter()
        .rev()
        .find_map(|event| match event {
            pi_ai::AssistantMessageEvent::Done { message, .. } => Some(message.clone()),
            _ => None,
        })
        .expect("stream should end with done")
}

#[test]
fn provider_streams_text_from_responses_events() {
    let body = concat!(
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\"}}\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"你好\"}\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"，世界\"}\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"你好，世界\",\"annotations\":[]}]}},\"partial\":null}\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\",\"usage\":{\"input_tokens\":10,\"input_tokens_details\":{\"cached_tokens\":4},\"output_tokens\":20,\"output_tokens_details\":{\"reasoning_tokens\":5},\"total_tokens\":30}}}\n",
    );
    let provider = OpenAiResponsesProvider::new(
        Box::new(FakeTransport {
            body: body.to_owned(),
        }),
        "test-key",
    );

    let events: Vec<pi_ai::AssistantMessageEvent> = provider
        .stream(
            &model(),
            &context(None, vec![user_text("hi").into()]),
            &RequestOptions::default(),
        )
        .collect();

    assert!(matches!(
        events.first(),
        Some(pi_ai::AssistantMessageEvent::Start { .. })
    ));
    let message = done_message(&events);
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.response_id.as_deref(), Some("resp_1"));
    match &message.content[0] {
        AssistantContent::Text(text) => assert_eq!(text.text, "你好，世界"),
        other => panic!("expected text content, got {other:?}"),
    }
    // 用量：input = 10 - 4 缓存。
    assert_eq!(message.usage.input, 6);
    assert_eq!(message.usage.cache_read, 4);
    assert_eq!(message.usage.output, 20);
    assert_eq!(message.usage.reasoning, Some(5));
}

#[test]
fn provider_aggregates_tool_calls_and_uses_tool_use_reason() {
    let body = concat!(
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"echo\",\"arguments\":\"\"}}\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"text\\\":\"}\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"\\\"hi\\\"}\"}\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"echo\",\"arguments\":\"{\\\"text\\\":\\\"hi\\\"}\"}}\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_2\",\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"total_tokens\":15}}}\n",
    );
    let provider = OpenAiResponsesProvider::new(
        Box::new(FakeTransport {
            body: body.to_owned(),
        }),
        "test-key",
    );

    let events: Vec<pi_ai::AssistantMessageEvent> = provider
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
            assert_eq!(call.id, "call_1");
            assert_eq!(call.name, "echo");
            assert_eq!(
                call.arguments
                    .get("text")
                    .and_then(serde_json::Value::as_str),
                Some("hi")
            );
        }
        other => panic!("expected tool call, got {other:?}"),
    }
}

#[test]
fn provider_aggregates_thinking_deltas() {
    let body = concat!(
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\"}}\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"想一下\"}\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"想一下\"}]}}\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"message\",\"role\":\"assistant\"}}\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"delta\":\"答\"}\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_3\",\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"total_tokens\":15}}}\n",
    );
    let provider = OpenAiResponsesProvider::new(
        Box::new(FakeTransport {
            body: body.to_owned(),
        }),
        "test-key",
    );

    let events: Vec<pi_ai::AssistantMessageEvent> = provider
        .stream(
            &model(),
            &context(None, vec![user_text("hi").into()]),
            &RequestOptions::default(),
        )
        .collect();

    let message = done_message(&events);
    // 思考块在前、文本块在后。
    match &message.content[0] {
        AssistantContent::Thinking(thinking) => assert_eq!(thinking.thinking, "想一下"),
        other => panic!("expected thinking, got {other:?}"),
    }
    match &message.content[1] {
        AssistantContent::Text(text) => assert_eq!(text.text, "答"),
        other => panic!("expected text, got {other:?}"),
    }
}

#[test]
fn provider_ignores_unknown_events() {
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_4\"}}\n",
        "data: {\"type\":\"response.in_progress\",\"response\":{\"id\":\"resp_4\"}}\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\"}}\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"ok\"}\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_4\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n",
    );
    let provider = OpenAiResponsesProvider::new(
        Box::new(FakeTransport {
            body: body.to_owned(),
        }),
        "test-key",
    );

    let events: Vec<pi_ai::AssistantMessageEvent> = provider
        .stream(
            &model(),
            &context(None, vec![user_text("hi").into()]),
            &RequestOptions::default(),
        )
        .collect();

    let message = done_message(&events);
    match &message.content[0] {
        AssistantContent::Text(text) => assert_eq!(text.text, "ok"),
        other => panic!("expected text, got {other:?}"),
    }
}
