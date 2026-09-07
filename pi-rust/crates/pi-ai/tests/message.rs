use pi_ai::{
    AssistantContent, AssistantMessage, AssistantMessageDiagnostic, AssistantRole, DeferredHandle,
    DiagnosticErrorCode, DiagnosticErrorInfo, ImageContent, StopReason, TextContent,
    ThinkingContent, ToolCall, Usage, UsageCost, UserContent, UserMessage, UserMessageContent,
    UserRole,
};
use serde_json::{Map, Number, Value, json};

#[test]
fn plain_text_message_matches_the_typescript_json_shape() {
    let message = UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Text("你好，Pi".to_owned()),
        timestamp: 1_700_000_000_123,
    };
    let expected = json!({
        "role": "user",
        "content": "你好，Pi",
        "timestamp": 1700000000123_u64
    });

    assert_eq!(serde_json::to_value(&message).unwrap(), expected);
    assert_eq!(
        serde_json::from_value::<UserMessage>(expected).unwrap(),
        message
    );
}

#[test]
fn mixed_content_preserves_block_order_and_signatures() {
    // Test opaque image data preservation, not image decoding or validation.
    let image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a9WQAAAAASUVORK5CYII=";
    let message = UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Blocks(vec![
            UserContent::Text(TextContent {
                text: "描述这张图片".to_owned(),
                text_signature: Some("opaque-signature".to_owned()),
            }),
            UserContent::Image(ImageContent {
                data: image_data.to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ]),
        timestamp: 1_700_000_000_123,
    };
    let expected = json!({
        "role": "user",
        "content": [
            { "type": "text", "text": "描述这张图片", "textSignature": "opaque-signature" },
            { "type": "image", "data": image_data, "mimeType": "image/png" }
        ],
        "timestamp": 1700000000123_u64
    });

    assert_eq!(serde_json::to_value(&message).unwrap(), expected);
    assert_eq!(
        serde_json::from_value::<UserMessage>(expected).unwrap(),
        message
    );
}

#[test]
fn invalid_user_messages_are_rejected() {
    let invalid_messages = [
        json!({ "role": "assistant", "content": "hello", "timestamp": 0 }),
        json!({ "content": "hello", "timestamp": 0 }),
        json!({ "role": "user", "content": "hello" }),
        json!({ "role": "user", "content": "hello", "timestamp": -1 }),
        json!({ "role": "user", "content": 42, "timestamp": 0 }),
        json!({
            "role": "user",
            "content": [{ "type": "thinking", "thinking": "not user content" }],
            "timestamp": 0
        }),
        json!({
            "role": "user",
            "content": [{ "type": "toolCall", "id": "call_1", "name": "read", "arguments": {} }],
            "timestamp": 0
        }),
    ];

    for value in invalid_messages {
        let result = serde_json::from_value::<UserMessage>(value.clone());
        assert!(
            result.is_err(),
            "unexpectedly accepted user message: {value}"
        );
    }
}

fn example_usage() -> Usage {
    Usage {
        input: 120,
        output: 80,
        cache_read: 20,
        cache_write: 10,
        cache_write_1h: None,
        reasoning: Some(30),
        total_tokens: 210,
        cost: UsageCost {
            input: 0.001,
            output: 0.002,
            cache_read: 0.0001,
            cache_write: 0.0002,
            total: 0.0033,
        },
    }
}

#[test]
fn assistant_message_matches_the_typescript_json_shape() {
    let message = AssistantMessage {
        role: AssistantRole::Assistant,
        content: vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "Need a tool".to_owned(),
                thinking_signature: None,
                redacted: None,
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "call_1".to_owned(),
                name: "read_file".to_owned(),
                arguments: Map::from_iter([(
                    "path".to_owned(),
                    Value::String("README.md".to_owned()),
                )]),
                thought_signature: None,
                namespace: None,
            }),
        ],
        api: "openai-responses".to_owned(),
        provider: "openai".to_owned(),
        model: "gpt-5".to_owned(),
        response_model: Some("gpt-5-2026-08-07".to_owned()),
        response_id: Some("resp_1".to_owned()),
        diagnostics: None,
        usage: example_usage(),
        stop_reason: StopReason::ToolUse,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: Some(false),
        timestamp: 1_700_000_000_123,
    };

    let expected = json!({
        "role": "assistant",
        "content": [
            { "type": "thinking", "thinking": "Need a tool" },
            {
                "type": "toolCall",
                "id": "call_1",
                "name": "read_file",
                "arguments": { "path": "README.md" }
            }
        ],
        "api": "openai-responses",
        "provider": "openai",
        "model": "gpt-5",
        "responseModel": "gpt-5-2026-08-07",
        "responseId": "resp_1",
        "usage": {
            "input": 120,
            "output": 80,
            "cacheRead": 20,
            "cacheWrite": 10,
            "reasoning": 30,
            "totalTokens": 210,
            "cost": {
                "input": 0.001,
                "output": 0.002,
                "cacheRead": 0.0001,
                "cacheWrite": 0.0002,
                "total": 0.0033
            }
        },
        "stopReason": "toolUse",
        "endTurn": false,
        "timestamp": 1700000000123_u64
    });

    assert_eq!(serde_json::to_value(&message).unwrap(), expected);
    assert_eq!(
        serde_json::from_value::<AssistantMessage>(expected).unwrap(),
        message
    );
}

#[test]
fn assistant_message_preserves_diagnostics_and_deferred_handle() {
    let message = AssistantMessage {
        role: AssistantRole::Assistant,
        content: Vec::new(),
        api: "faux".to_owned(),
        provider: "faux".to_owned(),
        model: "faux-1".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Some(vec![AssistantMessageDiagnostic {
            kind: "provider_transport_failure".to_owned(),
            timestamp: 1_700_000_000_100,
            error: Some(DiagnosticErrorInfo {
                name: Some("HttpError".to_owned()),
                message: "Too many requests".to_owned(),
                stack: None,
                code: Some(DiagnosticErrorCode::Number(Number::from(429))),
            }),
            details: Some(Map::from_iter([(
                "retryAfterMs".to_owned(),
                Value::Number(Number::from(1_000)),
            )])),
        }]),
        usage: example_usage(),
        stop_reason: StopReason::Deferred,
        deferred: Some(DeferredHandle {
            provider: "faux".to_owned(),
            model_id: "faux-1".to_owned(),
            api: "faux".to_owned(),
            id: "response_1".to_owned(),
            expires_at: Some(1_700_000_100_000),
            poll_after_ms: Some(1_000),
            data: Some(json!({ "batch": "batch_1" })),
        }),
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1_700_000_000_123,
    };

    let json = serde_json::to_value(&message).unwrap();
    assert_eq!(json["diagnostics"][0]["type"], "provider_transport_failure");
    assert_eq!(json["deferred"]["modelId"], "faux-1");
    assert_eq!(
        serde_json::from_value::<AssistantMessage>(json).unwrap(),
        message
    );
}

#[test]
fn invalid_assistant_messages_are_rejected() {
    let valid = serde_json::to_value(AssistantMessage {
        role: AssistantRole::Assistant,
        content: Vec::new(),
        api: "faux".to_owned(),
        provider: "faux".to_owned(),
        model: "faux-1".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: example_usage(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    })
    .unwrap();

    let mut wrong_role = valid.clone();
    wrong_role["role"] = json!("user");
    let mut image_content = valid.clone();
    image_content["content"] =
        json!([{ "type": "image", "data": "aGVsbG8=", "mimeType": "image/png" }]);
    let mut missing_usage = valid;
    missing_usage.as_object_mut().unwrap().remove("usage");

    for value in [wrong_role, image_content, missing_usage] {
        assert!(serde_json::from_value::<AssistantMessage>(value).is_err());
    }
}
