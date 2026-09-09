use pi_ai::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantRole, StopReason,
    TextContent, ToolCall, Usage, UsageCost,
};
use serde_json::{Map, Value, json};

fn sample_partial(text: &str) -> AssistantMessage {
    AssistantMessage {
        role: AssistantRole::Assistant,
        content: vec![AssistantContent::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: "openai-responses".to_owned(),
        provider: "openai".to_owned(),
        model: "gpt-5".to_owned(),
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
        stop_reason: StopReason::Pending,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1_700_000_000_000,
    }
}

#[test]
fn text_delta_event_uses_camel_case_field_names() {
    let event = AssistantMessageEvent::TextDelta {
        content_index: 0,
        delta: "你好".to_owned(),
        partial: sample_partial("你好"),
    };

    let expected = json!({
        "type": "text_delta",
        "contentIndex": 0,
        "delta": "你好",
        "partial": {
            "role": "assistant",
            "content": [ { "type": "text", "text": "你好" } ],
            "api": "openai-responses",
            "provider": "openai",
            "model": "gpt-5",
            "usage": {
                "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                "totalTokens": 0,
                "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 }
            },
            "stopReason": "pending",
            "timestamp": 1700000000000_u64
        }
    });

    assert_eq!(serde_json::to_value(&event).unwrap(), expected);
    assert_eq!(
        serde_json::from_value::<AssistantMessageEvent>(expected).unwrap(),
        event
    );
}

#[test]
fn toolcall_end_carries_a_full_tool_call() {
    let mut arguments = Map::new();
    arguments.insert("path".to_owned(), Value::String("README.md".to_owned()));
    let event = AssistantMessageEvent::ToolcallEnd {
        content_index: 0,
        tool_call: ToolCall {
            id: "call_1".to_owned(),
            name: "read_file".to_owned(),
            arguments,
            thought_signature: None,
            namespace: None,
        },
        partial: sample_partial(""),
    };

    let expected = json!({
        "type": "toolcall_end",
        "contentIndex": 0,
        "toolCall": {
            "id": "call_1",
            "name": "read_file",
            "arguments": { "path": "README.md" }
        },
        "partial": {
            "role": "assistant",
            "content": [ { "type": "text", "text": "" } ],
            "api": "openai-responses",
            "provider": "openai",
            "model": "gpt-5",
            "usage": {
                "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                "totalTokens": 0,
                "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 }
            },
            "stopReason": "pending",
            "timestamp": 1700000000000_u64
        }
    });

    assert_eq!(serde_json::to_value(&event).unwrap(), expected);
}

#[test]
fn done_event_uses_message_field() {
    let event = AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message: sample_partial("完成"),
    };

    let value = serde_json::to_value(&event).unwrap();
    assert_eq!(value["type"], Value::String("done".to_owned()));
    assert_eq!(value["reason"], Value::String("stop".to_owned()));
    assert!(value.get("message").is_some());
}
