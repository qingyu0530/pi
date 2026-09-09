use pi_ai::{
    AssistantContent, AssistantMessage, AssistantRole, Context, ConversationMessage,
    GrammarVariants, JsonSchemaStrict, StopReason, TextContent, Tool, Usage, UsageCost,
    UserMessage, UserMessageContent, UserRole,
};
use serde_json::{Map, Value, json};

fn sample_context_messages() -> Vec<ConversationMessage> {
    vec![
        UserMessage {
            role: UserRole::User,
            content: UserMessageContent::Text("帮我读一下 README".to_owned()),
            timestamp: 1_700_000_000_000,
        }
        .into(),
        AssistantMessage {
            role: AssistantRole::Assistant,
            content: vec![AssistantContent::Text(TextContent {
                text: "好的".to_owned(),
                text_signature: None,
            })],
            api: "openai-responses".to_owned(),
            provider: "openai".to_owned(),
            model: "gpt-5".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage {
                input: 10,
                output: 2,
                cache_read: 0,
                cache_write: 0,
                cache_write_1h: None,
                reasoning: None,
                total_tokens: 12,
                cost: UsageCost {
                    input: 0.001,
                    output: 0.0002,
                    cache_read: 0.0,
                    cache_write: 0.0,
                    total: 0.0012,
                },
            },
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 1_700_000_000_001,
        }
        .into(),
    ]
}

#[test]
fn context_matches_the_typescript_json_shape() {
    let context = Context {
        system_prompt: Some("你是 Pi".to_owned()),
        messages: sample_context_messages(),
        tools: Some(vec![Tool {
            name: "read_file".to_owned(),
            description: "读取一个文件".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
            constrained_sampling: None,
        }]),
    };

    let expected = json!({
        "systemPrompt": "你是 Pi",
        "messages": [
            {
                "role": "user",
                "content": "帮我读一下 README",
                "timestamp": 1700000000000_u64
            },
            {
                "role": "assistant",
                "content": [ { "type": "text", "text": "好的" } ],
                "api": "openai-responses",
                "provider": "openai",
                "model": "gpt-5",
                "usage": {
                    "input": 10,
                    "output": 2,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "totalTokens": 12,
                    "cost": {
                        "input": 0.001,
                        "output": 0.0002,
                        "cacheRead": 0.0,
                        "cacheWrite": 0.0,
                        "total": 0.0012
                    }
                },
                "stopReason": "stop",
                "timestamp": 1700000000001_u64
            }
        ],
        "tools": [
            {
                "name": "read_file",
                "description": "读取一个文件",
                "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }
            }
        ]
    });

    assert_eq!(serde_json::to_value(&context).unwrap(), expected);
    assert_eq!(
        serde_json::from_value::<Context>(expected).unwrap(),
        context
    );
}

#[test]
fn tool_with_grammar_constrained_sampling() {
    let mut variants = Map::new();
    variants.insert("openai_regex".to_owned(), Value::String("a+".to_owned()));
    let variants: GrammarVariants = serde_json::from_value(Value::Object(variants)).unwrap();

    let tool = Tool {
        name: "echo".to_owned(),
        description: "原样返回".to_owned(),
        parameters: json!({ "type": "string" }),
        constrained_sampling: Some(pi_ai::ConstrainedSamplingConfig::Grammar { variants }),
    };

    let expected = json!({
        "name": "echo",
        "description": "原样返回",
        "parameters": { "type": "string" },
        "constrainedSampling": {
            "type": "grammar",
            "variants": { "openai_regex": "a+" }
        }
    });

    assert_eq!(serde_json::to_value(&tool).unwrap(), expected);
}

#[test]
fn json_schema_constrained_sampling_uses_strict_enum() {
    let tool = Tool {
        name: "search".to_owned(),
        description: "搜索".to_owned(),
        parameters: json!({ "type": "object" }),
        constrained_sampling: Some(pi_ai::ConstrainedSamplingConfig::JsonSchema {
            strict: JsonSchemaStrict::Require,
        }),
    };

    let value = serde_json::to_value(&tool).unwrap();
    assert_eq!(
        value["constrainedSampling"]["strict"],
        Value::String("require".to_owned())
    );
}
