use pi_ai::api::openai_completions::{ChatMessage, build_request, convert_messages};
use pi_ai::{
    AssistantContent, AssistantMessage, AssistantRole, Context, ConversationMessage, ImageContent,
    InputType, Model, ModelCost, ModelCostRates, StopReason, TextContent, Tool, ToolCall,
    ToolResultContent, ToolResultMessage, ToolResultRole, Usage, UsageCost, UserContent,
    UserMessage, UserMessageContent, UserRole,
};
use serde_json::json;

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
    let messages = convert_messages(&context(Some("你是助手"), vec![user_text("你好").into()]));

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

    let messages = convert_messages(&context(None, vec![message.into()]));

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

    let messages = convert_messages(&context(None, vec![message.into()]));

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

    let converted = convert_messages(&context(None, messages));

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

    let messages = convert_messages(&context(None, vec![result.into()]));

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
                }
            }
        }])
    );
}
