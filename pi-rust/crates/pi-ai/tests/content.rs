use pi_ai::{AssistantContent, ImageContent, TextContent, ThinkingContent, ToolCall, UserContent};
use serde_json::{Map, Value, json};

#[test]
fn tool_call_uses_the_typescript_wire_shape() {
    let arguments = Map::from_iter([("path".to_owned(), Value::String("README.md".to_owned()))]);
    let content = AssistantContent::ToolCall(ToolCall {
        id: "call_1".to_owned(),
        name: "read_file".to_owned(),
        arguments,
        thought_signature: None,
        namespace: None,
    });

    assert_eq!(
        serde_json::to_value(content).expect("tool call should serialize"),
        json!({
            "type": "toolCall",
            "id": "call_1",
            "name": "read_file",
            "arguments": { "path": "README.md" }
        })
    );
}

#[test]
fn role_specific_content_round_trips() {
    let user = UserContent::Image(ImageContent {
        data: "aGVsbG8=".to_owned(),
        mime_type: "image/png".to_owned(),
    });
    let assistant = vec![
        AssistantContent::Thinking(ThinkingContent {
            thinking: "Inspect the project".to_owned(),
            thinking_signature: Some("opaque".to_owned()),
            redacted: None,
        }),
        AssistantContent::Text(TextContent {
            text: "Done".to_owned(),
            text_signature: None,
        }),
    ];

    let user_json = serde_json::to_string(&user).expect("user content should serialize");
    let assistant_json =
        serde_json::to_string(&assistant).expect("assistant content should serialize");

    assert_eq!(
        serde_json::from_str::<UserContent>(&user_json).expect("user content should deserialize"),
        user
    );
    assert_eq!(
        serde_json::from_str::<Vec<AssistantContent>>(&assistant_json)
            .expect("assistant content should deserialize"),
        assistant
    );
}
