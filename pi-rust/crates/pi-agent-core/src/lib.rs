//! Core state and behavior for a Pi agent.

use pi_ai::ConversationMessage;

/// Conversation state owned by an agent instance.
#[derive(Debug, Default)]
pub struct Agent {
    messages: Vec<ConversationMessage>,
}

impl Agent {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one message to the in-memory transcript.
    pub fn add_message(&mut self, message: impl Into<ConversationMessage>) {
        self.messages.push(message.into());
    }

    #[must_use]
    pub fn messages(&self) -> &[ConversationMessage] {
        &self.messages
    }
}

#[cfg(test)]
mod tests {
    use super::Agent;
    use pi_ai::{
        AssistantContent, AssistantMessage, AssistantRole, ConversationMessage, StopReason,
        TextContent, ToolCall, ToolResultContent, ToolResultMessage, ToolResultRole, Usage,
        UsageCost, UserMessage, UserMessageContent, UserRole,
    };
    use serde_json::Map;

    #[test]
    fn transcript_round_trips_user_tool_call_and_result() {
        let mut agent = Agent::new();

        agent.add_message(UserMessage {
            role: UserRole::User,
            content: UserMessageContent::Text("读取 README".to_owned()),
            timestamp: 1,
        });
        agent.add_message(AssistantMessage {
            role: AssistantRole::Assistant,
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: "call_1".to_owned(),
                name: "read_file".to_owned(),
                arguments: Map::new(),
                thought_signature: None,
                namespace: None,
            })],
            api: "faux".to_owned(),
            provider: "faux".to_owned(),
            model: "faux-1".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage {
                input: 10,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                cache_write_1h: None,
                reasoning: None,
                total_tokens: 10,
                cost: UsageCost {
                    input: 0.0,
                    output: 0.0,
                    cache_read: 0.0,
                    cache_write: 0.0,
                    total: 0.0,
                },
            },
            stop_reason: StopReason::ToolUse,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: Some(false),
            timestamp: 2,
        });
        agent.add_message(ToolResultMessage {
            role: ToolResultRole::ToolResult,
            tool_call_id: "call_1".to_owned(),
            tool_name: "read_file".to_owned(),
            content: vec![ToolResultContent::Text(TextContent {
                text: "Hello, Pi".to_owned(),
                text_signature: None,
            })],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 3,
        });

        let messages = agent.messages();
        assert_eq!(messages.len(), 3);
        assert!(matches!(messages[0], ConversationMessage::User(_)));
        assert!(matches!(messages[1], ConversationMessage::Assistant(_)));
        assert!(matches!(messages[2], ConversationMessage::ToolResult(_)));
    }
}
