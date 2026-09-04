//! Core state and behavior for a Pi agent.

use pi_ai::{Message, Role};

/// Conversation state owned by an agent instance.
#[derive(Debug, Default)]
pub struct Agent {
    messages: Vec<Message>,
}

impl Agent {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_message(&mut self, role: Role, content: impl Into<String>) {
        self.messages.push(Message::new(role, content));
    }

    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
}
