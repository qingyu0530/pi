//! Model and conversation types shared by Pi providers.
// 声明 content 模块
// 因此编译器会加载同一目录下的 content.rs
mod content;
// 接入 pi-ai
mod message;

pub use content::{
    AssistantContent, ImageContent, TextContent, TextPhase, TextSignature, ThinkingContent,
    ToolCall, ToolResultContent, UserContent,
};
pub use message::{
    AssistantMessage, AssistantMessageDiagnostic, AssistantRole, DeferredHandle,
    ConversationMessage, DiagnosticErrorCode, DiagnosticErrorInfo, StopReason, ToolResultMessage,
    ToolResultRole, Usage, UsageCost, UserMessage, UserMessageContent, UserRole,
};

/// The author of a conversation message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A text message in an agent conversation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    #[must_use]
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// Identifies a model exposed by a provider.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelId {
    pub provider: String,
    pub model: String,
}

impl ModelId {
    #[must_use]
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }
}
