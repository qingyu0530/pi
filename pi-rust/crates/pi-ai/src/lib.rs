//! Model and conversation types shared by Pi providers.
// 声明 content 模块
// 因此编译器会加载同一目录下的 content.rs
mod content;
// 接入 pi-ai
mod message;
// 工具与上下文
mod context;
// 流式事件协议
mod event;
// 图片生成
mod images;
// 模型成本
mod cost;
// 兼容配置与路由
mod compat;
// 模型目录
mod model;

pub use compat::{
    AnthropicAllowedFallbackModel, BedrockCompat, CacheControlFormat, ChatTemplateKwargValue,
    ChatTemplateVar, DataCollection, DeferredToolsMode, Latency, MaxPrice, MaxTokensField,
    OpenAICompletionsCompat, OpenAIResponsesCompat, OpenRouterRouting, Percentiles, PriceValue,
    SessionAffinityFormat, Sort, ThinkingFormat, ThinkingTokenBudgetField, Throughput,
    VercelGatewayRouting,
};
pub use content::{
    AssistantContent, ImageContent, TextContent, TextPhase, TextSignature, ThinkingContent,
    ToolCall, ToolResultContent, UserContent,
};
pub use context::{
    ConstrainedSamplingConfig, Context, GrammarFormat, GrammarVariants, JsonSchemaStrict, Tool,
};
pub use cost::{ModelCost, ModelCostRates, ModelCostTier};
pub use event::AssistantMessageEvent;
pub use images::{AssistantImages, ImagesContext, ImagesStopReason};
pub use message::{
    AssistantMessage, AssistantMessageDiagnostic, AssistantRole, ConversationMessage,
    DeferredHandle, DiagnosticErrorCode, DiagnosticErrorInfo, StopReason, ToolResultMessage,
    ToolResultRole, Usage, UsageCost, UserMessage, UserMessageContent, UserRole,
};
pub use model::{
    ImagesModel, InputType, Model, ModelCompat, ModelThinkingLevel, ThinkingLevel, ThinkingLevelMap,
};
