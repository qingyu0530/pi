//! 流式响应的事件协议类型。
//!
//! message.rs 里的 AssistantMessage 是“最终完整的一条助手消息”；
//! 但流式响应是逐块到达的，因此需要一类“中间事件”来描述：
//! 一次流先发 start，然后不断发各种 *_delta，最后以 done 或 error 收尾。
//! 原版：export type AssistantMessageEvent = ...（types.ts 第 535 行）
//!
//! C++ 对照：这些事件可以看成一种“带类型的变体”：
//! std::variant<StartEvent, TextStartEvent, ...>
//! 但每个变体携带的字段各不相同，用 Rust 的枚举 + 结构体变体表达最直观。

use serde::{Deserialize, Serialize};

use crate::content::ToolCall;
use crate::message::{AssistantMessage, StopReason};

/// 流式响应中可能出现的一种事件。
///
/// 使用 tag = "type" 让 JSON 用 "type" 字段区分事件种类，与 content.rs 一致。
/// 大部分事件携带一个 partial：这是“到目前为止已经组装好的部分助手消息”。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AssistantMessageEvent {
    /// 流开始，第一次发送 partial。
    #[serde(rename = "start")]
    Start {
        #[serde(rename = "partial")]
        partial: AssistantMessage,
    },

    /// 新增一段文本内容块。
    #[serde(rename = "text_start")]
    TextStart {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        #[serde(rename = "partial")]
        partial: AssistantMessage,
    },
    /// 文本内容块的新增片段。
    #[serde(rename = "text_delta")]
    TextDelta {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        #[serde(rename = "delta")]
        delta: String,
        #[serde(rename = "partial")]
        partial: AssistantMessage,
    },
    /// 文本内容块结束，content 是完整文本。
    #[serde(rename = "text_end")]
    TextEnd {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        #[serde(rename = "content")]
        content: String,
        #[serde(rename = "partial")]
        partial: AssistantMessage,
    },

    /// 新增一段思考内容块。
    #[serde(rename = "thinking_start")]
    ThinkingStart {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        #[serde(rename = "partial")]
        partial: AssistantMessage,
    },
    /// 思考内容块的新增片段。
    #[serde(rename = "thinking_delta")]
    ThinkingDelta {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        #[serde(rename = "delta")]
        delta: String,
        #[serde(rename = "partial")]
        partial: AssistantMessage,
    },
    /// 思考内容块结束，content 是完整思考文本。
    #[serde(rename = "thinking_end")]
    ThinkingEnd {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        #[serde(rename = "content")]
        content: String,
        #[serde(rename = "partial")]
        partial: AssistantMessage,
    },

    /// 新增一个工具调用内容块。
    #[serde(rename = "toolcall_start")]
    ToolcallStart {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        #[serde(rename = "partial")]
        partial: AssistantMessage,
    },
    /// 工具调用参数的新增片段。
    #[serde(rename = "toolcall_delta")]
    ToolcallDelta {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        #[serde(rename = "delta")]
        delta: String,
        #[serde(rename = "partial")]
        partial: AssistantMessage,
    },
    /// 工具调用结束，tool_call 是完整的 ToolCall。
    #[serde(rename = "toolcall_end")]
    ToolcallEnd {
        #[serde(rename = "contentIndex")]
        content_index: u32,
        #[serde(rename = "toolCall")]
        tool_call: ToolCall,
        #[serde(rename = "partial")]
        partial: AssistantMessage,
    },

    /// 流成功结束，message 是最终完整消息。
    #[serde(rename = "done")]
    Done {
        #[serde(rename = "reason")]
        reason: StopReason,
        #[serde(rename = "message")]
        message: AssistantMessage,
    },
    /// 流出错或被中止，error 是带着 error 原因和 errorMessage 的最终消息。
    #[serde(rename = "error")]
    Error {
        #[serde(rename = "reason")]
        reason: StopReason,
        #[serde(rename = "error")]
        error: AssistantMessage,
    },
}
