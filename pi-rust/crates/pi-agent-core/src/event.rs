//! Agent 运行事件（trace）：供 UI 实时更新。
//!
//! 原版：packages/agent/src/types.ts 的 `AgentEvent`。
//! 事件分三层：
//! - 整个 run：agent_start / agent_end
//! - 单个 turn：turn_start / turn_end
//! - 消息与工具：message_* / tool_execution_*
//!
//! C++ 对照：`AgentEvent` ≈ 一个 `std::variant`，
//! 每个变体对应一种事件，携带各自的数据。
//!
//! 注意：消息、事件、工具结果都比较大，直接放进枚举会让所有变体都变胖。
//! 因此这些字段统一用 `Box` 装箱，只占一个指针（与 `ConversationMessage` 的做法一致）。

use pi_ai::{AssistantMessage, AssistantMessageEvent, ConversationMessage, ToolResultMessage};
use serde_json::Value;

use crate::tool::ToolResult;

/// Agent 在一次 `run` 过程中发出的事件。
#[derive(Clone, Debug, PartialEq)]
pub enum AgentEvent {
    /// 一次 run 开始。
    AgentStart,
    /// 一次 run 结束，携带完整对话记录。
    AgentEnd { messages: Vec<ConversationMessage> },

    /// 一个 turn 开始。一个 turn = 一次助手回复 + 它的工具调用/结果。
    TurnStart,
    /// 一个 turn 结束，携带本轮助手消息与工具结果。
    TurnEnd {
        message: Box<AssistantMessage>,
        tool_results: Vec<ToolResultMessage>,
    },

    /// 一条消息开始。用户消息、助手消息、工具结果消息都会发。
    MessageStart { message: Box<ConversationMessage> },
    /// 助手消息流式更新，携带底层 `AssistantMessageEvent`。
    /// 当前部分助手消息（还没流完）
    MessageUpdate {
        message: Box<AssistantMessage>,
        event: Box<AssistantMessageEvent>,
    },
    /// 一条消息结束。
    MessageEnd { message: Box<ConversationMessage> },

    /// 一次工具执行开始。
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: Value,
    },
    /// 工具执行中的部分更新（当前工具还不支持流式部分结果，暂不发出）。
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: Value,
        partial_result: Box<ToolResult>,
    },
    /// 一次工具执行结束。
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: Box<ToolResult>,
        is_error: bool,
    },
}
