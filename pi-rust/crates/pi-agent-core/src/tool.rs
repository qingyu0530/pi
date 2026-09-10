//! Agent 侧的工具抽象。
//!
//! pi-ai 的 `Tool` 只是「发给模型的工具定义」（名字、描述、参数 schema）。
//! 真正执行工具的逻辑属于 Agent，所以这里定义 `AgentTool` trait：
//! 它既有定义，也有 `execute` 行为。
//!
//! C++ 对照：
//!   AgentTool ≈ 抽象基类（含纯虚函数 execute）
//!   具体工具   ≈ 继承它的子类，例如下面的 EchoTool

use pi_ai::{TextContent, ToolCall, ToolResultContent, Usage};
use serde_json::Value;

/// 工具执行失败时返回的错误。
///
/// 原版 TypeScript 里工具执行失败是「抛异常」；Rust 没有异常，
/// 所以用 `Result<ToolResult, ToolError>` 表达失败。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolError {
    pub message: String,
}

impl ToolError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// 工具执行的结果。
///
/// 原版是 `AgentToolResult<T>`，其中 details 是泛型；
/// 这里先用 `Value` 表示任意 JSON，保持简单。
#[derive(Clone, Debug, PartialEq)]
pub struct ToolResult {
    /// 返回给模型的内容（文本或图片）。
    pub content: Vec<ToolResultContent>,
    /// 供日志或 UI 使用的结构化附加数据，不会作为主要内容发给模型。
    pub details: Option<Value>,
    /// 工具自身的用量，不计入模型的上下文 token。
    pub usage: Option<Usage>,
    /// 这次执行后新增、可供模型在下一轮调用的工具名。
    pub added_tool_names: Option<Vec<String>>,
    /// 是否建议在本批工具执行后结束循环。
    pub terminate: bool,
}

impl ToolResult {
    /// 构造一个只含文本、无附加信息的结果。
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::Text(TextContent {
                text: text.into(),
                text_signature: None,
            })],
            details: None,
            usage: None,
            added_tool_names: None,
            terminate: false,
        }
    }
}

/// Agent 可执行的一种工具。
///
/// 实现者需要提供名字、描述、参数 schema，以及 `execute` 行为。
pub trait AgentTool {
    /// 工具名，对应模型发起的 `ToolCall.name`。
    fn name(&self) -> &str;

    /// 给模型看的描述，说明这个工具是干什么的。
    fn description(&self) -> &str;

    /// 参数的 JSON Schema，会随工具定义一起发给模型。
    fn parameters(&self) -> Value;

    /// 执行一次工具调用。
    ///
    /// 失败时返回 `Err`；Agent 会把它转换成一条 `is_error = true` 的工具结果消息。
    fn execute(&self, call: &ToolCall) -> Result<ToolResult, ToolError>;
}

/// 一个示例工具：把参数里的 `text` 原样回显。
///
/// 它不访问网络或文件系统，用于演示 `AgentTool` 的实现方式和测试。
pub struct EchoTool;

impl AgentTool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "回显参数中的 text 字段"
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        })
    }

    fn execute(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let text = call
            .arguments
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::new("缺少字符串参数 text"))?;
        Ok(ToolResult::text(text))
    }
}
