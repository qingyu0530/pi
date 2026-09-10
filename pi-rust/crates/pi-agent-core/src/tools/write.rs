//! `write` 工具：把内容写入文件。

use pi_ai::ToolCall;
use serde_json::Value;

use crate::environment::Environment;
use crate::tool::{AgentTool, ToolError, ToolResult};

/// 写入文件的工具。
pub struct WriteTool {
    env: Box<dyn Environment>,
}

impl WriteTool {
    #[must_use]
    pub fn new(env: Box<dyn Environment>) -> Self {
        Self { env }
    }
}

impl AgentTool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "把 content 写入（覆盖）指定文件。"
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "要写入的文件路径" },
                "content": { "type": "string", "description": "要写入的完整内容" }
            },
            "required": ["path", "content"]
        })
    }

    fn execute(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let path = call
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::new("缺少字符串参数 path"))?;
        let content = call
            .arguments
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::new("缺少字符串参数 content"))?;

        self.env
            .write_file(path, content)
            .map_err(|error| ToolError::new(error.message))?;

        Ok(ToolResult::text(format!(
            "已写入 {} 个字符到 {path}",
            content.chars().count()
        )))
    }
}
