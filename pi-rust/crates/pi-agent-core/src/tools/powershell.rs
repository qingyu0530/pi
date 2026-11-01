//! `powershell` 工具：在 Windows 上执行 PowerShell 命令。
//!
//! 移植原版 `packages/coding-agent/src/core/tools/powershell.ts`（简化版）：
//! 与 bash 工具共用同一套执行逻辑（参数校验、输出截断、错误判定），
//! 区别只在工具名、描述和实际用的 Shell 实现。

use pi_ai::ToolCall;
use serde_json::Value;

use crate::shell::Shell;
use crate::tool::{AgentTool, ToolError, ToolResult};

/// 在 Windows 上执行 PowerShell 命令的工具。
pub struct PowerShellTool {
    shell: Box<dyn Shell>,
}

impl PowerShellTool {
    #[must_use]
    pub fn new(shell: Box<dyn Shell>) -> Self {
        Self { shell }
    }
}

impl AgentTool for PowerShellTool {
    fn name(&self) -> &str {
        "powershell"
    }

    fn description(&self) -> &str {
        "在 Windows 上执行一条 PowerShell 命令，返回 stdout 和 stderr。\
         输出截断到末尾 2000 行或 50KB（先到者为准）。\
         可用 timeout 指定超时秒数（可选，默认不限时）。"
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "要执行的 PowerShell 命令" },
                "timeout": { "type": "number", "description": "超时秒数（可选，默认不限时）" }
            },
            "required": ["command"]
        })
    }

    fn execute(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        super::execute_shell_command(self.shell.as_ref(), call)
    }
}
