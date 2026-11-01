//! `bash` 工具：执行 shell 命令。
// 实现 bash 工具：模型调用时执行一条 shell 命令，把输出（末尾截断后）返回；超时或非 0 退出码算错误。
// 实际执行逻辑与 powershell 工具共用（见 tools/mod.rs 的 execute_shell_command）。
use pi_ai::ToolCall;
use serde_json::Value;

use crate::shell::Shell;
use crate::tool::{AgentTool, ToolError, ToolResult};

/// 执行 shell 命令的工具。
pub struct BashTool {
    shell: Box<dyn Shell>,
}

impl BashTool {
    #[must_use]
    pub fn new(shell: Box<dyn Shell>) -> Self {
        Self { shell }
    }
}
// 告诉 Agent 这个工具的名字、说明和参数。command 必填，timeout 可选（数字秒）。
impl AgentTool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "在当前工作目录执行一条 shell 命令，返回 stdout 和 stderr。\
         输出截断到末尾 2000 行或 50KB（先到者为准）。\
         可用 timeout 指定超时秒数（可选，默认不限时）。"
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "要执行的 shell 命令" },
                "timeout": { "type": "number", "description": "超时秒数（可选，默认不限时）" }
            },
            "required": ["command"]
        })
    }
    // 工具主逻辑：与 powershell 共用（校验 → 执行 → 截断 → 错误判定）。
    fn execute(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        super::execute_shell_command(self.shell.as_ref(), call)
    }
}
