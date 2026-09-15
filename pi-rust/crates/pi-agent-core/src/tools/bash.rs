//! `bash` 工具：执行 shell 命令。
// 实现 bash 工具：模型调用时执行一条 shell 命令，把输出（末尾截断后）返回；超时或非 0 退出码算错误。
use std::time::Duration;

use pi_ai::ToolCall;
use serde_json::Value;

use crate::shell::Shell;
use crate::tool::{AgentTool, ToolError, ToolResult};
use crate::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_tail};

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
    // 工具主逻辑。取 command 和可选 timeout → 校验 timeout → 调 shell.run → 把输出按末尾截断 → 超时/非 0 退出码返回错误 → 否则返回文本。
    fn execute(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let command = call
            .arguments
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::new("缺少字符串参数 command"))?;

        // timeout 是可选的正数秒；给了非法值直接报错。
        let timeout_seconds = call.arguments.get("timeout").and_then(Value::as_f64);
        let timeout = match timeout_seconds {
            Some(seconds) if seconds > 0.0 && seconds.is_finite() => {
                Some(Duration::from_secs_f64(seconds))
            }
            Some(_) => return Err(ToolError::new("无效的 timeout：必须是正的秒数")),
            None => None,
        };

        let result = self
            .shell
            .run(command, timeout)
            .map_err(|error| ToolError::new(error.message))?;

        // 命令输出保留末尾：错误和结果通常在最后。
        let truncation = truncate_tail(&result.output, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
        let mut text = if truncation.content.is_empty() {
            "(无输出)".to_owned()
        } else {
            truncation.content
        };
        if truncation.truncated {
            text.push_str(&format!(
                "\n\n[已截断：仅显示最后 {} 行，共 {} 行]",
                truncation.output_lines, truncation.total_lines
            ));
        }

        // 超时和非 0 退出码都算执行失败，但把已捕获的输出一并带回。
        if result.timed_out {
            let seconds = timeout_seconds.unwrap_or(0.0);
            return Err(ToolError::new(format!(
                "{text}\n\n命令在 {seconds} 秒后超时"
            )));
        }
        if let Some(code) = result.exit_code {
            if code != 0 {
                return Err(ToolError::new(format!(
                    "{text}\n\n命令以退出码 {code} 结束"
                )));
            }
        }

        Ok(ToolResult::text(text))
    }
}
