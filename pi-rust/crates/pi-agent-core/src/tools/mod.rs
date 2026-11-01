//! 内置工具实现。

mod bash;
mod edit;
mod find;
mod grep;
mod ls;
mod powershell;
mod read;
mod walk;
mod write;

pub use bash::BashTool;
pub use edit::EditTool;
pub use find::FindTool;
pub use grep::GrepTool;
pub use ls::LsTool;
pub use powershell::PowerShellTool;
pub use read::ReadTool;
pub use write::WriteTool;

use std::time::Duration;

use pi_ai::ToolCall;
use serde_json::Value;

use crate::shell::Shell;
use crate::tool::{ToolError, ToolResult};
use crate::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_tail};

/// `bash` / `powershell` 共用的执行逻辑：
/// 校验参数 → 跑命令 → 输出按末尾截断 → 超时 / 非 0 退出码算错误。
///
/// 两个工具只在名字、描述和参数说明上有差别，行为保持一致
/// （原版也是同一个 createShellToolDefinition 骨架派生的）。
pub(super) fn execute_shell_command(
    shell: &dyn Shell,
    call: &ToolCall,
) -> Result<ToolResult, ToolError> {
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

    let result = shell
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
