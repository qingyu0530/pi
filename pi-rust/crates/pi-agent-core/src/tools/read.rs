//! `read` 工具：读取文本文件。

use pi_ai::ToolCall;
use serde_json::Value;

use crate::environment::Environment;
use crate::tool::{AgentTool, ToolError, ToolResult};
use crate::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_head};

/// 读取文本文件的工具。
///
/// 它持有一个 `Box<dyn Environment>`，通过环境读取文件，
/// 而不是直接调用 `std::fs`，方便测试替换。
pub struct ReadTool { // 工具结构体
    env: Box<dyn Environment>,
    max_lines: usize, // 该工具实例的截断上限（可定制）。
    max_bytes: usize,
}

impl ReadTool {
    /// 用默认上限（2000 行 / 50KB）创建读取工具。
    #[must_use]
    pub fn new(env: Box<dyn Environment>) -> Self {
        Self {
            env,
            max_lines: DEFAULT_MAX_LINES,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

impl AgentTool for ReadTool { // 告诉 Agent：这个工具叫 read。模型发起的 ToolCall.name == "read" 就会命中它。
    fn name(&self) -> &str {
        "read"
    }
    // 给模型看的说明。\ 在字符串末尾是续行符（把下一行接上，不真的换行）。
    fn description(&self) -> &str {
        "读取文本文件。支持 offset（起始行，从 1 开始）和 limit（最多读取行数）。\
         输出会按 2000 行或 50KB 截断。"
    }
    // 实现 parameters
    // offset	从第几行开始读（从 1 开始）
    // limit	最多读多少
    // offset（从第几行开始，1 起）和 limit（最多几行）让模型能分块读取大文件
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "要读取的文件路径（相对或绝对）" },
                "offset": { "type": "integer", "description": "起始行号（从 1 开始）" },
                "limit": { "type": "integer", "description": "最多读取多少行" }
            },
            "required": ["path"]
        })
    }
    // 接收模型发起的工具调用 call（含 id、name、arguments），返回结果或错误
    fn execute(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let path = call  // 取必填参数 path
            .arguments // 参数 map
            .get("path") // 查 path，得到 Option<&Value
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::new("缺少字符串参数 path"))?;
        let offset = call.arguments.get("offset").and_then(Value::as_u64);
        let limit = call.arguments.get("limit").and_then(Value::as_u64);
        // 通过环境读文件
        let content = self
            .env
            .read_file(path)
            .map_err(|error| ToolError::new(error.message))?;
        let all_lines: Vec<&str> = content.split('\n').collect(); // 按换行切，total 是总行数
        let total = all_lines.len();

        // offset 是 1 起的；转换成 0 起的下标。
        let start = offset.map_or(0, |value| (value as usize).saturating_sub(1));
        if start >= total {
            return Err(ToolError::new(format!(
                "offset {} 超出文件末尾（共 {total} 行）",
                start + 1
            )));
        }
        // 切片 + 截断
        let end = limit.map_or(total, |value| (start + value as usize).min(total));
        let selected = all_lines[start..end].join("\n");
        let truncation = truncate_head(&selected, self.max_lines, self.max_bytes);

        let mut output = truncation.content;
        if truncation.truncated {
            let last_shown = start + truncation.output_lines;
            let next_offset = last_shown + 1;
            output.push_str(&format!(
                "\n\n[已显示到第 {last_shown} 行，共 {total} 行。用 offset={next_offset} 继续读取。]"
            ));
        } else if end < total {
            let remaining = total - end;
            output.push_str(&format!(
                "\n\n[文件还有 {remaining} 行。用 offset={} 继续读取。]",
                end + 1
            ));
        }

        Ok(ToolResult::text(output))
    }
}
