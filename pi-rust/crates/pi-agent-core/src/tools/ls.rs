//! `ls` 工具：列出目录内容。

use pi_ai::ToolCall;
use serde_json::Value;

use crate::environment::Environment;
use crate::tool::{AgentTool, ToolError, ToolResult};
use crate::truncate::{DEFAULT_MAX_BYTES, truncate_head};

/// 默认最多返回多少个条目（原版同值）。
const DEFAULT_LIMIT: usize = 500;

/// 列出目录内容的工具。  LsTool 是工具本体，内部持有一个「环境」用来读目录
pub struct LsTool {
    env: Box<dyn Environment>,
}

impl LsTool {
    #[must_use]
    pub fn new(env: Box<dyn Environment>) -> Self {
        Self { env }
    }
}

impl AgentTool for LsTool {
    fn name(&self) -> &str { // 返回工具名。
        "ls"
    }
    // 给模型看的文字说明，描述这个工具干什么、有什么限制
    fn description(&self) -> &str {
        "列出目录内容。条目按字母排序，目录带 `/` 后缀，包含隐藏文件。\
         输出截断到 500 条或 50KB（先到者为准）。"
    }
    // 声明该工具接受哪些参数及类型，会随工具定义发给模型（模型据此构造参数）。
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "要列出的目录（默认当前目录）" },
                "limit": { "type": "integer", "description": "最多返回多少条（默认 500）" }
            },
            "required": []
        })
    }
    // 工具主逻辑
    // 真正执行一次 ls
    // 输入是模型的工具调用 call，输出是 ToolResult（成功）或 ToolError（失败）。
    // 流程：取参数 → 读目录 → 排序 → 格式化（到上限）→ 字节截断 → 拼提示 → 返回。
    fn execute(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        // path 是可选的：没给就用 "."（当前目录）。
        let path = call
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or(".");
        // limit 也是可选的：没给就用默认值。
        let limit = call
            .arguments
            .get("limit")
            .and_then(Value::as_u64)
            .map_or(DEFAULT_LIMIT, |value| value as usize);

        let mut entries = self
            .env
            .read_dir(path)
            .map_err(|error| ToolError::new(error.message))?;

        // 按名字大小写不敏感排序。to_lowercase 会分配一个新 String，先算好再比较。
        entries.sort_by_key(|entry| entry.name.to_lowercase());

        // 逐个格式化成一行：目录加 "/" 后缀。到 limit 就停，并记住是否被条目数截断。
        let mut lines = Vec::new();
        let mut entry_limit_reached = false;
        for entry in entries {
            if lines.len() >= limit {
                entry_limit_reached = true;
                break;
            }
            if entry.is_dir {
                lines.push(format!("{}/", entry.name));
            } else {
                lines.push(entry.name);
            }
        }

        if lines.is_empty() {
            return Ok(ToolResult::text("(空目录)"));
        }

        // 条目数已经封顶，这里只做字节截断（行数上限给 usize::MAX 等于不限制）。
        let raw_output = lines.join("\n");
        let truncation = truncate_head(&raw_output, usize::MAX, DEFAULT_MAX_BYTES);
        let mut output = truncation.content;

        // 收集截断提示，拼在末尾，告诉模型怎么拿到更多。
        let mut notices = Vec::new();
        if entry_limit_reached {
            notices.push(format!(
                "已达到 {limit} 条上限，可用 limit={} 获取更多",
                limit * 2
            ));
        }
        if truncation.truncated {
            notices.push(format!("已达到 {}KB 上限", DEFAULT_MAX_BYTES / 1024));
        }
        if !notices.is_empty() {
            output.push_str(&format!("\n\n[{}]", notices.join("。")));
        }

        Ok(ToolResult::text(output))
    }
}
