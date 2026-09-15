//! `find` 工具：按 glob 模式查找文件。
// 实现 find 工具：按 glob 模式（如 *.rs、src/**/*.json）在目录树里找文件，返回相对路径列表。它复用 walk_files 遍历，用 glob::Pattern 做匹配。
use glob::{MatchOptions, Pattern};
use pi_ai::ToolCall;
use serde_json::Value;

use super::walk::walk_files;
use crate::environment::Environment;
use crate::tool::{AgentTool, ToolError, ToolResult};
use crate::truncate::{DEFAULT_MAX_BYTES, truncate_head};

/// 默认最多返回多少条结果（原版同值）。
const DEFAULT_LIMIT: usize = 1000;

/// glob 匹配选项。
///
/// - `require_literal_separator = true`：让 `*` / `?` 不跨过 `/`，
///   这样 `src/*.rs` 只匹配 `src` 的直接子文件，不会误配 `src/a/b.rs`。
/// - `case_sensitive = true`：保持大小写敏感（和原版 fd 一致）。
/// - `require_literal_leading_dot = false`：允许 `*` 匹配以 `.` 开头的隐藏文件。
const MATCH_OPTIONS: MatchOptions = MatchOptions {
    case_sensitive: true,
    require_literal_separator: true,
    require_literal_leading_dot: false,
};

/// 按 glob 模式查找文件的工具。   工具本体，持有环境
pub struct FindTool {
    env: Box<dyn Environment>,
}

impl FindTool {
    #[must_use]
    pub fn new(env: Box<dyn Environment>) -> Self {
        Self { env }
    }
}

impl AgentTool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn description(&self) -> &str {
        "按 glob 模式查找文件，例如 '*.rs'、'**/*.json'、'src/**/*.spec.ts'。\
         返回相对搜索目录的路径，输出截断到 1000 条或 50KB（先到者为准）。\
         会跳过 .git 和 node_modules 目录。"
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "glob 模式，如 '*.ts' 或 'src/**/*.rs'" },
                "path": { "type": "string", "description": "搜索起始目录（默认当前目录）" },
                "limit": { "type": "integer", "description": "最多返回多少条（默认 1000）" }
            },
            "required": ["pattern"]
        })
    }
    // 执行一次查找。流程：取参数 → 编译 glob → 遍历目录收集匹配路径（到上限提前停）→ 排序 → 字节截断 → 拼提示 → 返回。
    fn execute(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let pattern_text = call
            .arguments
            .get("pattern")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::new("缺少字符串参数 pattern"))?;
        let path = call
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or(".");
        let limit = call
            .arguments
            .get("limit")
            .and_then(Value::as_u64)
            .map_or(DEFAULT_LIMIT, |value| value as usize);

        let pattern = Pattern::new(pattern_text).map_err(|error| {
            ToolError::new(format!("无效的 glob 模式 `{pattern_text}`: {error}"))
        })?;
        // 原版规则：模式里含 '/' 时按完整相对路径匹配，否则只匹配文件名。
        let match_full_path = pattern_text.contains('/');

        let mut results = Vec::new();
        walk_files(self.env.as_ref(), path, |relative_path, name| {
            let candidate = if match_full_path { relative_path } else { name };
            if pattern.matches_with(candidate, MATCH_OPTIONS) {
                if results.len() >= limit {
                    return false;
                }
                results.push(relative_path.to_owned());
            }
            true
        })
        .map_err(|error| ToolError::new(error.message))?;

        // 排序让输出稳定（原版 fd 的顺序依赖文件系统，这里改成确定顺序）。
        results.sort();

        if results.is_empty() {
            return Ok(ToolResult::text("未找到匹配的文件"));
        }

        let result_limit_reached = results.len() >= limit;
        let raw_output = results.join("\n");
        let truncation = truncate_head(&raw_output, usize::MAX, DEFAULT_MAX_BYTES);
        let mut output = truncation.content;

        let mut notices = Vec::new();
        if result_limit_reached {
            notices.push(format!(
                "已达到 {limit} 条结果上限，可用 limit={} 获取更多，或细化 pattern",
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
