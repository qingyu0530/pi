//! `grep` 工具：按正则（或字面量）搜索文件内容。
// 按正则（或字面量）搜索目录/文件的内容，返回「文件:行号: 匹配行」。支持 glob 过滤文件、忽略大小写、上下文行、条数上限。
use glob::{MatchOptions, Pattern};
use pi_ai::ToolCall;
use regex::RegexBuilder;
use serde_json::Value;

use super::walk::walk_files;
use crate::environment::Environment;
use crate::tool::{AgentTool, ToolError, ToolResult};
use crate::truncate::{DEFAULT_MAX_BYTES, GREP_MAX_LINE_LENGTH, truncate_head, truncate_line};

/// 默认最多返回多少条匹配（原版同值）。 默认最多 100 条匹配。
const DEFAULT_LIMIT: usize = 100;

/// 用于 `glob` 过滤参数的匹配选项（语义同 find）。
const MATCH_OPTIONS: MatchOptions = MatchOptions {
    case_sensitive: true,
    require_literal_separator: true,
    require_literal_leading_dot: false,
};

/// 搜索文件内容的工具。
pub struct GrepTool {
    env: Box<dyn Environment>,
}

impl GrepTool {
    #[must_use]
    pub fn new(env: Box<dyn Environment>) -> Self {
        Self { env }
    }
}

impl AgentTool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "按正则搜索文件内容，返回匹配行及其文件路径和行号。\
         输出截断到 100 条匹配或 50KB（先到者为准），单行截断到 500 字符。\
         会跳过 .git 和 node_modules 目录。"
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "搜索模式（正则，或用 literal=true 当字面量）" },
                "path": { "type": "string", "description": "要搜索的目录或文件（默认当前目录）" },
                "glob": { "type": "string", "description": "用 glob 过滤文件，如 '*.ts' 或 '**/*.spec.ts'" },
                "ignoreCase": { "type": "boolean", "description": "忽略大小写（默认 false）" },
                "literal": { "type": "boolean", "description": "把 pattern 当字面量而非正则（默认 false）" },
                "context": { "type": "integer", "description": "每条匹配前后各显示多少行（默认 0）" },
                "limit": { "type": "integer", "description": "最多返回多少条匹配（默认 100）" }
            },
            "required": ["pattern"]
        })
    }
    // 执行一次搜索。
    // 流程：取 7 个参数 → 构造正则 → 构造 glob 过滤器 → 判断 path 是目录还是单文件 → 收集目标文件 → 逐文件逐行匹配 → 拼行（含上下文）→ 截断 → 拼提示 → 返回。
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
        let glob_text = call.arguments.get("glob").and_then(Value::as_str);
        let ignore_case = call
            .arguments
            .get("ignoreCase")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let literal = call
            .arguments
            .get("literal")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let context = call
            .arguments
            .get("context")
            .and_then(Value::as_u64)
            .map_or(0, |value| value as usize);
        // 原版：limit 至少为 1。
        let limit = call
            .arguments
            .get("limit")
            .and_then(Value::as_u64)
            .map_or(DEFAULT_LIMIT, |value| value as usize)
            .max(1);

        // literal=true 时先转义，让正则引擎按字面量匹配。
        let expression = if literal {
            regex::escape(pattern_text)
        } else {
            pattern_text.to_owned()
        };
        let regex = RegexBuilder::new(&expression)
            .case_insensitive(ignore_case)
            .build()
            .map_err(|error| {
                ToolError::new(format!("无效的正则表达式 `{pattern_text}`: {error}"))
            })?;

        // 可选的 glob 文件过滤。含 '/' 时按相对路径匹配，否则按文件名。
        let glob_filter = match glob_text {
            Some(text) => Some((
                Pattern::new(text).map_err(|error| {
                    ToolError::new(format!("无效的 glob 模式 `{text}`: {error}"))
                })?,
                text.contains('/'),
            )),
            None => None,
        };

        // 先判断 path 是目录还是单个文件。read_dir 成功即为目录。
        let is_directory = self.env.read_dir(path).is_ok();
        // (读取路径, 显示路径)
        let mut targets: Vec<(String, String)> = Vec::new();
        if is_directory {
            walk_files(self.env.as_ref(), path, |relative_path, name| {
                let matched = match &glob_filter {
                    None => true,
                    Some((pattern, full_path)) => {
                        let candidate = if *full_path { relative_path } else { name };
                        pattern.matches_with(candidate, MATCH_OPTIONS)
                    }
                };
                if matched {
                    targets.push((relative_path.to_owned(), relative_path.to_owned()));
                }
                true
            })
            .map_err(|error| ToolError::new(error.message))?;
        } else {
            // 单文件：显示路径用文件名（和原版一致）。
            targets.push((path.to_owned(), basename(path)));
        }

        // 搜索状态。
        let mut output_lines: Vec<String> = Vec::new();
        let mut match_count = 0usize;
        let mut match_limit_reached = false;
        let mut lines_truncated = false;

        'files: for (read_path, display) in &targets {
            let content = match self.env.read_file(read_path) {
                Ok(content) => content,
                Err(error) => {
                    // 单文件模式下读不了就直接报错；目录模式下跳过不可读的文件。
                    if is_directory {
                        continue;
                    }
                    return Err(ToolError::new(format!(
                        "无法读取 `{read_path}`: {}",
                        error.message
                    )));
                }
            };
            // 统一换行符后再按行切分。
            let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
            let lines: Vec<&str> = normalized.split('\n').collect();

            for (index, line) in lines.iter().enumerate() {
                if !regex.is_match(line) {
                    continue;
                }
                match_count += 1;
                let line_number = index + 1;

                if context == 0 {
                    let (text, truncated) = truncate_line(line, GREP_MAX_LINE_LENGTH);
                    lines_truncated |= truncated;
                    output_lines.push(format!("{display}:{line_number}: {text}"));
                } else {
                    // 显示匹配行前后各 context 行；匹配行用 ':'，上下文行用 '-'。
                    let start = line_number.saturating_sub(context).max(1);
                    let end = (line_number + context).min(lines.len());
                    for current in start..=end {
                        let text_line = lines.get(current - 1).copied().unwrap_or("");
                        let (text, truncated) = truncate_line(text_line, GREP_MAX_LINE_LENGTH);
                        lines_truncated |= truncated;
                        if current == line_number {
                            output_lines.push(format!("{display}:{current}: {text}"));
                        } else {
                            output_lines.push(format!("{display}-{current}- {text}"));
                        }
                    }
                }

                if match_count >= limit {
                    match_limit_reached = true;
                    break 'files;
                }
            }
        }

        if match_count == 0 {
            return Ok(ToolResult::text("未找到匹配"));
        }

        let raw_output = output_lines.join("\n");
        let truncation = truncate_head(&raw_output, usize::MAX, DEFAULT_MAX_BYTES);
        let mut output = truncation.content;

        let mut notices = Vec::new();
        if match_limit_reached {
            notices.push(format!(
                "已达到 {limit} 条匹配上限，可用 limit={} 获取更多，或细化 pattern",
                limit * 2
            ));
        }
        if truncation.truncated {
            notices.push(format!("已达到 {}KB 上限", DEFAULT_MAX_BYTES / 1024));
        }
        if lines_truncated {
            notices.push(format!(
                "部分行已截断到 {GREP_MAX_LINE_LENGTH} 字符，可用 read 工具查看完整内容"
            ));
        }
        if !notices.is_empty() {
            output.push_str(&format!("\n\n[{}]", notices.join("。")));
        }

        Ok(ToolResult::text(output))
    }
}

/// 取路径的最后一段作为显示名。
fn basename(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_owned()
}
