//! 默认系统提示的构造。
//!
//! 移植自原版 `packages/coding-agent/src/core/system-prompt.ts` 的
//! `buildSystemPrompt`，做了简化：暂不包含 pi 文档路径、skills 与
//! appendSystemPrompt，这些等对应功能接入后再补。

use super::context_files::ContextFile;

/// 一行工具说明：`- 名字: 描述`。
fn tool_line((name, description): &(&str, &str)) -> String {
    format!("- {name}: {description}")
}

/// 渲染 `<project_context>` 块；没有上下文文件时返回空串。
fn context_section(files: &[ContextFile]) -> String {
    if files.is_empty() {
        return String::new();
    }
    let mut section =
        String::from("\n<project_context>\n\nProject-specific instructions and guidelines:\n\n");
    for file in files {
        section.push_str(&format!(
            "<project_instructions path=\"{}\">\n{}\n</project_instructions>\n\n",
            file.path, file.content
        ));
    }
    section.push_str("</project_context>\n");
    section
}

/// 构造系统提示。
///
/// - `custom`：`--system-prompt` 给定的自定义提示，**完全替换**默认模板
///   （与原版 customPrompt 语义一致），但上下文文件与工作目录行仍会追加；
/// - `tools`：已注册工具的 `(名字, 描述)` 清单；
/// - `context_files`：已加载的项目上下文文件（AGENTS.md 等）；
/// - `cwd`：当前工作目录（反斜杠会统一成正斜杠，与原版一致）。
#[must_use]
pub fn build_system_prompt(
    custom: Option<&str>,
    tools: &[(&str, &str)],
    context_files: &[ContextFile],
    cwd: &str,
) -> String {
    let prompt_cwd = cwd.replace('\\', "/");
    let context = context_section(context_files);

    if let Some(custom) = custom {
        // 自定义提示替换默认模板，但保留项目上下文和工作目录信息。
        return format!("{custom}\n{context}\nCurrent working directory: {prompt_cwd}\n");
    }

    // 工具清单：没有工具时明确写 (none)，避免模型凭空猜测可用能力。
    let tools_list = if tools.is_empty() {
        "(none)".to_owned()
    } else {
        tools.iter().map(tool_line).collect::<Vec<_>>().join("\n")
    };

    // 准则：只有 bash 而没有 grep/find/ls 时，引导模型用 shell 做文件操作；
    // 有专用搜索工具时不需要这条。
    let names: Vec<&str> = tools.iter().map(|(name, _)| *name).collect();
    let has = |name: &str| names.contains(&name);
    let mut guidelines: Vec<&str> = Vec::new();
    if has("bash") && !has("grep") && !has("find") && !has("ls") {
        guidelines.push("Use bash for file operations like ls, rg, find");
    }
    guidelines.push("Be concise in your responses");
    guidelines.push("Show file paths clearly when working with files");
    let guidelines_list = guidelines
        .iter()
        .map(|guideline| format!("- {guideline}"))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "You are an expert coding assistant operating inside pi, a coding agent harness. \
         You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
{tools_list}

Guidelines:
{guidelines_list}
{context}
Current working directory: {prompt_cwd}
"
    )
}

#[cfg(test)]
mod tests {
    use super::build_system_prompt;
    use crate::context_files::ContextFile;

    fn full_tools() -> Vec<(&'static str, &'static str)> {
        vec![
            ("read", "读取文本文件"),
            ("write", "写入文本文件"),
            ("edit", "精确替换文件内容"),
            ("bash", "执行 shell 命令"),
            ("ls", "列出目录内容"),
            ("find", "按 glob 查找文件"),
            ("grep", "按正则搜索文件内容"),
        ]
    }

    #[test]
    fn default_prompt_lists_tools_guidelines_and_cwd() {
        let prompt = build_system_prompt(None, &full_tools(), &[], "E:\\vscode\\pi");

        assert!(prompt.contains("You are an expert coding assistant"));
        assert!(prompt.contains("- read: 读取文本文件"));
        assert!(prompt.contains("- grep: 按正则搜索文件内容"));
        assert!(prompt.contains("- Be concise in your responses"));
        // Windows 路径统一成正斜杠。
        assert!(prompt.contains("Current working directory: E:/vscode/pi"));
        // 有 grep/find/ls 时不出现 bash 文件操作那条准则。
        assert!(!prompt.contains("Use bash for file operations"));
    }

    #[test]
    fn bash_only_tools_get_file_operation_guideline() {
        let tools = vec![("read", "读取"), ("bash", "执行命令")];
        let prompt = build_system_prompt(None, &tools, &[], "/tmp");

        assert!(prompt.contains("Use bash for file operations like ls, rg, find"));
    }

    #[test]
    fn custom_prompt_replaces_default_but_keeps_cwd() {
        let prompt =
            build_system_prompt(Some("你是一个翻译助手。"), &full_tools(), &[], "E:\\work");

        assert!(prompt.starts_with("你是一个翻译助手。"));
        assert!(prompt.contains("Current working directory: E:/work"));
        assert!(!prompt.contains("You are an expert coding assistant"));
    }

    #[test]
    fn no_tools_renders_none_placeholder() {
        let prompt = build_system_prompt(None, &[], &[], "/tmp");

        assert!(prompt.contains("Available tools:\n(none)"));
    }

    #[test]
    fn context_files_render_in_default_and_custom_prompts() {
        let files = vec![
            ContextFile {
                path: "/repo/AGENTS.md".to_owned(),
                content: "外层说明".to_owned(),
            },
            ContextFile {
                path: "/repo/sub/AGENTS.md".to_owned(),
                content: "内层说明".to_owned(),
            },
        ];

        let default_prompt = build_system_prompt(None, &full_tools(), &files, "/repo/sub");
        assert!(default_prompt.contains("<project_context>"));
        assert!(default_prompt.contains("<project_instructions path=\"/repo/AGENTS.md\">"));
        assert!(default_prompt.contains("外层说明"));
        assert!(default_prompt.contains("内层说明"));
        assert!(default_prompt.contains("</project_context>"));
        // 上下文块在准则之后、工作目录之前。
        let context_pos = default_prompt.find("<project_context>").unwrap();
        let guidelines_pos = default_prompt.find("Guidelines:").unwrap();
        let cwd_pos = default_prompt.find("Current working directory:").unwrap();
        assert!(guidelines_pos < context_pos && context_pos < cwd_pos);

        // 自定义提示也保留上下文块。
        let custom_prompt =
            build_system_prompt(Some("自定义提示。"), &full_tools(), &files, "/repo/sub");
        assert!(custom_prompt.contains("<project_instructions path=\"/repo/AGENTS.md\">"));
        assert!(custom_prompt.contains("自定义提示。"));
    }
}
