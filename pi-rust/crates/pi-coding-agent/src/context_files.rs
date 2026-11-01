//! 项目上下文文件（AGENTS.md / CLAUDE.md）的发现与加载。
//!
//! 移植自原版 `packages/coding-agent/src/core/resource-loader.ts` 的
//! `loadContextFileFromDir` / `loadProjectContextFiles`，做了简化：
//! 不处理 git worktree 的「主仓库文件被嵌套 worktree 遮蔽」逻辑。
//!
//! 加载顺序（与原版一致）：
//! 1. 全局 agent 目录（`~/.pi/agent`）里的候选文件；
//! 2. cwd 的祖先目录链，从最外层到 cwd（越具体越靠后）。
//!
//! 同一路径只保留一份。

use std::collections::BTreeSet;
use std::path::Path;

use pi_agent_core::Environment;

/// 一个已加载的上下文文件。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextFile {
    /// 文件路径（拼进系统提示的 `path` 属性，让模型知道出处）。
    pub path: String,
    /// 去掉 BOM 后的完整内容。
    pub content: String,
}

/// 每个目录里按顺序尝试的候选文件名；第一个能读到的生效。
const CANDIDATES: [&str; 5] = [
    "AGENTS.override.md",
    "AGENTS.md",
    "AGENTS.MD",
    "CLAUDE.md",
    "CLAUDE.MD",
];

/// 去掉文件开头的 UTF-8 BOM（解码后是 U+FEFF）。
fn strip_bom(content: String) -> String {
    match content.strip_prefix('\u{feff}') {
        Some(rest) => rest.to_owned(),
        None => content,
    }
}

/// 在单个目录里找上下文文件；返回第一个能读到的候选。
///
/// 文件不存在、是目录、或读取失败都当作「该目录没有」，静默跳过
/// （原版读取失败会打 warning，这里不区分「不存在」和「读失败」）。
fn load_from_dir(env: &dyn Environment, dir: &str) -> Option<ContextFile> {
    for filename in CANDIDATES {
        let path = Path::new(dir).join(filename).to_string_lossy().into_owned();
        if let Ok(content) = env.read_file(&path) {
            return Some(ContextFile {
                path,
                content: strip_bom(content),
            });
        }
    }
    None
}

/// 全局 agent 配置目录（原版 `~/.pi/agent`）。
///
/// Windows 用 `USERPROFILE`，其余用 `HOME`；都取不到时返回 `None`。
#[must_use]
pub fn default_agent_dir() -> Option<String> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()?;
    Some(
        Path::new(&home)
            .join(".pi")
            .join("agent")
            .to_string_lossy()
            .into_owned(),
    )
}

/// 收集全局目录 + cwd 祖先链上的所有上下文文件。
///
/// 返回顺序：全局在前，然后从最外层祖先到 cwd（最具体的在最后）。
#[must_use]
pub fn load_context_files(env: &dyn Environment, cwd: &str, agent_dir: &str) -> Vec<ContextFile> {
    let mut files = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();

    if let Some(global) = load_from_dir(env, agent_dir) {
        seen.insert(global.path.clone());
        files.push(global);
    }

    // ancestors()：cwd、父目录、祖父目录……直到盘符根 / 根目录。
    // 收集顺序是「从 cwd 往外」，最后反转成「从外到 cwd」。
    let mut ancestors = Vec::new();
    for dir in Path::new(cwd).ancestors() {
        if let Some(file) = load_from_dir(env, &dir.to_string_lossy()) {
            if !seen.contains(&file.path) {
                seen.insert(file.path.clone());
                ancestors.push(file);
            }
        }
    }
    ancestors.reverse();
    files.extend(ancestors);
    files
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{ContextFile, load_context_files};
    use pi_agent_core::{EnvError, Environment};

    /// 内存文件系统：路径 -> 内容。只实现读取，够加载逻辑测试用。
    struct MemoryEnvironment {
        files: HashMap<String, String>,
    }

    impl Environment for MemoryEnvironment {
        fn read_file(&self, path: &str) -> Result<String, EnvError> {
            self.files
                .get(path)
                .cloned()
                .ok_or_else(|| EnvError::new(format!("文件不存在: {path}")))
        }

        fn write_file(&self, _path: &str, _content: &str) -> Result<(), EnvError> {
            Ok(())
        }

        fn read_dir(&self, _path: &str) -> Result<Vec<pi_agent_core::DirEntry>, EnvError> {
            Ok(Vec::new())
        }
    }

    /// 与实现相同的路径拼接方式，保证测试键与查找键一致（跨平台）。
    fn file_at(base: &str, name: &str) -> (String, String) {
        let path = std::path::Path::new(base)
            .join(name)
            .to_string_lossy()
            .into_owned();
        let content = format!("内容 of {name} @ {base}");
        (path, content)
    }

    #[test]
    fn loads_global_then_ancestors_outermost_first() {
        // 目录结构：/repo/AGENTS.md、/repo/sub/CLAUDE.md、全局 agent 目录文件。
        let global = file_at("/agent-home", "AGENTS.md");
        let outer = file_at("/repo", "AGENTS.md");
        let inner = file_at("/repo/sub", "CLAUDE.md");
        let env = MemoryEnvironment {
            files: HashMap::from([global.clone(), outer.clone(), inner.clone()]),
        };

        let files = load_context_files(&env, "/repo/sub", "/agent-home");

        // 顺序：全局 -> 最外层 -> cwd。
        assert_eq!(
            files,
            vec![
                ContextFile {
                    path: global.0,
                    content: global.1
                },
                ContextFile {
                    path: outer.0,
                    content: outer.1
                },
                ContextFile {
                    path: inner.0,
                    content: inner.1
                },
            ]
        );
    }

    #[test]
    fn override_beats_agents_md_in_same_dir() {
        let plain = file_at("/repo", "AGENTS.md");
        let override_file = file_at("/repo", "AGENTS.override.md");
        let env = MemoryEnvironment {
            files: HashMap::from([plain, override_file.clone()]),
        };

        let files = load_context_files(&env, "/repo", "/no-global");

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, override_file.0);
    }

    #[test]
    fn strips_bom() {
        let (path, _) = file_at("/repo", "AGENTS.md");
        let env = MemoryEnvironment {
            files: HashMap::from([(path.clone(), "\u{feff}带 BOM 的内容".to_owned())]),
        };

        let files = load_context_files(&env, "/repo", "/no-global");

        assert_eq!(files[0].content, "带 BOM 的内容");
    }

    #[test]
    fn missing_files_yield_empty_list() {
        let env = MemoryEnvironment {
            files: HashMap::new(),
        };

        assert!(load_context_files(&env, "/repo", "/no-global").is_empty());
    }
}
