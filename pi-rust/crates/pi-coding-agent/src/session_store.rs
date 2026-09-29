//! 会话文件目录管理：列出、选择最近、生成新路径。
//!
//! 会话文件都放在一个目录下，文件名带时间戳前缀（`<timestamp>_<id>.jsonl`），
//! 所以「最近的会话」就是名字排序后的最后一个。

use pi_agent_core::Environment;

/// 默认会话目录（可用 `PI_SESSION_DIR` 或 `--session-dir` 覆盖）。
pub const DEFAULT_SESSION_DIR: &str = "pi-sessions";

/// 列出目录下的会话文件（`.jsonl`），按名字升序返回完整路径。
pub fn list_sessions(env: &dyn Environment, dir: &str) -> Result<Vec<String>, String> {
    let entries = env.read_dir(dir).map_err(|error| error.message)?;
    let mut files: Vec<String> = entries
        .into_iter()
        .filter(|entry| !entry.is_dir && entry.name.ends_with(".jsonl"))
        .map(|entry| format!("{dir}/{}", entry.name))
        .collect();
    files.sort();
    Ok(files)
}

/// 目录里最近的会话文件；目录不存在或没有会话时返回 `None`。
/// 返回目录里名字最大的那个会话文件路径；没有会话或目录不存在时返回 None。
pub fn most_recent_session(env: &dyn Environment, dir: &str) -> Option<String> {
    // 目录不存在等价于「还没有任何会话」，算 None，不当错误。
    list_sessions(env, dir)
        .ok()
        .and_then(|files| files.into_iter().last())
}

/// 生成新会话文件路径：`<dir>/<timestamp>_<id>.jsonl`。
#[must_use]
pub fn new_session_path(dir: &str, timestamp: u64, id: &str) -> String {
    format!("{dir}/{timestamp}_{id}.jsonl")
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    use pi_agent_core::{DirEntry, EnvError, Environment};

    use super::{list_sessions, most_recent_session, new_session_path};

    /// 只存文件的内存环境（够用来测试会话目录管理）。
    #[derive(Clone, Default)]
    struct FakeEnv {
        files: Rc<RefCell<HashMap<String, String>>>,
    }

    impl FakeEnv {
        fn with_file(self, path: &str) -> Self {
            self.files
                .borrow_mut()
                .insert(path.to_owned(), String::new());
            self
        }
    }

    impl Environment for FakeEnv {
        fn read_file(&self, path: &str) -> Result<String, EnvError> {
            self.files
                .borrow()
                .get(path)
                .cloned()
                .ok_or_else(|| EnvError::new(format!("文件不存在: {path}")))
        }

        fn write_file(&self, path: &str, content: &str) -> Result<(), EnvError> {
            self.files
                .borrow_mut()
                .insert(path.to_owned(), content.to_owned());
            Ok(())
        }

        fn read_dir(&self, path: &str) -> Result<Vec<DirEntry>, EnvError> {
            let prefix = format!("{}/", path.trim_end_matches('/'));
            let mut entries: std::collections::BTreeMap<String, bool> =
                std::collections::BTreeMap::new();
            for key in self.files.borrow().keys() {
                let Some(rest) = key.strip_prefix(&prefix) else {
                    continue;
                };
                match rest.split_once('/') {
                    Some((name, _)) => {
                        entries.insert(name.to_owned(), true);
                    }
                    None => {
                        entries.insert(rest.to_owned(), false);
                    }
                }
            }
            if entries.is_empty() {
                return Err(EnvError::new(format!("目录不存在: {path}")));
            }
            Ok(entries
                .into_iter()
                .map(|(name, is_dir)| DirEntry { name, is_dir })
                .collect())
        }
    }

    #[test]
    fn lists_only_jsonl_files_sorted() {
        let env = FakeEnv::default()
            .with_file("s/200_b.jsonl")
            .with_file("s/100_a.jsonl")
            .with_file("s/readme.txt");

        let files = list_sessions(&env, "s").unwrap();

        assert_eq!(files, vec!["s/100_a.jsonl", "s/200_b.jsonl"]);
    }

    #[test]
    fn picks_most_recent_by_name() {
        let env = FakeEnv::default()
            .with_file("s/200_b.jsonl")
            .with_file("s/100_a.jsonl");

        assert_eq!(
            most_recent_session(&env, "s").as_deref(),
            Some("s/200_b.jsonl")
        );
    }

    #[test]
    fn missing_directory_returns_none() {
        let env = FakeEnv::default();

        assert_eq!(most_recent_session(&env, "nope"), None);
    }

    #[test]
    fn builds_new_session_path() {
        assert_eq!(
            new_session_path("s", 123, "abc"),
            "s/123_abc.jsonl".to_owned()
        );
    }
}
