use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use pi_agent_core::{AgentTool, DirEntry, EnvError, Environment, LsTool, ToolResult};
use pi_ai::{ToolCall, ToolResultContent};
use serde_json::{Value, json};

/// 内存文件系统：同时记录文件和显式目录，用来测试列目录。
#[derive(Clone, Default)]
struct FakeEnv {
    files: Rc<RefCell<HashMap<String, String>>>,
    dirs: Rc<RefCell<BTreeSet<String>>>,
}

impl FakeEnv {
    fn new() -> Self {
        Self::default()
    }

    fn with_file(self, path: &str, content: &str) -> Self {
        self.files
            .borrow_mut()
            .insert(path.to_owned(), content.to_owned());
        self
    }

    fn with_dir(self, path: &str) -> Self {
        self.dirs
            .borrow_mut()
            .insert(path.trim_end_matches('/').to_owned());
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
        let normalized = path.trim_end_matches('/');
        let prefix = if normalized.is_empty() || normalized == "." {
            String::new()
        } else {
            format!("{normalized}/")
        };

        // 条目名 -> 是否为目录。BTreeMap 保证输出顺序稳定。
        let mut entries: BTreeMap<String, bool> = BTreeMap::new();

        // 从文件路径反推目录结构。
        for key in self.files.borrow().keys() {
            let rest = if prefix.is_empty() {
                key.as_str()
            } else {
                match key.strip_prefix(&prefix) {
                    Some(rest) => rest,
                    None => continue,
                }
            };
            if rest.is_empty() {
                continue;
            }
            match rest.split_once('/') {
                Some((name, _)) => {
                    entries.insert(name.to_owned(), true);
                }
                None => {
                    entries.insert(rest.to_owned(), false);
                }
            }
        }

        // 显式声明的目录（可能是空的）。
        for dir in self.dirs.borrow().iter() {
            let rest = if prefix.is_empty() {
                dir.as_str()
            } else {
                match dir.strip_prefix(&prefix) {
                    Some(rest) => rest,
                    None => continue,
                }
            };
            if rest.is_empty() {
                continue;
            }
            match rest.split_once('/') {
                Some((name, _)) => {
                    entries.insert(name.to_owned(), true);
                }
                None => {
                    entries.insert(rest.to_owned(), true);
                }
            }
        }

        // 目录必须存在：根目录、显式目录，或有文件位于其下。
        let exists = prefix.is_empty()
            || self.dirs.borrow().contains(normalized)
            || self
                .files
                .borrow()
                .keys()
                .any(|key| key.starts_with(&prefix));
        if !exists {
            return Err(EnvError::new(format!("目录不存在: {path}")));
        }

        Ok(entries
            .into_iter()
            .map(|(name, is_dir)| DirEntry { name, is_dir })
            .collect())
    }
}

fn tool_call(name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: "call_1".to_owned(),
        name: name.to_owned(),
        arguments: arguments.as_object().unwrap().clone(),
        thought_signature: None,
        namespace: None,
    }
}

fn text_of(result: &ToolResult) -> String {
    match &result.content[0] {
        ToolResultContent::Text(text) => text.text.clone(),
        other => panic!("expected text content, got {other:?}"),
    }
}

#[test]
fn ls_lists_sorted_with_dir_suffix() {
    let env = FakeEnv::new()
        .with_file("a.txt", "a")
        .with_file("B.txt", "b")
        .with_file("dir/c.txt", "c");
    let tool = LsTool::new(Box::new(env));

    let result = tool.execute(&tool_call("ls", json!({}))).unwrap();

    // 大小写不敏感排序：a.txt, B.txt, dir/。
    assert_eq!(text_of(&result), "a.txt\nB.txt\ndir/");
}

#[test]
fn ls_lists_a_subdirectory() {
    let env = FakeEnv::new()
        .with_file("dir/x.rs", "x")
        .with_file("dir/y.rs", "y");
    let tool = LsTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call("ls", json!({ "path": "dir" })))
        .unwrap();

    assert_eq!(text_of(&result), "x.rs\ny.rs");
}

#[test]
fn ls_respects_limit_and_reports_it() {
    let env = FakeEnv::new()
        .with_file("f0", "0")
        .with_file("f1", "1")
        .with_file("f2", "2")
        .with_file("f3", "3");
    let tool = LsTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call("ls", json!({ "limit": 2 })))
        .unwrap();
    let text = text_of(&result);

    assert!(text.starts_with("f0\nf1"));
    assert!(!text.contains("f2"));
    assert!(text.contains("已达到 2 条上限"));
}

#[test]
fn ls_reports_empty_directory() {
    let env = FakeEnv::new().with_dir("empty");
    let tool = LsTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call("ls", json!({ "path": "empty" })))
        .unwrap();

    assert_eq!(text_of(&result), "(空目录)");
}

#[test]
fn ls_reports_missing_directory() {
    let env = FakeEnv::new();
    let tool = LsTool::new(Box::new(env));

    let error = tool
        .execute(&tool_call("ls", json!({ "path": "nope" })))
        .unwrap_err();

    assert!(error.message.contains("目录不存在"));
}
