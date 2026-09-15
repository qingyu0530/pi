use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use pi_agent_core::{AgentTool, DirEntry, EnvError, Environment, FindTool, ToolResult};
use pi_ai::{ToolCall, ToolResultContent};
use serde_json::{Value, json};

/// 内存文件系统（与 ls 测试同一套实现）。
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

        let mut entries: BTreeMap<String, bool> = BTreeMap::new();
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
fn find_matches_basename_pattern_recursively() {
    let env = FakeEnv::new()
        .with_file("a.rs", "a")
        .with_file("b.txt", "b")
        .with_file("src/c.rs", "c");
    let tool = FindTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call("find", json!({ "pattern": "*.rs" })))
        .unwrap();

    assert_eq!(text_of(&result), "a.rs\nsrc/c.rs");
}

#[test]
fn find_full_path_pattern_does_not_cross_separators() {
    let env = FakeEnv::new()
        .with_file("src/a.rs", "a")
        .with_file("src/nested/b.rs", "b")
        .with_file("other/c.rs", "c");
    let tool = FindTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call("find", json!({ "pattern": "src/*.rs" })))
        .unwrap();

    // `*` 不跨 `/`：只匹配 src 的直接子文件。
    assert_eq!(text_of(&result), "src/a.rs");
}

#[test]
fn find_double_star_matches_nested() {
    let env = FakeEnv::new()
        .with_file("src/a.rs", "a")
        .with_file("src/nested/b.rs", "b");
    let tool = FindTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call("find", json!({ "pattern": "src/**/*.rs" })))
        .unwrap();

    assert!(text_of(&result).contains("src/nested/b.rs"));
}

#[test]
fn find_respects_limit_and_reports_it() {
    let env = FakeEnv::new()
        .with_file("f0.txt", "0")
        .with_file("f1.txt", "1")
        .with_file("f2.txt", "2")
        .with_file("f3.txt", "3");
    let tool = FindTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call(
            "find",
            json!({ "pattern": "*.txt", "limit": 2 }),
        ))
        .unwrap();
    let text = text_of(&result);

    assert!(text.starts_with("f0.txt\nf1.txt"));
    assert!(!text.contains("f2.txt"));
    assert!(text.contains("已达到 2 条结果上限"));
}

#[test]
fn find_skips_git_and_node_modules() {
    let env = FakeEnv::new()
        .with_file(".git/config", "x")
        .with_file("node_modules/pkg/index.js", "x")
        .with_file("src/a.js", "a");
    let tool = FindTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call("find", json!({ "pattern": "*.js" })))
        .unwrap();

    assert_eq!(text_of(&result), "src/a.js");
}

#[test]
fn find_reports_no_matches() {
    let env = FakeEnv::new().with_file("a.txt", "a");
    let tool = FindTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call("find", json!({ "pattern": "*.zzz" })))
        .unwrap();

    assert_eq!(text_of(&result), "未找到匹配的文件");
}

#[test]
fn find_rejects_invalid_pattern() {
    let env = FakeEnv::new();
    let tool = FindTool::new(Box::new(env));

    let error = tool
        .execute(&tool_call("find", json!({ "pattern": "[" })))
        .unwrap_err();

    assert!(error.message.contains("无效的 glob 模式"));
}

#[test]
fn find_requires_pattern_argument() {
    let env = FakeEnv::new();
    let tool = FindTool::new(Box::new(env));

    let error = tool.execute(&tool_call("find", json!({}))).unwrap_err();

    assert!(error.message.contains("pattern"));
}
