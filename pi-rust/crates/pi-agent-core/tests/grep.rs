use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

use pi_agent_core::{AgentTool, DirEntry, EnvError, Environment, GrepTool, ToolResult};
use pi_ai::{ToolCall, ToolResultContent};
use serde_json::{Value, json};

/// 内存文件系统（与 ls/find 测试同一套实现）。
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
fn grep_reports_path_and_line_number() {
    let env = FakeEnv::new().with_file("src/a.rs", "fn main() {\n    let x = 1;\n}\n");
    let tool = GrepTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call("grep", json!({ "pattern": "fn main" })))
        .unwrap();

    assert_eq!(text_of(&result), "src/a.rs:1: fn main() {");
}

#[test]
fn grep_ignore_case_option() {
    let env = FakeEnv::new().with_file("f", "Hello World\n");
    let tool = GrepTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call(
            "grep",
            json!({ "pattern": "hello", "ignoreCase": true }),
        ))
        .unwrap();

    assert_eq!(text_of(&result), "f:1: Hello World");
}

#[test]
fn grep_literal_option_disables_regex() {
    let env = FakeEnv::new()
        .with_file("hit.txt", "a.b\n")
        .with_file("miss.txt", "axb\n");
    let tool = GrepTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call(
            "grep",
            json!({ "pattern": "a.b", "literal": true }),
        ))
        .unwrap();
    let text = text_of(&result);

    assert!(text.contains("hit.txt"));
    assert!(!text.contains("miss.txt"));
}

#[test]
fn grep_glob_filters_files() {
    let env = FakeEnv::new()
        .with_file("a.rs", "target\n")
        .with_file("b.txt", "target\n");
    let tool = GrepTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call(
            "grep",
            json!({ "pattern": "target", "glob": "*.rs" }),
        ))
        .unwrap();

    assert_eq!(text_of(&result), "a.rs:1: target");
}

#[test]
fn grep_context_shows_neighbor_lines() {
    let env = FakeEnv::new().with_file("f", "l1\nl2\nMATCH\nl4\nl5\n");
    let tool = GrepTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call(
            "grep",
            json!({ "pattern": "MATCH", "context": 1 }),
        ))
        .unwrap();

    assert_eq!(text_of(&result), "f-2- l2\nf:3: MATCH\nf-4- l4");
}

#[test]
fn grep_respects_limit_and_reports_it() {
    let env = FakeEnv::new().with_file("f", "m\nm\nm\n");
    let tool = GrepTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call("grep", json!({ "pattern": "m", "limit": 2 })))
        .unwrap();
    let text = text_of(&result);

    assert_eq!(text.matches("f:").count(), 2);
    assert!(text.contains("已达到 2 条匹配上限"));
}

#[test]
fn grep_searches_a_single_file() {
    let env = FakeEnv::new()
        .with_file("dir/a.rs", "needle\n")
        .with_file("dir/b.rs", "other\n");
    let tool = GrepTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call(
            "grep",
            json!({ "pattern": "needle", "path": "dir/a.rs" }),
        ))
        .unwrap();

    assert_eq!(text_of(&result), "a.rs:1: needle");
}

#[test]
fn grep_reports_no_matches() {
    let env = FakeEnv::new().with_file("f", "nothing here\n");
    let tool = GrepTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call("grep", json!({ "pattern": "zzz" })))
        .unwrap();

    assert_eq!(text_of(&result), "未找到匹配");
}

#[test]
fn grep_rejects_invalid_regex() {
    let env = FakeEnv::new().with_file("f", "x\n");
    let tool = GrepTool::new(Box::new(env));

    let error = tool
        .execute(&tool_call("grep", json!({ "pattern": "(" })))
        .unwrap_err();

    assert!(error.message.contains("无效的正则表达式"));
}

#[test]
fn grep_truncates_long_lines() {
    let long_line = "x".repeat(600);
    let env = FakeEnv::new().with_file("f", &format!("{long_line}\n"));
    let tool = GrepTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call("grep", json!({ "pattern": "x" })))
        .unwrap();
    let text = text_of(&result);

    assert!(text.contains("... [truncated]"));
    assert!(text.contains("部分行已截断"));
}
