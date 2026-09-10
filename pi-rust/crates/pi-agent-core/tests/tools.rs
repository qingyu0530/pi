use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use pi_agent_core::{AgentTool, EditTool, EnvError, Environment, ReadTool, ToolResult, WriteTool};
use pi_ai::{ToolCall, ToolResultContent};
use serde_json::{Value, json};

/// 内存文件系统，用于在不碰真实磁盘的情况下测试工具。
#[derive(Clone, Default)]
struct FakeEnv {
    files: Rc<RefCell<HashMap<String, String>>>,
}

impl FakeEnv {
    fn with_file(path: &str, content: &str) -> Self {
        let env = Self::default();
        env.files
            .borrow_mut()
            .insert(path.to_owned(), content.to_owned());
        env
    }

    fn get(&self, path: &str) -> Option<String> {
        self.files.borrow().get(path).cloned()
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
fn read_returns_file_content() {
    let env = FakeEnv::with_file("a.txt", "line1\nline2\nline3");
    let tool = ReadTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call("read", json!({ "path": "a.txt" })))
        .unwrap();

    assert_eq!(text_of(&result), "line1\nline2\nline3");
}

#[test]
fn read_supports_offset_and_limit() {
    let env = FakeEnv::with_file("f", "a\nb\nc\nd");
    let tool = ReadTool::new(Box::new(env));

    let result = tool
        .execute(&tool_call(
            "read",
            json!({ "path": "f", "offset": 2, "limit": 2 }),
        ))
        .unwrap();
    let text = text_of(&result);

    assert!(text.starts_with("b\nc"));
    assert!(text.contains("文件还有 1 行"));
}

#[test]
fn read_rejects_missing_path_argument() {
    let tool = ReadTool::new(Box::new(FakeEnv::default()));

    let error = tool.execute(&tool_call("read", json!({}))).unwrap_err();

    assert!(error.message.contains("path"));
}

#[test]
fn read_reports_offset_beyond_end() {
    let env = FakeEnv::with_file("f", "a\nb");
    let tool = ReadTool::new(Box::new(env));

    let error = tool
        .execute(&tool_call("read", json!({ "path": "f", "offset": 99 })))
        .unwrap_err();

    assert!(error.message.contains("超出文件末尾"));
}

#[test]
fn write_then_read_round_trips() {
    let env = FakeEnv::default();
    let writer = WriteTool::new(Box::new(env.clone()));

    writer
        .execute(&tool_call(
            "write",
            json!({ "path": "f", "content": "hello" }),
        ))
        .unwrap();
    assert_eq!(env.get("f").as_deref(), Some("hello"));

    let reader = ReadTool::new(Box::new(env));
    let result = reader
        .execute(&tool_call("read", json!({ "path": "f" })))
        .unwrap();
    assert_eq!(text_of(&result), "hello");
}

#[test]
fn edit_replaces_a_unique_block() {
    let env = FakeEnv::with_file("f", "let x = 1;\nlet y = 2;\n");
    let tool = EditTool::new(Box::new(env.clone()));

    tool.execute(&tool_call(
        "edit",
        json!({ "path": "f", "edits": [{ "oldText": "let x = 1;", "newText": "let x = 10;" }] }),
    ))
    .unwrap();

    assert_eq!(env.get("f").as_deref(), Some("let x = 10;\nlet y = 2;\n"));
}

#[test]
fn edit_rejects_non_unique_old_text() {
    let env = FakeEnv::with_file("f", "a\na\n");
    let tool = EditTool::new(Box::new(env));

    let error = tool
        .execute(&tool_call(
            "edit",
            json!({ "path": "f", "edits": [{ "oldText": "a", "newText": "b" }] }),
        ))
        .unwrap_err();

    assert!(error.message.contains("不唯一"));
}

#[test]
fn edit_rejects_missing_old_text() {
    let env = FakeEnv::with_file("f", "hello");
    let tool = EditTool::new(Box::new(env));

    let error = tool
        .execute(&tool_call(
            "edit",
            json!({ "path": "f", "edits": [{ "oldText": "nope", "newText": "b" }] }),
        ))
        .unwrap_err();

    assert!(error.message.contains("未找到"));
}

#[test]
fn edit_applies_multiple_non_overlapping_edits() {
    let env = FakeEnv::with_file("f", "alpha beta gamma");
    let tool = EditTool::new(Box::new(env.clone()));

    tool.execute(&tool_call(
        "edit",
        json!({
            "path": "f",
            "edits": [
                { "oldText": "alpha", "newText": "A" },
                { "oldText": "gamma", "newText": "G" }
            ]
        }),
    ))
    .unwrap();

    assert_eq!(env.get("f").as_deref(), Some("A beta G"));
}
