use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use pi_agent_core::{AgentTool, BashTool, Shell, ShellError, ShellOutput, ToolResult};
use pi_ai::{ToolCall, ToolResultContent};
use serde_json::{Value, json};

/// 假 shell：不启动真实进程，按预设返回结果，并把收到的超时值记到共享单元里。
#[derive(Clone)]
struct FakeShell {
    output: String,
    exit_code: Option<i32>,
    timed_out: bool,
    last_timeout: Rc<RefCell<Option<Option<Duration>>>>,
}

impl FakeShell {
    fn new(output: &str, exit_code: Option<i32>) -> Self {
        Self {
            output: output.to_owned(),
            exit_code,
            timed_out: false,
            last_timeout: Rc::new(RefCell::new(None)),
        }
    }

    fn timed_out(output: &str) -> Self {
        Self {
            output: output.to_owned(),
            exit_code: None,
            timed_out: true,
            last_timeout: Rc::new(RefCell::new(None)),
        }
    }

    fn last_timeout(&self) -> Option<Option<Duration>> {
        *self.last_timeout.borrow()
    }
}

impl Shell for FakeShell {
    fn run(&self, _command: &str, timeout: Option<Duration>) -> Result<ShellOutput, ShellError> {
        self.last_timeout.replace(Some(timeout));
        Ok(ShellOutput {
            output: self.output.clone(),
            exit_code: self.exit_code,
            timed_out: self.timed_out,
        })
    }
}

fn tool_call(arguments: Value) -> ToolCall {
    ToolCall {
        id: "call_1".to_owned(),
        name: "bash".to_owned(),
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
fn bash_returns_output() {
    let tool = BashTool::new(Box::new(FakeShell::new("hello\nworld\n", Some(0))));

    let result = tool
        .execute(&tool_call(json!({ "command": "echo hi" })))
        .unwrap();

    assert_eq!(text_of(&result), "hello\nworld\n");
}

#[test]
fn bash_no_output_uses_placeholder() {
    let tool = BashTool::new(Box::new(FakeShell::new("", Some(0))));

    let result = tool
        .execute(&tool_call(json!({ "command": "true" })))
        .unwrap();

    assert_eq!(text_of(&result), "(无输出)");
}

#[test]
fn bash_reports_nonzero_exit() {
    let tool = BashTool::new(Box::new(FakeShell::new("boom\n", Some(1))));

    let error = tool
        .execute(&tool_call(json!({ "command": "false" })))
        .unwrap_err();

    assert!(error.message.contains("boom"));
    assert!(error.message.contains("退出码 1"));
}

#[test]
fn bash_reports_timeout() {
    let tool = BashTool::new(Box::new(FakeShell::timed_out("partial\n")));

    let error = tool
        .execute(&tool_call(json!({ "command": "sleep 1", "timeout": 2 })))
        .unwrap_err();

    assert!(error.message.contains("partial"));
    assert!(error.message.contains("超时"));
}

#[test]
fn bash_forwards_timeout_to_shell() {
    let shell = FakeShell::new("ok", Some(0));
    let tool = BashTool::new(Box::new(shell.clone()));

    tool.execute(&tool_call(json!({ "command": "sleep 1", "timeout": 2.5 })))
        .unwrap();

    assert_eq!(
        shell.last_timeout(),
        Some(Some(Duration::from_secs_f64(2.5)))
    );
}

#[test]
fn bash_truncates_tail_keeping_last_lines() {
    let mut content = String::new();
    for index in 0..2001 {
        content.push_str(&format!("l{index}\n"));
    }
    let tool = BashTool::new(Box::new(FakeShell::new(&content, Some(0))));

    let result = tool
        .execute(&tool_call(json!({ "command": "seq" })))
        .unwrap();
    let text = text_of(&result);

    assert!(!text.contains("l0\n"));
    assert!(text.contains("l2000"));
    assert!(text.contains("已截断"));
}

#[test]
fn bash_rejects_invalid_timeout() {
    let tool = BashTool::new(Box::new(FakeShell::new("x", Some(0))));

    let error = tool
        .execute(&tool_call(json!({ "command": "true", "timeout": 0 })))
        .unwrap_err();

    assert!(error.message.contains("无效的 timeout"));
}

#[test]
fn bash_requires_command() {
    let tool = BashTool::new(Box::new(FakeShell::new("x", Some(0))));

    let error = tool.execute(&tool_call(json!({}))).unwrap_err();

    assert!(error.message.contains("command"));
}

#[cfg(unix)]
#[test]
fn real_shell_runs_command() {
    let shell = pi_agent_core::RealShell;
    let result = shell.run("echo hi", None).unwrap();

    assert!(result.output.contains("hi"));
    assert_eq!(result.exit_code, Some(0));
    assert!(!result.timed_out);
}

#[cfg(unix)]
#[test]
fn real_shell_merges_stderr_and_reports_exit_code() {
    let shell = pi_agent_core::RealShell;
    let result = shell.run("echo out; echo err 1>&2; exit 3", None).unwrap();

    assert!(result.output.contains("out"));
    assert!(result.output.contains("err"));
    assert_eq!(result.exit_code, Some(3));
}

#[cfg(unix)]
#[test]
fn real_shell_times_out() {
    let shell = pi_agent_core::RealShell;
    let result = shell
        .run("sleep 5", Some(Duration::from_millis(100)))
        .unwrap();

    assert!(result.timed_out);
    assert_eq!(result.exit_code, None);
}
