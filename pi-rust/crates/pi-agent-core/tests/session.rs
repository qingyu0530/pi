use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use pi_agent_core::{Entry, EnvError, Environment, Session};
use pi_ai::{
    AssistantContent, AssistantMessage, AssistantRole, ConversationMessage, StopReason,
    TextContent, Usage, UsageCost, UserMessage, UserMessageContent, UserRole,
};
use serde_json::{Value, json};

/// 内存文件系统（和 tools 测试里同一套思路）。
#[derive(Clone, Default)]
struct FakeEnv {
    files: Rc<RefCell<HashMap<String, String>>>,
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

fn user(text: &str) -> ConversationMessage {
    UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Text(text.to_owned()),
        timestamp: 1,
    }
    .into()
}

fn assistant_text(text: &str) -> ConversationMessage {
    AssistantMessage {
        role: AssistantRole::Assistant,
        content: vec![AssistantContent::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: "openai-completions".to_owned(),
        provider: "openai".to_owned(),
        model: "gpt-4o-mini".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 0,
            cost: UsageCost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        },
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 2,
    }
    .into()
}

#[test]
fn session_round_trips_through_jsonl() {
    let mut session = Session::new();
    session.push_message(user("你好"));
    session.push_message(assistant_text("你好！"));

    let jsonl = session.to_jsonl().unwrap();
    let restored = Session::from_jsonl(&jsonl).unwrap();

    assert_eq!(restored.entries().len(), 2);
    assert_eq!(restored.messages(), session.messages());
}

#[test]
fn entry_uses_expected_wire_shape() {
    let entry = Entry::Message {
        id: "entry-0".to_owned(),
        seq: 0,
        parent_id: None,
        timestamp: 123,
        message: user("hi"),
    };

    let value = serde_json::to_value(&entry).unwrap();

    assert_eq!(value["type"], json!("message"));
    assert_eq!(value["id"], json!("entry-0"));
    assert_eq!(value["seq"], json!(0));
    assert_eq!(value["parentId"], Value::Null);
    assert_eq!(value["message"]["role"], json!("user"));
    assert_eq!(value["message"]["content"], json!("hi"));
}

#[test]
fn parent_id_chains_entries_after_reload() {
    let mut session = Session::new();
    session.push_message(user("a"));
    session.push_message(user("b"));
    session.push_custom("note", Some(json!({ "k": 1 })));

    let jsonl = session.to_jsonl().unwrap();
    let mut restored = Session::from_jsonl(&jsonl).unwrap();
    restored.push_message(user("c"));

    let entries = restored.entries();
    let last = entries.last().unwrap();
    let previous = &entries[entries.len() - 2];
    match last {
        Entry::Message { parent_id, seq, .. } => {
            assert_eq!(parent_id.as_deref(), Some(previous.id()));
            assert_eq!(*seq, 3);
        }
        other => panic!("expected message entry, got {other:?}"),
    }
}

#[test]
fn from_jsonl_skips_blank_lines() {
    let jsonl = concat!(
        "\n",
        "{\"type\":\"message\",\"id\":\"entry-0\",\"seq\":0,\"parentId\":null,\"timestamp\":1,",
        "\"message\":{\"role\":\"user\",\"content\":\"hi\",\"timestamp\":1}}\n",
        "\n",
    );

    let session = Session::from_jsonl(jsonl).unwrap();

    assert_eq!(session.entries().len(), 1);
}

#[test]
fn session_saves_and_loads_via_environment() {
    let env = FakeEnv::default();
    let mut session = Session::new();
    session.push_message(user("持久化这条"));

    session.save(&env, "session.jsonl").unwrap();
    let restored = Session::load(&env, "session.jsonl").unwrap();

    assert_eq!(restored.messages(), session.messages());
}
