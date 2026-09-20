use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use pi_agent_core::{DirEntry, Entry, EnvError, Environment, Session};
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

    fn read_dir(&self, path: &str) -> Result<Vec<DirEntry>, EnvError> {
        // 内存里只存文件，从文件路径反推目录结构（与 tools 测试同一套思路）。
        let prefix = format!("{}/", path.trim_end_matches('/'));
        let files = self.files.borrow();
        let mut entries: std::collections::BTreeMap<String, bool> =
            std::collections::BTreeMap::new();
        for key in files.keys() {
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

#[test]
fn append_new_only_adds_new_lines() {
    let env = FakeEnv::default();
    let mut session = Session::new();
    session.push_message(user("a"));
    session.append_new(&env, "s.jsonl").unwrap();

    let after_first = env.read_file("s.jsonl").unwrap();
    assert_eq!(after_first.lines().count(), 1);

    session.push_message(user("b"));
    session.append_new(&env, "s.jsonl").unwrap();

    let after_second = env.read_file("s.jsonl").unwrap();
    assert_eq!(after_second.lines().count(), 2);
    // 第一行没有被重写，只是追加。
    assert!(after_second.starts_with(&after_first));

    let restored = Session::load(&env, "s.jsonl").unwrap();
    assert_eq!(restored.messages().len(), 2);
}

#[test]
fn fork_from_copies_prefix_and_can_continue() {
    let mut session = Session::new();
    session.push_message(user("a"));
    session.push_message(user("b"));
    session.push_message(user("c"));

    let fork_point = session.entries()[1].id().to_owned();
    let mut fork = session.fork_from(&fork_point).unwrap();

    assert_eq!(fork.entries().len(), 2);
    assert_eq!(fork.leaf_id(), Some(fork_point.as_str()));

    // 分叉后继续追加，父指针指向分叉点。
    fork.push_message(user("d"));
    match fork.entries().last().unwrap() {
        Entry::Message { parent_id, .. } => {
            assert_eq!(parent_id.as_deref(), Some(fork_point.as_str()));
        }
        other => panic!("expected message entry, got {other:?}"),
    }
}

#[test]
fn fork_from_unknown_id_is_error() {
    let session = Session::new();

    assert!(session.fork_from("nope").is_err());
}

#[test]
fn entry_lookup_by_id() {
    let mut session = Session::new();
    session.push_message(user("a"));

    let id = session.leaf_id().unwrap().to_owned();
    assert!(session.entry(&id).is_some());
    assert!(session.entry("missing").is_none());
}
