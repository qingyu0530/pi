//! 会话持久化：把对话记录按 JSONL（每行一个 JSON）追加保存、再读回。
//!
//! 原版 `packages/agent/src/harness/session/` 是一套很完整的会话系统
//! （entries / records / lanes / branches / forks / compaction）。
//! 这里先做最小可用版：一个 append-only 的入口日志 + JSONL 读写。
//!
//! C++ 对照：`Entry` 类似一个带 `type` 标签的 `std::variant`，
//! 每个变体是一种日志记录；`Session` 是这些记录的线性容器。

use pi_ai::ConversationMessage; // 要保存的消息类型
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::environment::Environment;

/// 会话相关的错误。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionError {
    pub message: String,
}

impl SessionError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
// 让错误能用 {} 打印
impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// 会话日志里的一条记录。
///
/// 每条都带公共字段：
/// - `id`：本条记录的唯一 id；
/// - `seq`：递增序号；
/// - `parentId`：上一条记录的 id（形成链）；
/// - `timestamp`：Unix 毫秒时间。
///
/// 用 `#[serde(tag = "type")]` 让 JSON 用 `type` 字段区分种类。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Entry {
    /// 一条对话消息。
    Message {
        id: String, // 本条记录唯一 id
        seq: u64, // 递增序号
        #[serde(rename = "parentId")]
        parent_id: Option<String>, // 上一条记录的 id（链）
        timestamp: u64,
        message: ConversationMessage,
    },
    /// 一条自定义记录（应用层自己用）。
    Custom {
        id: String,
        seq: u64,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: u64,
        #[serde(rename = "customType")]
        custom_type: String, // 自定义类型名
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<Value>,
    },
}

impl Entry {
    /// 本条记录的 id。
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Message { id, .. } | Self::Custom { id, .. } => id.as_str(),
        }
    }

    /// 本条记录的序号。
    #[must_use]
    pub fn seq(&self) -> u64 {
        match self {
            Self::Message { seq, .. } | Self::Custom { seq, .. } => *seq,
        }
    }
}

/// 一次会话的记录集合。
///
/// 记录是**只追加**的：新记录接在末尾，`parent_id` 指向上一条，`seq` 递增。
#[derive(Clone, Debug, Default)]
pub struct Session {
    entries: Vec<Entry>,
    next_seq: u64,
    last_id: Option<String>,
}

impl Session {
    /// 创建空会话。
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 只读访问所有记录。
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// 追加一条消息记录，自动填 id/seq/parentId/timestamp。
    pub fn push_message(&mut self, message: ConversationMessage) {
        let seq = self.next_seq;
        let id = format!("entry-{seq}");
        self.entries.push(Entry::Message {
            id: id.clone(),
            seq,
            parent_id: self.last_id.clone(),
            timestamp: now_millis(),
            message,
        });
        self.next_seq += 1;
        self.last_id = Some(id);
    }

    /// 追加一条自定义记录。
    pub fn push_custom(&mut self, custom_type: impl Into<String>, data: Option<Value>) {
        let seq = self.next_seq;
        let id = format!("entry-{seq}");
        self.entries.push(Entry::Custom {
            id: id.clone(),
            seq,
            parent_id: self.last_id.clone(),
            timestamp: now_millis(),
            custom_type: custom_type.into(),
            data,
        });
        self.next_seq += 1;
        self.last_id = Some(id);
    }

    /// 提取会话里所有消息（按顺序），忽略自定义记录。
    #[must_use]
    pub fn messages(&self) -> Vec<ConversationMessage> {
        self.entries
            .iter()
            .filter_map(|entry| match entry {
                Entry::Message { message, .. } => Some(message.clone()),
                Entry::Custom { .. } => None,
            })
            .collect()
    }

    /// 序列化成 JSONL：每行一个记录。
    pub fn to_jsonl(&self) -> Result<String, SessionError> {
        let mut output = String::new();
        for entry in &self.entries {
            let line = serde_json::to_string(entry)
                .map_err(|error| SessionError::new(format!("序列化记录失败: {error}")))?;
            output.push_str(&line);
            output.push('\n');
        }
        Ok(output)
    }

    /// 从 JSONL 解析；空行跳过。
    pub fn from_jsonl(text: &str) -> Result<Self, SessionError> {
        let mut entries = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let entry: Entry = serde_json::from_str(line).map_err(|error| {
                SessionError::new(format!("解析第 {} 行失败: {error}", index + 1))
            })?;
            entries.push(entry);
        }

        // 恢复「下个序号」和「最后一条 id」，保证继续追加时链不断。
        let next_seq = entries
            .iter()
            .map(Entry::seq)
            .max()
            .map_or(0, |max| max + 1);
        let last_id = entries.last().map(|entry| entry.id().to_owned());
        Ok(Self {
            entries,
            next_seq,
            last_id,
        })
    }

    /// 保存到文件（通过 `Environment`，测试可换成内存实现）。
    pub fn save(&self, env: &dyn Environment, path: &str) -> Result<(), SessionError> {
        let text = self.to_jsonl()?;
        env.write_file(path, &text)
            .map_err(|error| SessionError::new(error.message))
    }

    /// 从文件加载。
    pub fn load(env: &dyn Environment, path: &str) -> Result<Self, SessionError> {
        let text = env
            .read_file(path)
            .map_err(|error| SessionError::new(error.message))?;
        Self::from_jsonl(&text)
    }
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}
