//! 会话持久化：把对话记录按 JSONL（每行一个 JSON）追加保存、再读回。
//!
//! 原版 `packages/agent/src/harness/session/` 是一套很完整的会话系统
//! （entries / records / lanes / branches / forks / compaction）。
//! 这里做最小可用版：append-only 的入口日志 + JSONL 读写 + **分支（lane）**。
//!
//! 分支模型：一个会话里有若干「分支指针」（lane -> 叶条目 id），默认分支叫 `main`。
//! 新消息追加到**活动分支**的叶上；`create_lane` 从任意条目开新分支，
//! `switch_lane` 切换活动分支。条目记录里带 `lane` 字段（省略视为 main），
//! 分支指针本身也作为一条 `Lane` 记录写进日志，重放即可恢复全部分支。
//!
//! C++ 对照：`Entry` 类似一个带 `type` 标签的 `std::variant`，
//! `Session` 是这些记录的线性容器 + 一张分支指针表。

use std::collections::BTreeMap;

use pi_ai::ConversationMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::environment::Environment;

/// 默认分支名。
pub const MAIN_LANE: &str = "main";

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
/// - `parentId`：上一条记录的 id（形成链；`Lane` 记录不用它）；
/// - `timestamp`：Unix 毫秒时间。
///
/// 用 `#[serde(tag = "type")]` 让 JSON 用 `type` 字段区分种类。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Entry {
    /// 一条对话消息。
    Message {
        id: String, // 本条记录唯一 id
        seq: u64,   // 递增序号
        #[serde(rename = "parentId")]
        parent_id: Option<String>, // 上一条记录的 id（链）
        timestamp: u64,
        /// 追加到哪个分支；省略（旧格式）视为 `main`。
        #[serde(skip_serializing_if = "Option::is_none", default)]
        lane: Option<String>,
        message: ConversationMessage,
    },
    /// 一条自定义记录（应用层自己用）。
    Custom {
        id: String,
        seq: u64,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: u64,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        lane: Option<String>,
        #[serde(rename = "customType")]
        custom_type: String, // 自定义类型名
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<Value>,
    },
    /// 一条分支指针记录：创建或移动一个 lane，指向某个条目。
    Lane {
        id: String,
        seq: u64,
        timestamp: u64,
        /// 分支名。
        lane: String,
        /// 指向的条目 id；`None` 表示空分支。
        #[serde(rename = "leafId")]
        leaf_id: Option<String>,
    },
}

impl Entry {
    /// 本条记录的 id。
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Message { id, .. } | Self::Custom { id, .. } | Self::Lane { id, .. } => {
                id.as_str()
            }
        }
    }

    /// 本条记录的序号。
    #[must_use]
    pub fn seq(&self) -> u64 {
        match self {
            Self::Message { seq, .. } | Self::Custom { seq, .. } | Self::Lane { seq, .. } => *seq,
        }
    }
}

/// 一次会话的记录集合 + 分支指针表。
///
/// 记录是**只追加**的：新记录接在末尾；`Message`/`Custom` 追加到活动分支的叶上，
/// 分支指针通过 `Lane` 记录变更。
#[derive(Clone, Debug)]
pub struct Session {
    entries: Vec<Entry>,
    next_seq: u64,
    /// 分支指针表：分支名 -> 叶条目 id（`None` 表示空分支）。
    lanes: BTreeMap<String, Option<String>>,
    /// 当前追加写入的分支（切换是内存操作，不写日志）。
    active_lane: String,
    /// 已经写入文件的条目数；`append_new` 只写它之后的新条目。
    persisted: usize,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            next_seq: 0,
            lanes: BTreeMap::from([(MAIN_LANE.to_owned(), None)]),
            active_lane: MAIN_LANE.to_owned(),
            persisted: 0,
        }
    }
}

impl Session {
    /// 创建空会话（只有空的 main 分支）。
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 只读访问所有记录。
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// 所有分支指针（分支名 -> 叶条目 id）。
    #[must_use]
    pub fn lanes(&self) -> &BTreeMap<String, Option<String>> {
        &self.lanes
    }

    /// 当前活动分支名。
    #[must_use]
    pub fn active_lane(&self) -> &str {
        &self.active_lane
    }

    /// 活动分支的叶条目 id；空分支返回 `None`。
    #[must_use]
    pub fn leaf_id(&self) -> Option<&str> {
        self.lanes.get(&self.active_lane).and_then(Option::as_deref)
    }

    /// 追加一条消息记录到活动分支，自动填 id/seq/parentId/timestamp。
    pub fn push_message(&mut self, message: ConversationMessage) {
        let seq = self.next_seq;
        let id = format!("entry-{seq}");
        let parent_id = self.lanes.get(&self.active_lane).cloned().flatten();
        self.entries.push(Entry::Message {
            id: id.clone(),
            seq,
            parent_id,
            timestamp: now_millis(),
            // main 分支省略 lane 字段，保持与旧格式一致。
            lane: (self.active_lane != MAIN_LANE).then(|| self.active_lane.clone()),
            message,
        });
        self.next_seq += 1;
        self.lanes.insert(self.active_lane.clone(), Some(id));
    }

    /// 追加一条自定义记录到活动分支。
    pub fn push_custom(&mut self, custom_type: impl Into<String>, data: Option<Value>) {
        let seq = self.next_seq;
        let id = format!("entry-{seq}");
        let parent_id = self.lanes.get(&self.active_lane).cloned().flatten();
        self.entries.push(Entry::Custom {
            id: id.clone(),
            seq,
            parent_id,
            timestamp: now_millis(),
            lane: (self.active_lane != MAIN_LANE).then(|| self.active_lane.clone()),
            custom_type: custom_type.into(),
            data,
        });
        self.next_seq += 1;
        self.lanes.insert(self.active_lane.clone(), Some(id));
    }

    /// 从指定条目创建一个新分支（不切换活动分支）。
    ///
    /// 分支指针作为一条 `Lane` 记录写进日志，落盘后可恢复。
    pub fn create_lane(&mut self, name: &str, at: Option<&str>) -> Result<(), SessionError> {
        if self.lanes.contains_key(name) {
            return Err(SessionError::new(format!("分支已存在: {name}")));
        }
        if let Some(target) = at {
            if self.entry(target).is_none() {
                return Err(SessionError::new(format!("找不到记录: {target}")));
            }
        }
        let seq = self.next_seq;
        let id = format!("entry-{seq}");
        self.entries.push(Entry::Lane {
            id,
            seq,
            timestamp: now_millis(),
            lane: name.to_owned(),
            leaf_id: at.map(str::to_owned),
        });
        self.next_seq += 1;
        self.lanes.insert(name.to_owned(), at.map(str::to_owned));
        Ok(())
    }

    /// 切换活动分支（内存操作，不写日志；落盘后默认回到 main）。
    pub fn switch_lane(&mut self, name: &str) -> Result<(), SessionError> {
        if !self.lanes.contains_key(name) {
            return Err(SessionError::new(format!("分支不存在: {name}")));
        }
        self.active_lane = name.to_owned();
        Ok(())
    }

    /// 把活动分支切回 main。
    pub fn switch_to_main(&mut self) {
        self.active_lane = MAIN_LANE.to_owned();
    }

    /// 从指定记录处分叉：返回一个新会话，包含从开头到该记录（含）的前缀链。
    ///
    /// 返回的会话 `persisted = 0`，表示它的内容还没写到任何文件。
    pub fn fork_from(&self, entry_id: &str) -> Result<Self, SessionError> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.id() == entry_id)
            .ok_or_else(|| SessionError::new(format!("找不到记录: {entry_id}")))?;
        let entries = self.entries[..=index].to_vec();
        let next_seq = entries
            .iter()
            .map(Entry::seq)
            .max()
            .map_or(0, |max| max + 1);
        let last_id = entries.last().map(|entry| entry.id().to_owned());
        Ok(Self {
            entries,
            next_seq,
            lanes: BTreeMap::from([(MAIN_LANE.to_owned(), last_id)]),
            active_lane: MAIN_LANE.to_owned(),
            persisted: 0,
        })
    }

    /// 按 id 查找一条记录。
    #[must_use]
    pub fn entry(&self, id: &str) -> Option<&Entry> {
        self.entries.iter().find(|entry| entry.id() == id)
    }

    /// 提取某个分支上的所有消息（沿 parent 链从叶往回走，再反转）。
    #[must_use]
    pub fn lane_messages(&self, lane: &str) -> Vec<ConversationMessage> {
        let mut chain = Vec::new();
        let mut cursor = self.lanes.get(lane).and_then(Option::clone);
        while let Some(id) = cursor {
            let Some(entry) = self.entry(&id) else {
                break;
            };
            match entry {
                Entry::Message {
                    parent_id, message, ..
                } => {
                    chain.push(message.clone());
                    cursor = parent_id.clone();
                }
                // 自定义记录在链上，但不产出消息。
                Entry::Custom { parent_id, .. } => {
                    cursor = parent_id.clone();
                }
                // 分支指针记录不在消息链上。
                Entry::Lane { .. } => break,
            }
        }
        chain.reverse();
        chain
    }

    /// 提取活动分支上的所有消息（按顺序）。
    #[must_use]
    pub fn messages(&self) -> Vec<ConversationMessage> {
        self.lane_messages(&self.active_lane.clone())
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

    /// 从 JSONL 解析；空行跳过。重放记录恢复分支指针表。
    pub fn from_jsonl(text: &str) -> Result<Self, SessionError> {
        let mut session = Session::new();
        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let entry: Entry = serde_json::from_str(line).map_err(|error| {
                SessionError::new(format!("解析第 {} 行失败: {error}", index + 1))
            })?;
            session.apply_entry(entry);
        }
        Ok(session)
    }

    /// 按日志顺序重放一条记录：追加并更新分支指针表。
    fn apply_entry(&mut self, entry: Entry) {
        match &entry {
            // 分支指针记录：更新（或创建）对应分支。
            Entry::Lane { lane, leaf_id, .. } => {
                self.lanes.insert(lane.clone(), leaf_id.clone());
            }
            // 消息/自定义记录：追加到它声明的分支（省略视为 main）。
            Entry::Message { id, lane, .. } | Entry::Custom { id, lane, .. } => {
                let lane = lane.clone().unwrap_or_else(|| MAIN_LANE.to_owned());
                self.lanes.entry(lane.clone()).or_insert(None);
                if let Some(map) = self.lanes.get_mut(&lane) {
                    *map = Some(id.clone());
                }
            }
        }
        self.next_seq = self.next_seq.max(entry.seq() + 1);
        self.entries.push(entry);
        // 重放完成后默认活动分支是 main。
    }

    /// 保存到文件：整文件重写（通过 `Environment`，测试可换成内存实现）。
    ///
    /// 用于压缩后重建、或首次写盘；正常追加用 [`Session::append_new`]。
    pub fn save(&mut self, env: &dyn Environment, path: &str) -> Result<(), SessionError> {
        let text = self.to_jsonl()?;
        env.write_file(path, &text)
            .map_err(|error| SessionError::new(error.message))?;
        self.persisted = self.entries.len();
        Ok(())
    }

    /// 只把尚未写入文件的条目追加到文件末尾。
    ///
    /// 利用 `Environment::append_file`，正常每轮对话只追加新增的几行，不重写历史。
    pub fn append_new(&mut self, env: &dyn Environment, path: &str) -> Result<(), SessionError> {
        if self.persisted >= self.entries.len() {
            return Ok(());
        }
        let mut text = String::new();
        for entry in &self.entries[self.persisted..] {
            let line = serde_json::to_string(entry)
                .map_err(|error| SessionError::new(format!("序列化记录失败: {error}")))?;
            text.push_str(&line);
            text.push('\n');
        }
        env.append_file(path, &text)
            .map_err(|error| SessionError::new(error.message))?;
        self.persisted = self.entries.len();
        Ok(())
    }

    /// 从文件加载。
    pub fn load(env: &dyn Environment, path: &str) -> Result<Self, SessionError> {
        let text = env
            .read_file(path)
            .map_err(|error| SessionError::new(error.message))?;
        let mut session = Self::from_jsonl(&text)?;
        session.persisted = session.entries.len();
        Ok(session)
    }
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}
