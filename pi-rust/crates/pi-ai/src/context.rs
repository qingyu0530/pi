// 工具与对话上下文类型。
//
// 前面的 message.rs 定义了“对话里有哪些消息”。
// 这一节回答“把一次模型调用打包起来时，还要带上什么”：
// Tool：告诉模型“你可以调用哪些工具”，以及每个工具的参数格式。
// Context：把系统提示、消息列表、工具列表组合成一次完整请求的输入。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::ConversationMessage;

/// OpenAI 语法变体，用于受约束采样。
// OpenAI 风格的语法格式有两种变体——openai_lark（Lark 语法）和 openai_regex
/// 原版：export type GrammarFormat = "openai_lark" | "openai_regex";
/// Rust 用枚举表达“只能取两个固定值”的联合类型。
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum GrammarFormat {
    // lark 语法
    #[serde(rename = "openai_lark")]
    OpenaiLark,
    // 正则语法
    #[serde(rename = "openai_regex")]
    OpenaiRegex,
}

/// 语法变体到字符串的映射。
/// 原版：export type GrammarVariants = Partial<Record<GrammarFormat, string>>;
/// Rust 用 HashMap<GrammarFormat, String> 表达“键是固定枚举、值是字符串”的映射。
// 字符串是：它是限定模型输出格式的语法规则。
// 值必须是 String：语法规则是任意的文本，长短、格式都不固定，
// 没有结构化的 Rust 类型能天然表示它，所以用一个字符串原样保存。
/// 原版的 Partial 表示某个键可以缺失；HashMap 本身允许键不存在。
pub type GrammarVariants = HashMap<GrammarFormat, String>;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum JsonSchemaStrict {
    // Prefer 和 Require 是 strict（严格度）的两个取值，表
    // 示模型遵守 JSON Schema 的强制程度
    // 相当于一个二选一的开关，告诉 Provider“这个 schema 是必须遵守，
    // 还是仅供参考”。
    #[serde(rename = "prefer")]
    Prefer,
    #[serde(rename = "require")]
    Require,
}

/// 可选的服务商侧受约束采样配置。
/// 原版是一个联合类型：
/// | { type: "json_schema"; strict: "prefer" | "require" }
/// | { type: "grammar"; variants: GrammarVariants }
///
/// 使用 tag = "type" 让 JSON 用 "type" 字段区分两种形式，与 content.rs 里的枚举一致。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ConstrainedSamplingConfig {
    // 严格程度
    #[serde(rename = "json_schema")]
    JsonSchema {
        #[serde(rename = "strict")]
        strict: JsonSchemaStrict,
    },
    // 语法规则
    #[serde(rename = "grammar")]
    Grammar {
        variants: GrammarVariants,
    },
}

/// 一个可供模型调用的工具定义。

/// TParameters 是工具参数的格式；原版用 typebox 的 TSchema，
// 这里默认用任意 JSON 值 Value。
/// 调用方若想强约束，可以换成自己的 schema 类型。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool<TParameters = Value> {
    /// 工具名，例如 "read_file"。
    pub name: String,
    /// 给模型看的描述，说明这个工具是干什么的。
    pub description: String,
    /// 工具参数格式。
    pub parameters: TParameters,
    /// 受约束采样配置，默认不启用。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constrained_sampling: Option<ConstrainedSamplingConfig>,
}

/// 一次模型调用的完整上下文输入。
/// 原版：export interface Context
/// 它把一次对话所需的全部信息打包起来：
/// 系统提示 + 消息列表 + 可用工具。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    /// 系统提示，告诉模型它的身份和行为准则。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// 对话历史。原版是 Message[]，Rust 里是 Vec<ConversationMessage>。
    pub messages: Vec<ConversationMessage>,
    /// 本次可用的工具列表。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}
