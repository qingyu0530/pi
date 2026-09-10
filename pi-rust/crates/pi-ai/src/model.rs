//! 模型目录类型。
//!
//! 前面的类型描述“一次调用”的数据；这里的 Model 描述“模型本身”的静态元数据：
//! 它叫什么、用哪个 API、价格多少、上下文窗口多大、支持哪些输入等。
//! 原版：types.ts 第 820-857 行。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::compat::{
    AnthropicMessagesCompat, BedrockCompat, OpenAICompletionsCompat, OpenAIResponsesCompat,
};
use crate::cost::ModelCost;
use crate::message::{Usage, UsageCost};

/// 思考级别。
/// 原版：export type ThinkingLevel = "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

/// 模型思考级别，比 ThinkingLevel 多一个 "off"（关闭思考）。
/// 原版：export type ModelThinkingLevel = "off" | ThinkingLevel;
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

/// 思考级别到模型具体参数的映射。
/// 原版：export type ThinkingLevelMap = Partial<Record<ModelThinkingLevel, string | null>>;
/// 用 HashMap 表达“键是思考级别，值是字符串或 null”。
/// 值为 None 表示该级别不受支持；键缺失表示使用 Provider 默认值。
pub type ThinkingLevelMap = HashMap<ModelThinkingLevel, Option<String>>;

/// 模型支持的输入内容类型。
/// 原版：input: ("text" | "image")[]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum InputType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "image")]
    Image,
}

/// 模型兼容配置。
///
/// 原版用条件类型：compat 的具体类型由 `api` 决定（types.ts 第 841-849 行）。
/// Rust 没有条件类型，用一个枚举表达“按 api 不同取四种配置之一”，
/// 调用方根据 api 选择对应的变体。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "api")]
pub enum ModelCompat {
    #[serde(rename = "openai-completions")]
    OpenaiCompletions(Box<OpenAICompletionsCompat>),
    #[serde(rename = "openai-responses")]
    OpenaiResponses(Box<OpenAIResponsesCompat>),
    #[serde(rename = "anthropic-messages")]
    AnthropicMessages(Box<AnthropicMessagesCompat>),
    #[serde(rename = "bedrock-converse-stream")]
    BedrockConverseStream(Box<BedrockCompat>),
}

/// 统一模型系统中的模型描述。
/// 原版：export interface Model<TApi extends Api>
/// 原版是泛型；这里简化成非泛型，api/provider 直接用 String（与 message.rs 一致）。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    /// 模型 id，例如 "gpt-5"。
    pub id: String,
    /// 展示用名称。
    pub name: String,
    /// 采用的 API 协议，例如 "openai-responses"。
    pub api: String,
    /// 模型服务商，例如 "openai"。
    pub provider: String,
    /// 请求地址。
    pub base_url: String,
    /// 是否支持推理（reasoning）。
    pub reasoning: bool,
    /// 思考级别映射；缺失键用 Provider 默认，null 表示不支持该级别。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    /// 支持的输入类型。
    pub input: Vec<InputType>,
    /// 模型价格。
    pub cost: ModelCost,
    /// 上下文窗口大小。
    pub context_window: u64,
    /// 最大输出 token 数。
    pub max_tokens: u64,
    /// 默认采样参数。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<HashMap<String, Value>>,
    /// 请求头。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// 兼容设置；按 api 选择对应类型。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<ModelCompat>,
}

/// 图片生成模型描述。
/// 原版：export interface ImagesModel<TApi extends ImagesApi>
/// 它省略了 Model 的 api/provider/reasoning/contextWindow/maxTokens/compat，
/// 改用图片 API 的 api/provider，并新增 output。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImagesModel {
    pub id: String,
    pub name: String,
    pub api: String,
    pub provider: String,
    pub base_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    pub input: Vec<InputType>,
    pub cost: ModelCost,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<HashMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// 生成的输出类型。
    pub output: Vec<InputType>,
}

/// 按模型价格表计算一次调用的费用。
///
/// 单价都以「每百万 token」计；`tiers` 表示输入超过阈值时整次请求按该档计价，
/// 取匹配到的最高那一档。1 小时缓存写按输入价 2 倍计（Anthropic 规则）。
#[must_use]
pub fn calculate_cost(model: &Model, usage: &Usage) -> UsageCost {
    let input_tokens = usage.input + usage.cache_read + usage.cache_write; // 都加起来先 决定用哪一档价
    let mut rates = model.cost.rates;
    let mut matched_threshold: i64 = -1; // 记录目前匹配到的最高阈值。初始 -1 是一个「比任何合法阈值都小」的哨兵值。
    for tier in model.cost.tiers.iter().flatten() {
        // 遍历所有分档
        if input_tokens > tier.input_tokens_above // 本次输入真的超过这档阈值。
            && (tier.input_tokens_above as i64) > matched_threshold
        // 这档阈值比已记录的更高。
        {
            rates = tier.rates;
            matched_threshold = tier.input_tokens_above as i64;
        }
    }
    //  1 小时缓存写的 token 数
    let long_write = usage.cache_write_1h.unwrap_or(0);
    // 其余（短时）缓存写
    // 为什么拆开？ 1 小时缓存写的费用是输入价的 2 倍，而普通缓存写用 cache_write 单价。
    let short_write = usage.cache_write.saturating_sub(long_write);
    let input = rates.input / 1_000_000.0 * usage.input as f64;
    let output = rates.output / 1_000_000.0 * usage.output as f64;
    let cache_read = rates.cache_read / 1_000_000.0 * usage.cache_read as f64;
    let cache_write = (rates.cache_write * short_write as f64
        + rates.input * 2.0 * long_write as f64)
        / 1_000_000.0;

    UsageCost {
        input,
        output,
        cache_read,
        cache_write,
        total: input + output + cache_read + cache_write,
    }
}
