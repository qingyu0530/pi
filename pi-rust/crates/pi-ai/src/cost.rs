//! 模型成本类型。
//!
//! message.rs 里的 Usage 记录“一次调用用了多少 token、花了多少钱”；
//! 而这里的成本类型描述的是“模型本身的价格表”：每百万 token 输入/输出各多少钱。
//! 原版：types.ts 第 803-818 行。

use serde::{Deserialize, Serialize};

/// 每百万 token 的单价。
/// 原版：export interface ModelCostRates
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostRates {
    /// 输入 token 单价（美元 / 百万 token）。
    pub input: f64,
    /// 输出 token 单价（美元 / 百万 token）。
    pub output: f64,
    /// 缓存读取单价。
    pub cache_read: f64,
    /// 缓存写入单价。
    pub cache_write: f64,
}

/// 价格档位：当输入 token 超过某个阈值时，按这一档计价。
/// 原版：export interface ModelCostTier extends ModelCostRates
///
/// C++ 对照：extends 相当于继承。Rust 没有继承，用组合 + #[serde(flatten)] 表达：
/// 把 rates 的字段“摊平”到本结构体顶层，序列化结果和直接继承一致。
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostTier {
    /// 继承自 ModelCostRates 的单价字段，flatten 让它们平铺在顶层。
    #[serde(flatten)]
    pub rates: ModelCostRates,
    /// 当整个请求的输入 token 数超过这个值时，启用这一档。
    pub input_tokens_above: u64,
}

/// 模型成本：基础单价 + 可选的分档价格。
/// 原版：export interface ModelCost extends ModelCostRates
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    /// 继承自 ModelCostRates 的单价字段。
    #[serde(flatten)]
    pub rates: ModelCostRates,
    /// 分档价格；最高的匹配阈值作用于整个请求。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tiers: Option<Vec<ModelCostTier>>,
}
