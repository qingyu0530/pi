//! Provider 兼容配置与路由偏好类型。
//!
//! 不同 Provider 对同一 API 协议的字段支持程度不同（例如有的支持 `store`、
//! 有的角色叫 `developer` 而不是 `system`）。这些配置用于告诉 pi 如何适配某家 Provider。
//! 原版：types.ts 第 86-97、110-116、307-311、557-801 行。

use serde::{Deserialize, Serialize};

use crate::cost::ModelCost;

/// 用于 `chat_template_kwargs` / `chat_template_args` 的单个值。
/// 原版：export type ChatTemplateKwargValue =
///   string | number | boolean | null | { $var: "thinking.enabled" | ...; omitWhenOff?: boolean }
///
/// 用 untagged 枚举表达“多种可能形态”：
/// 值可以是普通 JSON 标量，也可以是一个带 `$var` 的变量引用对象。
/// 聊天模板参数（kwarg）的取值
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatTemplateKwargValue {
    String(String),
    Number(f64),
    Bool(bool),
    Null,
    Var(ChatTemplateVar),
}

/// 引用 pi 控制的思考值的变量。
/// 原版：{ $var: "thinking.enabled" | "thinking.effort" | "thinking.budget"; omitWhenOff?: boolean }
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatTemplateVar {
    /// 引用的变量名，例如 "thinking.enabled"。
    /// $var 不是合法 Rust 标识符，所以用 rename 桥接到 JSON 的 "$var"。
    #[serde(rename = "$var")]
    pub var: String,
    /// 当变量关闭时是否省略该键。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub omit_when_off: Option<bool>,
}

/// 用于上限推理 token 的请求字段名。
/// 思考 token 上限的请求字段名
///
/// 原版：export type ThinkingTokenBudgetField = "thinking_token_budget" | "thinking_budget" | "thinking_budget_tokens";
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ThinkingTokenBudgetField {
    // 三个变体对应三个字符串值，
    // 用于告诉 pi 用哪个字段来限制推理 token 预算。
    #[serde(rename = "thinking_token_budget")]
    ThinkingTokenBudget,
    #[serde(rename = "thinking_budget")]
    ThinkingBudget,
    #[serde(rename = "thinking_budget_tokens")]
    ThinkingBudgetTokens,
}

/// 会话亲和（session affinity）请求头的格式。
/*
什么是会话亲和（session affinity）
LLM 请求通常走负载均衡，一次对话会被拆成多次请求。
会话亲和就是让同一会话的多次请求路由到同一台后端副本。
为什么重要？
因为提示词缓存（prompt caching）是按副本存的——如果请求每次都打到不同副本，
缓存永远命中不了，既慢又贵。

客户端通过发送特殊请求头，让网关按会话把这些请求粘到同一副本。这就是「亲和」。

不同 Provider 用的请求头格式不一样，pi 必须知道用哪种
最大化缓存命中。它只是个格式选择器，具体头名由变体决定。

*/
// 决定会话亲和头怎么拼
/// 原版：export type SessionAffinityFormat = "openai" | "openai-nosession" | "openrouter";
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SessionAffinityFormat {
    #[serde(rename = "openai")]
    Openai,
    #[serde(rename = "openai-nosession")]
    OpenaiNosession,
    #[serde(rename = "openrouter")]
    Openrouter,
}

/// Anthropic 允许作为服务端拒绝回退目标的模型。
/// 原版：export interface AnthropicAllowedFallbackModel
/*
先懂什么是「服务端拒绝回退」

这是 Anthropic 的一个特性。当主模型因为安全过滤等原因拒绝作答时，
Anthropic 的服务器不会直接返回拒绝，
而是自动改用你指定的「回退模型」重新作答。

流程：

1.你发请求，带 fallbacks: ["claude-haiku"]。
2.主模型（如 claude-opus）被安全策略拒绝。
3.Anthropic 服务器端自动拿 claude-haiku 重试，返回它的回答。
也就是说，回退发生在服务器内部，不是客户端重新请求。

Anthropic 不是随便哪个模型都能当回退目标。
它只允许特定模型作为 fallbacks，你指定不允许的模型，Anthropic 会拒绝整个请求。

所以 allowedFallbackModels 就是一份白名单：
告诉 pi「哪些模型被允许当这个主模型的后备，它们的价格是多少」


*/
/*
「主模型被安全拒绝时，Anthropic 服务器端可自动切换到的备用模型清单 + 它的价格」。
它让 pi 既能发起回退，又能按实际作答的模型正确算费。

*/
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnthropicAllowedFallbackModel {
    /// 模型服务商。  回退模型的服务商（通常是 "anthropic"）
    pub provider: String,
    /// 模型id 。 如 "claude-3-5-haiku
    pub model: String,
    /// 该回退模型的价格
    pub cost: ModelCost,
}

/// OpenAI 兼容补全 API 的兼容设置。
/// 原版：export interface OpenAICompletionsCompat
/// 这些字段大多是可选的布尔开关，用于覆盖按 URL 自动探测的结果。
/*

很多第三方服务商（如 OpenRouter、Together、DeepSeek、z.ai、vLLM 本地部署）都
宣称「兼容 OpenAI 补全 API（/chat/completions）」，但每个都对协议有增删改：

有的不认 store 字段；
有的角色叫 developer 不叫 system；
有的不支持 reasoning_effort；
有的流式响应不给 finish_reason。
OpenAICompletionsCompat 就是一张**「能力/怪癖清单」**：
告诉 pi「对这家，哪些字段能用、哪些不能用、哪些要换个字段名」。


*/
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAICompletionsCompat {
    // 这家认不认请求体里的 store 字段
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_store: Option<bool>,
    // 角色该用 developer 还是 system
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    // 认不认 reasoning_effort
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_reasoning_effort: Option<bool>,
    // 流式时能不能返回 token 用量
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_usage_in_streaming: Option<bool>,
    // 流式结尾给不给 finish_reason？不给就由 pi 推断。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_finish_reason: Option<bool>,
    /// 用于 max token 的字段名。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens_field: Option<MaxTokensField>,
    // 工具结果必须带 name 字段吗
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_tool_result_name: Option<bool>,
    // 工具结果后必须插一条 assistant 消息
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_assistant_after_tool_result: Option<bool>,
    // 思考块必须转成 <thinking> 文本吗
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_thinking_as_text: Option<bool>,
    // 重放 assistant 消息必须带空的 reasoning_content 吗
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_reasoning_content_on_assistant_messages: Option<bool>,
    /// 思考参数的格式。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_format: Option<ThinkingFormat>,
    /// chat-template 模式下的 kwargs。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<std::collections::HashMap<String, ChatTemplateKwargValue>>,
    /// baseten 模式下的 args。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_args: Option<std::collections::HashMap<String, ChatTemplateKwargValue>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub open_router_routing: Option<OpenRouterRouting>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vercel_gateway_routing: Option<VercelGatewayRouting>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zai_tool_stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_thinking_token_budget: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_openai_grammar_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    /// 缓存控制约定。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control_format: Option<CacheControlFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_session_affinity_headers: Option<bool>,
    /// 延迟工具序列化模式。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deferred_tools_mode: Option<DeferredToolsMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_affinity_format: Option<SessionAffinityFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
}

// 发送『最大输出 token 数』时用哪个请求字段名
/// 用于 max token 的字段名。
/// 原版：maxTokensField?: "max_completion_tokens" | "max_tokens";
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MaxTokensField {
    // 新版 OpenAI（o 系列、GPT-5）
    #[serde(rename = "max_completion_tokens")]
    MaxCompletionTokens,
    // 老版 OpenAI、大部分兼容服务商
    #[serde(rename = "max_tokens")]
    MaxTokens,
}

/// 思考参数的格式。
/// 原版：thinkingFormat?: "openai" | "openrouter" | ... | "ant-ling";
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ThinkingFormat {
    #[serde(rename = "openai")]
    Openai,
    #[serde(rename = "openrouter")]
    Openrouter,
    #[serde(rename = "deepseek")]
    Deepseek,
    #[serde(rename = "together")]
    Together,
    #[serde(rename = "baseten")]
    Baseten,
    #[serde(rename = "zai")]
    Zai,
    #[serde(rename = "qwen")]
    Qwen,
    #[serde(rename = "chat-template")]
    ChatTemplate,
    #[serde(rename = "qwen-chat-template")]
    QwenChatTemplate,
    #[serde(rename = "string-thinking")]
    StringThinking,
    #[serde(rename = "ant-ling")]
    AntLing,
}

/// 缓存控制约定。
/// 原版：cacheControlFormat?: "anthropic";
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CacheControlFormat {
    #[serde(rename = "anthropic")]
    Anthropic,
}

/// 延迟工具序列化模式。
/// 原版：deferredToolsMode?: "kimi";
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DeferredToolsMode {
    #[serde(rename = "kimi")]
    Kimi,
}

/// OpenAI Responses API 的兼容设置。
/// 原版：export interface OpenAIResponsesCompat
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAIResponsesCompat {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_affinity_format: Option<SessionAffinityFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_openai_grammar_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_additional_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_tool_search: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_explicit_prompt_cache_mode: Option<bool>,
}

/// Anthropic Messages API 的兼容设置。
/// 原版：export interface AnthropicMessagesCompat
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicMessagesCompat {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_eager_tool_input_streaming: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_session_affinity_headers: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_cache_control_on_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_temperature: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub force_adaptive_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_empty_signature: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_strict_tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_fallback_models: Option<Vec<AnthropicAllowedFallbackModel>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_tool_references: Option<bool>,
}

/// Amazon Bedrock 的兼容设置。
/// 原版：export interface BedrockCompat
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockCompat {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
}

/// OpenRouter 供应商路由偏好。
/// 原版：export interface OpenRouterRouting
/// 这些字段名保持 OpenRouter API 要求的下划线命名，所以这里没有 rename_all。
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterRouting {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    /// 数据收集设置。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<DataCollection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enforce_distillable_text: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<String>>,
    /// 排序策略：字符串或带 by/partition 的对象。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<Sort>,
    /// 每百万 token 的最高价格。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_price: Option<MaxPrice>,
    /// 首选的最小吞吐量。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_min_throughput: Option<Throughput>,
    /// 首选的最大延迟。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_max_latency: Option<Latency>,
}

/// 数据收集设置。
/// 原版：data_collection?: "deny" | "allow";
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DataCollection {
    #[serde(rename = "deny")]
    Deny,
    #[serde(rename = "allow")]
    Allow,
}

/// 排序策略：字符串，或带 by/partition 的对象。
/// 原版：sort?: string | { by?: string; partition?: string | null };
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Sort {
    Name(String),
    Spec {
        by: Option<String>,
        partition: Option<String>,
    },
}

/// 每百万 token 的最高价格。
/// 原版：max_price?: { prompt?, completion?, image?, audio?, request? }
/*
这是 OpenRouterRouting.max_price 字段。OpenRouter 会把一个模型请求路由到多家上游供应商。
你设了价格上限后，OpenRouter 只路由到价格低于该上限的供应商，用来控制成本。
*/
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaxPrice {
    /// 价格可以是数字或字符串（如 "0.01"）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PriceValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion: Option<PriceValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<PriceValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<PriceValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<PriceValue>,
}

/// 价格值：数字或字符串。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PriceValue {
    Number(f64),
    Text(String),
}

// OpenRouter 路由时，希望选中的供应商生成速度不低于某个值
// 吞吐量 = 模型每秒能生成多少 token（tokens/second）。OpenRouter 会给各供应商记录这个指标。
// 你设了「最小吞吐量」后，OpenRouter 会优先选择吞吐量达标的供应商，而不是只看价格。
/// 首选的最小吞吐量：数字，或按分位数设置的对象。
/// 原版：preferred_min_throughput?: number | { p50?, p75?, p90?, p99? }
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Throughput {
    Number(f64),
    Spec(Percentiles),
}
// OpenRouter 选供应商时，希望响应速度不快于某个上限*
// 延迟不能太大）
/// 首选的最大延迟：数字，或按分位数设置的对象。
/// 原版：preferred_max_latency?: number | { p50?, p75?, p90?, p99? }
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Latency {
    Number(f64),
    Spec(Percentiles),
}

/// 分位数阈值集合。
/*

用 p50/p75/p90/p99 四个百分位描述「性能分布」：p50 看一般情况，p90 看常见较差，
p99 看最坏情况。在 Throughput（吞吐量下限）
和 Latency（延迟上限）里，它让「最小/最大」的约束能按不同百分位分别设定。

*/
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Percentiles {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p50: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p75: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p90: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p99: Option<f64>,
}

/// Vercel AI Gateway 路由偏好。
/// 原版：export interface VercelGatewayRouting
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VercelGatewayRouting {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
}
