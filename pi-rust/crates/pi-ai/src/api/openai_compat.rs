//! OpenAI-compatible 兼容性的「探测 + 解析」。
//!
//! `compat.rs` 里的 `OpenAICompletionsCompat` 是一堆 `Option` 开关（原始配置）。
//! 这一层把它们解析成一组确定的布尔/枚举（`ResolvedOpenAICompletionsCompat`）：
//! 1. 先按 provider / base_url 自动探测默认能力（`detect`）；
//! 2. 再用 `model.compat` 里显式设置的值覆盖（`resolve`）。
//!
//! 对应原版 openai-completions.ts 的 `detectCompat` / `getCompat`。

// 可以理解为 是在处理模型的怪癖  
// 
use std::collections::HashMap;

use crate::compat::{
    CacheControlFormat, ChatTemplateKwargValue, DeferredToolsMode, MaxTokensField,
    OpenAICompletionsCompat, OpenRouterRouting, SessionAffinityFormat, ThinkingFormat,
    ThinkingTokenBudgetField, VercelGatewayRouting,
};
use crate::model::{Model, ModelCompat};

/// 解析后的 OpenAI-compatible 兼容设置：字段都是确定值，不再有 `Option`。
///
/// 对应原版 `ResolvedOpenAICompletionsCompat`。
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedOpenAICompletionsCompat {
    /// 是否接受请求体里的 `store` 字段。
    pub supports_store: bool,
    /// 系统提示是否可以（且应该）用 `developer` 角色。
    pub supports_developer_role: bool,
    /// 是否支持 `reasoning_effort`。
    pub supports_reasoning_effort: bool,
    /// 流式响应是否返回 token 用量。
    pub supports_usage_in_streaming: bool,
    /// 流式响应是否在结尾给 `finish_reason`。
    pub supports_finish_reason: bool,
    /// 用哪个字段名表达最大输出 token。
    pub max_tokens_field: MaxTokensField,
    /// 工具结果消息是否必须带 `name`。
    pub requires_tool_result_name: bool,
    /// 工具结果后若紧跟用户消息，是否要插一条合成 assistant 消息。
    pub requires_assistant_after_tool_result: bool,
    /// 思考内容是否必须转成普通文本。
    pub requires_thinking_as_text: bool,
    /// 重放 assistant 消息时是否必须带空的 `reasoning_content`。
    pub requires_reasoning_content_on_assistant_messages: bool,
    /// 思考参数的格式。
    pub thinking_format: ThinkingFormat,
    /// chat-template 模式下的 kwargs。
    pub chat_template_kwargs: Option<HashMap<String, ChatTemplateKwargValue>>,
    /// baseten 模式下的 args。
    pub chat_template_args: Option<HashMap<String, ChatTemplateKwargValue>>,
    /// OpenRouter 供应商路由偏好。
    pub open_router_routing: Option<OpenRouterRouting>,
    /// Vercel AI Gateway 路由偏好。
    pub vercel_gateway_routing: Option<VercelGatewayRouting>,
    /// z.ai 工具流模式。
    pub zai_tool_stream: bool,
    /// 指定上限推理 token 的字段名。
    pub thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
    /// 是否支持推理 token 预算。
    pub supports_thinking_token_budget: bool,
    /// 是否支持 OpenAI grammar 工具。
    pub supports_openai_grammar_tools: bool,
    /// 是否支持 strict 工具参数。
    pub supports_strict_mode: bool,
    /// 缓存控制约定。
    pub cache_control_format: Option<CacheControlFormat>,
    /// 是否发送会话亲和请求头。
    pub send_session_affinity_headers: bool,
    /// 延迟工具序列化模式。
    pub deferred_tools_mode: Option<DeferredToolsMode>,
    /// 会话亲和请求头的格式。
    pub session_affinity_format: SessionAffinityFormat,
    /// 是否支持长时间缓存保留。
    pub supports_long_cache_retention: bool,
}

/// 按 provider / base_url 自动探测默认兼容能力。
///
/// 对应原版 `detectCompat`。
/// 不查配置，只看 provider 名字和 base_url，猜这家 Provider 的能力
#[must_use]
pub fn detect_openai_completions_compat(model: &Model) -> ResolvedOpenAICompletionsCompat {
    let provider = model.provider.as_str();
    let base_url = model.base_url.as_str();
    let base_url_lower = base_url.to_lowercase();
    // 识别每家 Provider     为什么同时看 provider 和 base_url：用户可能没规范填 provider，但 base_url 会暴露真实服务商。任一命中就认为对应。
    let is_zai = provider == "zai"
        || provider == "zai-coding-cn"
        || base_url.contains("api.z.ai")
        || base_url.contains("open.bigmodel.cn");
    let is_together = provider == "together"
        || base_url.contains("api.together.ai")
        || base_url.contains("api.together.xyz");
    let is_moonshot = provider == "moonshotai"
        || provider == "moonshotai-cn"
        || base_url.contains("api.moonshot.");
    let is_openrouter = provider == "openrouter" || base_url.contains("openrouter.ai");
    let is_cloudflare_workers_ai =
        provider == "cloudflare-workers-ai" || base_url.contains("api.cloudflare.com");
    let is_cloudflare_ai_gateway =
        provider == "cloudflare-ai-gateway" || base_url.contains("gateway.ai.cloudflare.com");
    let is_nvidia = provider == "nvidia" || base_url.contains("integrate.api.nvidia.com");
    let is_ant_ling = provider == "ant-ling" || base_url.contains("api.ant-ling.com");
    let is_deepseek = provider == "deepseek" || base_url_lower.contains("deepseek.com");
    // 这些服务商对标准 OpenAI 协议有改动，不能按原版处理。后面多个字段都用它取反。
    let is_non_standard = is_nvidia
        || provider == "cerebras"
        || base_url.contains("cerebras.ai")
        || provider == "xai"
        || base_url.contains("api.x.ai")
        || is_together
        || base_url.contains("chutes.ai")
        || is_deepseek
        || is_zai
        || is_moonshot
        || provider == "opencode"
        || base_url.contains("opencode.ai")
        || is_cloudflare_workers_ai
        || is_cloudflare_ai_gateway
        || is_ant_ling;

    let use_max_tokens = base_url.contains("chutes.ai")
        || is_deepseek
        || is_moonshot
        || is_cloudflare_ai_gateway
        || is_together
        || is_nvidia
        || is_ant_ling
        || is_zai;

    let is_grok = provider == "xai" || base_url.contains("api.x.ai");
    let is_openrouter_developer_role_model =
        is_openrouter && (model.id.starts_with("anthropic/") || model.id.starts_with("openai/"));
    let cache_control_format = (provider == "openrouter" && model.id.starts_with("anthropic/"))
        .then_some(CacheControlFormat::Anthropic);

    ResolvedOpenAICompletionsCompat {
        supports_store: !is_non_standard,
        supports_developer_role: is_openrouter_developer_role_model
            || (!is_non_standard && !is_openrouter),
        supports_reasoning_effort: !is_grok
            && !is_zai
            && !is_moonshot
            && !is_together
            && !is_cloudflare_ai_gateway
            && !is_nvidia
            && !is_ant_ling,
        supports_usage_in_streaming: true,
        supports_finish_reason: true,
        max_tokens_field: if use_max_tokens {
            MaxTokensField::MaxTokens
        } else {
            MaxTokensField::MaxCompletionTokens
        },
        requires_tool_result_name: false,
        requires_assistant_after_tool_result: false,
        requires_thinking_as_text: false,
        requires_reasoning_content_on_assistant_messages: is_deepseek,
        thinking_format: if is_deepseek {
            ThinkingFormat::Deepseek
        } else if is_zai {
            ThinkingFormat::Zai
        } else if is_together {
            ThinkingFormat::Together
        } else if is_ant_ling {
            ThinkingFormat::AntLing
        } else if is_openrouter {
            ThinkingFormat::Openrouter
        } else {
            ThinkingFormat::Openai
        },
        chat_template_kwargs: None,
        chat_template_args: None,
        open_router_routing: None,
        vercel_gateway_routing: None,
        zai_tool_stream: false,
        thinking_token_budget_field: None,
        supports_thinking_token_budget: false,
        supports_openai_grammar_tools: false,
        supports_strict_mode: !is_moonshot
            && !is_together
            && !is_cloudflare_ai_gateway
            && !is_nvidia,
        cache_control_format,
        send_session_affinity_headers: false,
        deferred_tools_mode: None,
        session_affinity_format: if is_openrouter {
            SessionAffinityFormat::Openrouter
        } else {
            SessionAffinityFormat::Openai
        },
        supports_long_cache_retention: !(is_together
            || is_cloudflare_workers_ai
            || is_cloudflare_ai_gateway
            || is_nvidia
            || is_ant_ling),
    }
}

/// 解析兼容设置：先探测，再用 `model.compat` 中显式的 OpenAI-completions 配置覆盖。
///
/// 对应原版 `getCompat`。若 `model.compat` 不是 `OpenaiCompletions` 变体，则只用探测结果。
#[must_use]
pub fn resolve_openai_completions_compat(model: &Model) -> ResolvedOpenAICompletionsCompat {
    let detected = detect_openai_completions_compat(model);
    let Some(ModelCompat::OpenaiCompletions(compat)) = &model.compat else {
        return detected;
    };
    let compat: &OpenAICompletionsCompat = compat;

    ResolvedOpenAICompletionsCompat {
        supports_store: compat.supports_store.unwrap_or(detected.supports_store),
        supports_developer_role: compat
            .supports_developer_role
            .unwrap_or(detected.supports_developer_role),
        supports_reasoning_effort: compat
            .supports_reasoning_effort
            .unwrap_or(detected.supports_reasoning_effort),
        supports_usage_in_streaming: compat
            .supports_usage_in_streaming
            .unwrap_or(detected.supports_usage_in_streaming),
        supports_finish_reason: compat
            .supports_finish_reason
            .unwrap_or(detected.supports_finish_reason),
        max_tokens_field: compat.max_tokens_field.unwrap_or(detected.max_tokens_field),
        requires_tool_result_name: compat
            .requires_tool_result_name
            .unwrap_or(detected.requires_tool_result_name),
        requires_assistant_after_tool_result: compat
            .requires_assistant_after_tool_result
            .unwrap_or(detected.requires_assistant_after_tool_result),
        requires_thinking_as_text: compat
            .requires_thinking_as_text
            .unwrap_or(detected.requires_thinking_as_text),
        requires_reasoning_content_on_assistant_messages: compat
            .requires_reasoning_content_on_assistant_messages
            .unwrap_or(detected.requires_reasoning_content_on_assistant_messages),
        thinking_format: compat.thinking_format.unwrap_or(detected.thinking_format),
        chat_template_kwargs: compat
            .chat_template_kwargs
            .clone()
            .or(detected.chat_template_kwargs),
        chat_template_args: compat
            .chat_template_args
            .clone()
            .or(detected.chat_template_args),
        open_router_routing: compat
            .open_router_routing
            .clone()
            .or(detected.open_router_routing),
        vercel_gateway_routing: compat
            .vercel_gateway_routing
            .clone()
            .or(detected.vercel_gateway_routing),
        zai_tool_stream: compat.zai_tool_stream.unwrap_or(detected.zai_tool_stream),
        thinking_token_budget_field: compat
            .thinking_token_budget_field
            .or(detected.thinking_token_budget_field),
        supports_thinking_token_budget: compat
            .supports_thinking_token_budget
            .unwrap_or(detected.supports_thinking_token_budget),
        supports_openai_grammar_tools: compat
            .supports_openai_grammar_tools
            .unwrap_or(detected.supports_openai_grammar_tools),
        supports_strict_mode: compat
            .supports_strict_mode
            .unwrap_or(detected.supports_strict_mode),
        cache_control_format: compat
            .cache_control_format
            .or(detected.cache_control_format),
        send_session_affinity_headers: compat
            .send_session_affinity_headers
            .unwrap_or(detected.send_session_affinity_headers),
        deferred_tools_mode: compat.deferred_tools_mode.or(detected.deferred_tools_mode),
        session_affinity_format: compat
            .session_affinity_format
            .unwrap_or(detected.session_affinity_format),
        supports_long_cache_retention: compat
            .supports_long_cache_retention
            .unwrap_or(detected.supports_long_cache_retention),
    }
}
