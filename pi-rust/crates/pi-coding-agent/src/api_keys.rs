//! Provider 名 -> API Key 环境变量名的映射。
//!
//! 移植自原版 `packages/ai/src/env-api-keys.ts` 的 envMap。
//! 原版还支持 OAuth（ANTHROPIC_AUTH_TOKEN / sk-ant-oat）、AWS、gcloud
//! 等环境凭证；这里只保留「一个环境变量装一个 key」的简单形态，
//! 其余认证方式等接入了对应 Provider 再补。
//!
//! C++ 对照：一张静态查找表，等价于
//! `static const std::unordered_map<std::string, std::string>`。

/// 返回 provider 对应的 API Key 环境变量名；没有已知映射时返回 `None`。
///
/// 例如 `deepseek` -> `DEEPSEEK_API_KEY`、`zai` -> `ZAI_API_KEY`。
#[must_use]
pub fn api_key_env(provider: &str) -> Option<&'static str> {
    match provider {
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        "ant-ling" => Some("ANT_LING_API_KEY"),
        "qwen-token-plan" | "qwen-token-plan-individual" => Some("QWEN_TOKEN_PLAN_API_KEY"),
        "qwen-token-plan-cn" => Some("QWEN_TOKEN_PLAN_CN_API_KEY"),
        "openai" => Some("OPENAI_API_KEY"),
        "azure-openai-responses" => Some("AZURE_OPENAI_API_KEY"),
        "nvidia" => Some("NVIDIA_API_KEY"),
        "deepseek" => Some("DEEPSEEK_API_KEY"),
        "google" => Some("GEMINI_API_KEY"),
        "google-vertex" => Some("GOOGLE_CLOUD_API_KEY"),
        "groq" => Some("GROQ_API_KEY"),
        "cerebras" => Some("CEREBRAS_API_KEY"),
        "xai" => Some("XAI_API_KEY"),
        "radius" => Some("RADIUS_API_KEY"),
        "openrouter" => Some("OPENROUTER_API_KEY"),
        "vercel-ai-gateway" => Some("AI_GATEWAY_API_KEY"),
        "zai" => Some("ZAI_API_KEY"),
        "zai-coding-cn" => Some("ZAI_CODING_CN_API_KEY"),
        "mistral" => Some("MISTRAL_API_KEY"),
        "minimax" => Some("MINIMAX_API_KEY"),
        "minimax-cn" => Some("MINIMAX_CN_API_KEY"),
        "moonshotai" | "moonshotai-cn" => Some("MOONSHOT_API_KEY"),
        "huggingface" => Some("HF_TOKEN"),
        "fireworks" => Some("FIREWORKS_API_KEY"),
        "together" => Some("TOGETHER_API_KEY"),
        "baseten" => Some("BASETEN_API_KEY"),
        "opencode" | "opencode-go" => Some("OPENCODE_API_KEY"),
        "kimi-coding" => Some("KIMI_API_KEY"),
        "cloudflare-workers-ai" | "cloudflare-ai-gateway" => Some("CLOUDFLARE_API_KEY"),
        "xiaomi" => Some("XIAOMI_API_KEY"),
        "xiaomi-token-plan-cn" => Some("XIAOMI_TOKEN_PLAN_CN_API_KEY"),
        "xiaomi-token-plan-ams" => Some("XIAOMI_TOKEN_PLAN_AMS_API_KEY"),
        "xiaomi-token-plan-sgp" => Some("XIAOMI_TOKEN_PLAN_SGP_API_KEY"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::api_key_env;

    #[test]
    fn known_providers_map_to_their_env_vars() {
        assert_eq!(api_key_env("openai"), Some("OPENAI_API_KEY"));
        assert_eq!(api_key_env("anthropic"), Some("ANTHROPIC_API_KEY"));
        assert_eq!(api_key_env("deepseek"), Some("DEEPSEEK_API_KEY"));
        assert_eq!(api_key_env("zai"), Some("ZAI_API_KEY"));
        assert_eq!(api_key_env("moonshotai-cn"), Some("MOONSHOT_API_KEY"));
        assert_eq!(api_key_env("opencode-go"), Some("OPENCODE_API_KEY"));
    }

    #[test]
    fn unknown_providers_have_no_mapping() {
        assert_eq!(api_key_env("my-custom-gateway"), None);
        assert_eq!(api_key_env(""), None);
    }
}
