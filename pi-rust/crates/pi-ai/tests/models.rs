use pi_ai::{MaxTokensField, ModelCompat, ModelRegistry, ModelThinkingLevel, ThinkingFormat};

const SAMPLE: &str = r#"{
  "openai-completions": {
    "m1": { "id": "m1", "name": "M1", "api": "openai-completions", "provider": "p1", "baseUrl": "https://x", "reasoning": false, "input": ["text"], "cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 }, "contextWindow": 1000, "maxTokens": 100 },
    "m2": { "id": "m2", "name": "M2", "api": "openai-completions", "provider": "p2", "baseUrl": "https://y", "reasoning": true, "input": ["text"], "cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 }, "contextWindow": 2000, "maxTokens": 200 }
  }
}"#;

#[test]
fn registry_loads_and_looks_up_models() {
    let registry = ModelRegistry::from_json(SAMPLE).unwrap();

    assert_eq!(registry.models().len(), 2);

    let model = registry.get("p1", "m1").expect("m1 exists");
    assert_eq!(model.id, "m1");
    assert_eq!(model.base_url, "https://x");
    assert!(!model.reasoning);

    assert!(registry.get("p1", "missing").is_none());
    assert_eq!(registry.providers(), vec!["p1", "p2"]);
    assert_eq!(registry.models_for_provider("p2").len(), 1);
}

/// 原版数据形状：`compat` 是扁平对象（不含 `api`），并带 `thinkingLevelMap`。
const ORIGINAL_SHAPED: &str = r#"{
  "openai-completions": {
    "glm-4.6": {
      "id": "glm-4.6",
      "name": "GLM-4.6",
      "api": "openai-completions",
      "provider": "zai",
      "baseUrl": "https://api.z.ai/api/coding/paas/v4",
      "reasoning": true,
      "thinkingLevelMap": { "off": null, "minimal": "minimal", "high": "high" },
      "input": ["text"],
      "cost": { "input": 0.6, "output": 2.2, "cacheRead": 0.11, "cacheWrite": 0 },
      "contextWindow": 200000,
      "maxTokens": 131072,
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "thinkingFormat": "zai",
        "maxTokensField": "max_tokens"
      }
    }
  }
}"#;

#[test]
fn parses_flat_compat_and_thinking_level_map() {
    let registry = ModelRegistry::from_json(ORIGINAL_SHAPED).unwrap();
    let model = registry.get("zai", "glm-4.6").expect("glm-4.6 exists");

    assert!(model.reasoning);
    let map = model.thinking_level_map.as_ref().expect("thinkingLevelMap");
    // null 表示该级别不受支持。
    assert_eq!(map.get(&ModelThinkingLevel::Off), Some(&None));
    assert_eq!(
        map.get(&ModelThinkingLevel::High),
        Some(&Some("high".to_owned()))
    );

    match model.compat.as_ref().expect("compat") {
        ModelCompat::OpenaiCompletions(compat) => {
            assert_eq!(compat.supports_store, Some(false));
            assert_eq!(compat.supports_developer_role, Some(false));
            assert_eq!(compat.thinking_format, Some(ThinkingFormat::Zai));
            assert_eq!(compat.max_tokens_field, Some(MaxTokensField::MaxTokens));
        }
        other => panic!("expected openai-completions compat, got {other:?}"),
    }
}

#[test]
fn from_json_many_merges_files() {
    const A: &str = r#"{ "openai-completions": { "a": { "id": "a", "name": "A", "api": "openai-completions", "provider": "p1", "baseUrl": "https://x", "reasoning": false, "input": ["text"], "cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 }, "contextWindow": 1, "maxTokens": 1 } } }"#;
    const B: &str = r#"{ "anthropic-messages": { "b": { "id": "b", "name": "B", "api": "anthropic-messages", "provider": "p2", "baseUrl": "https://y", "reasoning": true, "input": ["text"], "cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 }, "contextWindow": 1, "maxTokens": 1 } } }"#;

    let registry = ModelRegistry::from_json_many(&[("a.json", A), ("b.json", B)]).unwrap();

    assert_eq!(registry.models().len(), 2);
    assert_eq!(registry.providers(), vec!["p1", "p2"]);
}

#[test]
fn builtin_catalog_parses_full_provider_data() {
    let registry = ModelRegistry::builtin();

    // 原版完整目录：几十个 provider、上千个模型。
    assert!(registry.models().len() > 1_000);
    for provider in ["openai", "anthropic", "deepseek", "openrouter", "zai"] {
        assert!(registry.providers().contains(&provider));
    }

    // openai 的模型走 openai-responses。
    let gpt = registry
        .get("openai", "gpt-4o-mini")
        .expect("gpt-4o-mini exists");
    assert_eq!(gpt.api, "openai-responses");

    // anthropic 走 anthropic-messages，带扁平 compat 和 thinkingLevelMap。
    let sonnet = registry
        .get("anthropic", "claude-sonnet-5")
        .expect("claude-sonnet-5 exists");
    assert_eq!(sonnet.api, "anthropic-messages");
    assert!(sonnet.reasoning);
    assert!(matches!(
        sonnet.compat,
        Some(ModelCompat::AnthropicMessages(_))
    ));
    assert!(sonnet.thinking_level_map.is_some());

    // openai-completions 的 provider（deepseek）。
    let deepseek = registry
        .get("deepseek", "deepseek-v4-flash")
        .expect("deepseek-v4-flash exists");
    assert_eq!(deepseek.api, "openai-completions");
    assert!(matches!(
        deepseek.compat,
        Some(ModelCompat::OpenaiCompletions(_))
    ));

    // 未实现的 api（例如 google-generative-ai）也会被加载，
    // 其 compat 被忽略而不是解析失败。
    let google = registry
        .models()
        .iter()
        .find(|model| model.api == "google-generative-ai");
    assert!(google.is_some());
}
