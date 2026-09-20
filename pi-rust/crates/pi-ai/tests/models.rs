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
fn builtin_catalog_parses_with_default_models() {
    let registry = ModelRegistry::builtin();

    assert!(registry.get("openai", "gpt-4o-mini").is_some());
    assert!(registry.get("deepseek", "deepseek-reasoner").is_some());
    assert!(registry.providers().contains(&"openai"));
    assert!(registry.providers().contains(&"deepseek"));

    // 内置数据里 deepseek-reasoner 带 thinkingLevelMap 和扁平 compat。
    let reasoner = registry.get("deepseek", "deepseek-reasoner").unwrap();
    assert!(reasoner.thinking_level_map.is_some());
    assert!(matches!(
        reasoner.compat,
        Some(ModelCompat::OpenaiCompletions(_))
    ));

    // 新增的 zai/glm-4.6。
    let glm = registry.get("zai", "glm-4.6").expect("glm-4.6 exists");
    assert_eq!(glm.max_tokens, 131_072);
}
