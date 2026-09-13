use pi_ai::ModelRegistry;

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

#[test]
fn builtin_catalog_parses_with_default_models() {
    let registry = ModelRegistry::builtin();

    assert!(registry.get("openai", "gpt-4o-mini").is_some());
    assert!(registry.get("deepseek", "deepseek-reasoner").is_some());
    assert!(registry.providers().contains(&"openai"));
    assert!(registry.providers().contains(&"deepseek"));
}
