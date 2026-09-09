use pi_ai::{
    ChatTemplateKwargValue, ChatTemplateVar, Model, ModelCompat, ModelCost, ModelCostRates,
    ModelCostTier, OpenAICompletionsCompat, ThinkingFormat, ThinkingTokenBudgetField,
};
use serde_json::json;

#[test]
fn model_cost_tier_flattens_inherited_rates() {
    let cost = ModelCost {
        rates: ModelCostRates {
            input: 1.0,
            output: 2.0,
            cache_read: 0.5,
            cache_write: 1.5,
        },
        tiers: Some(vec![ModelCostTier {
            rates: ModelCostRates {
                input: 0.8,
                output: 1.6,
                cache_read: 0.4,
                cache_write: 1.2,
            },
            input_tokens_above: 100_000,
        }]),
    };

    let expected = json!({
        "input": 1.0,
        "output": 2.0,
        "cacheRead": 0.5,
        "cacheWrite": 1.5,
        "tiers": [
            {
                "input": 0.8,
                "output": 1.6,
                "cacheRead": 0.4,
                "cacheWrite": 1.2,
                "inputTokensAbove": 100000
            }
        ]
    });

    assert_eq!(serde_json::to_value(&cost).unwrap(), expected);
}

#[test]
fn chat_template_kwarg_value_supports_scalars_and_vars() {
    let var = ChatTemplateVar {
        var: "thinking.enabled".to_owned(),
        omit_when_off: Some(true),
    };
    let values = [
        ChatTemplateKwargValue::String("你好".to_owned()),
        ChatTemplateKwargValue::Number(3.0),
        ChatTemplateKwargValue::Bool(true),
        ChatTemplateKwargValue::Null,
        ChatTemplateKwargValue::Var(var),
    ];

    let serialized: Vec<serde_json::Value> = values
        .iter()
        .map(|v| serde_json::to_value(v).unwrap())
        .collect();

    assert_eq!(
        serialized[0],
        json!("你好"),
        "字符串应直接输出，不包裹变体名"
    );
    assert_eq!(serialized[3], serde_json::Value::Null);
    assert_eq!(
        serialized[4],
        json!({ "$var": "thinking.enabled", "omitWhenOff": true })
    );
}

#[test]
fn model_with_completions_compat_round_trips() {
    let model = Model {
        id: "gpt-4o".to_owned(),
        name: "GPT-4o".to_owned(),
        api: "openai-completions".to_owned(),
        provider: "openai".to_owned(),
        base_url: "https://api.openai.com/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![pi_ai::InputType::Text],
        cost: ModelCost {
            rates: ModelCostRates {
                input: 2.5,
                output: 10.0,
                cache_read: 1.25,
                cache_write: 5.0,
            },
            tiers: None,
        },
        context_window: 128_000,
        max_tokens: 16_384,
        sampling_params: None,
        headers: None,
        compat: Some(ModelCompat::OpenaiCompletions(Box::new(
            OpenAICompletionsCompat {
                supports_store: Some(true),
                supports_developer_role: Some(false),
                supports_reasoning_effort: Some(true),
                supports_usage_in_streaming: None,
                supports_finish_reason: None,
                max_tokens_field: Some(pi_ai::MaxTokensField::MaxCompletionTokens),
                requires_tool_result_name: None,
                requires_assistant_after_tool_result: None,
                requires_thinking_as_text: None,
                requires_reasoning_content_on_assistant_messages: None,
                thinking_format: Some(ThinkingFormat::Openai),
                chat_template_kwargs: None,
                chat_template_args: None,
                open_router_routing: None,
                vercel_gateway_routing: None,
                zai_tool_stream: None,
                thinking_token_budget_field: Some(ThinkingTokenBudgetField::ThinkingTokenBudget),
                supports_thinking_token_budget: None,
                supports_openai_grammar_tools: None,
                supports_strict_mode: None,
                cache_control_format: None,
                send_session_affinity_headers: None,
                deferred_tools_mode: None,
                session_affinity_format: None,
                supports_long_cache_retention: None,
            },
        ))),
    };

    let expected = json!({
        "id": "gpt-4o",
        "name": "GPT-4o",
        "api": "openai-completions",
        "provider": "openai",
        "baseUrl": "https://api.openai.com/v1",
        "reasoning": false,
        "input": [ "text" ],
        "cost": {
            "input": 2.5, "output": 10.0, "cacheRead": 1.25, "cacheWrite": 5.0
        },
        "contextWindow": 128000,
        "maxTokens": 16384,
        "compat": {
            "api": "openai-completions",
            "supportsStore": true,
            "supportsDeveloperRole": false,
            "supportsReasoningEffort": true,
            "maxTokensField": "max_completion_tokens",
            "thinkingFormat": "openai",
            "thinkingTokenBudgetField": "thinking_token_budget"
        }
    });

    assert_eq!(serde_json::to_value(&model).unwrap(), expected);
    assert_eq!(serde_json::from_value::<Model>(expected).unwrap(), model);
}
