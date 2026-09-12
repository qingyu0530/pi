use pi_ai::{
    AssistantMessageEvent, Context, FauxProvider, FauxResponse, InputType, Model, ModelCost,
    ModelCostRates, Provider, RequestOptions, StopReason, UserMessage, UserMessageContent,
    UserRole,
};

fn faux_model() -> Model {
    Model {
        id: "faux-1".to_owned(),
        name: "Faux 1".to_owned(),
        api: "pi-messages".to_owned(),
        provider: "faux".to_owned(),
        base_url: "http://localhost".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![InputType::Text],
        cost: ModelCost {
            rates: ModelCostRates {
                input: 1.0,
                output: 2.0,
                cache_read: 0.5,
                cache_write: 1.0,
            },
            tiers: None,
        },
        context_window: 128_000,
        max_tokens: 16_384,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

#[test]
fn faux_provider_emits_start_text_and_done() {
    let provider = FauxProvider::new(vec![faux_model()]);
    let model = &provider.get_models()[0];
    let context = Context {
        system_prompt: None,
        messages: vec![
            UserMessage {
                role: UserRole::User,
                content: UserMessageContent::Text("你好，Pi".to_owned()),
                timestamp: 1,
            }
            .into(),
        ],
        tools: None,
    };

    let events: Vec<AssistantMessageEvent> = provider
        .stream(model, &context, &RequestOptions::default())
        .collect();

    // 第一件事必须是 start。
    assert!(matches!(events[0], AssistantMessageEvent::Start { .. }));
    // 中间一定有一段文本增量。
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AssistantMessageEvent::TextDelta { .. }))
    );
    // 最后以 done 收尾，携带完整助手消息。
    let done = events
        .iter()
        .find_map(|event| match event {
            AssistantMessageEvent::Done { message, .. } => Some(message),
            _ => None,
        })
        .expect("stream should end with done");
    assert_eq!(done.stop_reason, StopReason::Stop);
    assert!(!done.content.is_empty());
}

#[test]
fn faux_provider_emits_scripted_tool_call() {
    let arguments = serde_json::json!({ "text": "hi" })
        .as_object()
        .unwrap()
        .clone();
    let provider = FauxProvider::with_script(
        vec![faux_model()],
        vec![FauxResponse::tool_call("call_1", "echo", arguments)],
    );
    let model = &provider.get_models()[0];
    let context = Context {
        system_prompt: None,
        messages: vec![],
        tools: None,
    };

    let events: Vec<AssistantMessageEvent> = provider
        .stream(model, &context, &RequestOptions::default())
        .collect();

    assert!(matches!(events[0], AssistantMessageEvent::Start { .. }));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AssistantMessageEvent::ToolcallEnd { .. }))
    );
    let done = events
        .iter()
        .find_map(|event| match event {
            AssistantMessageEvent::Done { reason, message } => Some((reason, message)),
            _ => None,
        })
        .expect("stream should end with done");
    assert_eq!(*done.0, StopReason::ToolUse);
    assert!(matches!(
        done.1.content[0],
        pi_ai::AssistantContent::ToolCall(_)
    ));
}
