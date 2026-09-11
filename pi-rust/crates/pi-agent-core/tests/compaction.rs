use pi_agent_core::{
    CompactionSettings, build_summary_prompt, compact, estimate_context_tokens, estimate_tokens,
    plan_compaction, should_compact,
};
use pi_ai::{
    AssistantContent, AssistantMessage, AssistantRole, ConversationMessage, StopReason,
    TextContent, Usage, UsageCost, UserMessage, UserMessageContent, UserRole,
};

fn user(text: &str) -> ConversationMessage {
    UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Text(text.to_owned()),
        timestamp: 1,
    }
    .into()
}

fn assistant_text(text: &str, total_tokens: u64) -> ConversationMessage {
    AssistantMessage {
        role: AssistantRole::Assistant,
        content: vec![AssistantContent::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: "openai-completions".to_owned(),
        provider: "openai".to_owned(),
        model: "gpt-4o-mini".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens,
            cost: UsageCost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        },
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 2,
    }
    .into()
}

fn settings(reserve: usize, keep_recent: usize) -> CompactionSettings {
    CompactionSettings {
        enabled: true,
        reserve_tokens: reserve,
        keep_recent_tokens: keep_recent,
    }
}

#[test]
fn estimate_tokens_grows_with_text() {
    let short = estimate_tokens(&user("hi"));
    let long = estimate_tokens(&user(&"a".repeat(400)));

    assert!(long > short);
    // 400 个字符 / 每 token 4 字符 = 100。
    assert_eq!(long, 100);
}

#[test]
fn should_compact_respects_threshold_and_enabled_flag() {
    let active = settings(1_000, 100);
    assert!(!should_compact(1_000, 2_000, active));
    assert!(should_compact(1_001, 2_000, active));

    let disabled = CompactionSettings {
        enabled: false,
        ..active
    };
    assert!(!should_compact(99_999, 2_000, disabled));
}

#[test]
fn plan_keeps_only_the_recent_tail() {
    // 每条短消息约 1 token；保留 2 token 即保留最后两条。
    let messages = vec![user("a"), user("b"), user("c"), user("d")];
    let plan = plan_compaction(&messages, settings(100, 2));

    assert_eq!(plan.to_summarize.len(), 2);
    assert_eq!(plan.retained_tail.len(), 2);
    assert_eq!(plan.retained_tail.last().unwrap(), messages.last().unwrap());
}

#[test]
fn plan_keeps_at_least_one_message() {
    let messages = vec![user(&"a".repeat(400))];
    let plan = plan_compaction(&messages, settings(100, 0));

    assert_eq!(plan.to_summarize.len(), 0);
    assert_eq!(plan.retained_tail.len(), 1);
}

#[test]
fn compact_uses_summarizer_and_retains_tail() {
    let messages = vec![
        user("old request"),
        assistant_text("old reply", 0),
        user("recent request"),
    ];
    let result = compact(&messages, settings(10, 1), |system, prompt| {
        assert!(system.contains("summarization"));
        assert!(prompt.contains("old request"));
        Ok("SUMMARY".to_owned())
    })
    .unwrap();

    assert_eq!(result.summary, "SUMMARY");
    assert_eq!(
        result.retained_tail.last().unwrap(),
        messages.last().unwrap()
    );
    assert!(result.tokens_before > 0);
}

#[test]
fn estimate_context_tokens_uses_last_assistant_usage() {
    let messages = vec![
        user("hi"),
        assistant_text("ok", 500),
        user("more text after usage"),
    ];

    let tokens = estimate_context_tokens(&messages);

    // 至少包含服务端报告的 500。
    assert!(tokens >= 500);
}

#[test]
fn summary_prompt_contains_rendered_messages_and_instructions() {
    let prompt = build_summary_prompt(&[user("帮我重构"), assistant_text("好的", 0)]);

    assert!(prompt.contains("user: 帮我重构"));
    assert!(prompt.contains("assistant: 好的"));
    assert!(prompt.contains("## Goal"));
}
