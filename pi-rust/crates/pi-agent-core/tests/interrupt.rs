//! 中断（Ctrl+C）在 Agent 工具循环里的行为测试。
//!
//! 标志是进程级全局静态；所有场景放在一个测试函数里顺序执行，
//! 避免并行测试互相看到别人的标志。

use pi_agent_core::{Agent, AgentTool, ToolError, ToolResult};
use pi_ai::{
    AssistantContent, AssistantMessage, AssistantRole, FauxProvider, FauxResponse, InputType,
    Model, ModelCost, ModelCostRates, StopReason, ToolCall, ToolResultContent, Usage, UsageCost,
    UserMessage, UserMessageContent, UserRole,
};

fn zero_usage() -> Usage {
    Usage {
        input: 0,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 0,
        cost: UsageCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
    }
}
use serde_json::{Map, Value, json};

fn model() -> Model {
    Model {
        id: "faux-1".to_owned(),
        name: "Faux 1".to_owned(),
        api: "openai-completions".to_owned(),
        provider: "faux".to_owned(),
        base_url: "http://localhost".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![InputType::Text],
        cost: ModelCost {
            rates: ModelCostRates {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            tiers: None,
        },
        context_window: 8_192,
        max_tokens: 1_024,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// 执行后置位中断标志的工具（模拟第一个工具跑完时用户按了 Ctrl+C）。
struct ToolThatInterrupts;

impl AgentTool for ToolThatInterrupts {
    fn name(&self) -> &str {
        "first"
    }
    fn description(&self) -> &str {
        "第一个工具"
    }
    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    fn execute(&self, _call: &ToolCall) -> Result<ToolResult, ToolError> {
        pi_ai::interrupt::request();
        Ok(ToolResult::text("done"))
    }
}

/// 不应被执行的工具。
struct ShouldNotRunTool;

impl AgentTool for ShouldNotRunTool {
    fn name(&self) -> &str {
        "second"
    }
    fn description(&self) -> &str {
        "第二个工具（不应执行）"
    }
    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    fn execute(&self, _call: &ToolCall) -> Result<ToolResult, ToolError> {
        panic!("中断后不应执行这个工具");
    }
}

fn assistant_with_calls() -> AssistantMessage {
    AssistantMessage {
        role: AssistantRole::Assistant,
        content: vec![
            AssistantContent::ToolCall(ToolCall {
                id: "call_a".to_owned(),
                name: "first".to_owned(),
                arguments: Map::new(),
                thought_signature: None,
                namespace: None,
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "call_b".to_owned(),
                name: "second".to_owned(),
                arguments: Map::new(),
                thought_signature: None,
                namespace: None,
            }),
        ],
        api: "openai-completions".to_owned(),
        provider: "faux".to_owned(),
        model: "faux-1".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: zero_usage(),
        stop_reason: StopReason::ToolUse,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 2,
    }
}

#[test]
fn interrupt_stops_tool_batch_and_loop() {
    pi_ai::interrupt::clear();

    // ---- 场景 1：批次中中断 → 剩余调用补中断结果并强制终止 ----
    let faux = FauxProvider::new(vec![]);
    let mut agent = Agent::new(Box::new(faux), model());
    agent.add_tool(Box::new(ToolThatInterrupts));
    agent.add_tool(Box::new(ShouldNotRunTool));

    let (results, terminate) = agent.execute_tool_calls(&assistant_with_calls(), &mut |_| {});
    // 两个调用都有结果：first 正常执行，second 被中断。
    assert_eq!(results.len(), 2);
    match &results[0].content[0] {
        ToolResultContent::Text(text) => assert_eq!(text.text, "done"),
        other => panic!("expected text, got {other:?}"),
    }
    assert!(results[1].is_error);
    match &results[1].content[0] {
        ToolResultContent::Text(text) => assert_eq!(text.text, "已被用户中断"),
        other => panic!("expected text, got {other:?}"),
    }
    // 中断路径强制终止整批。
    assert!(terminate);

    pi_ai::interrupt::clear();

    // ---- 场景 2：完整 run 循环中工具执行时中断 → 循环停止，不再请求模型 ----
    let script = vec![
        FauxResponse::tool_call("call_1", "first", Map::new()),
        // 如果循环没停，第二个脚本项会被消费。
        FauxResponse::Text("不该到达".to_owned()),
    ];
    let faux = FauxProvider::with_script(vec![model()], script);
    let mut agent = Agent::new(Box::new(faux), model());
    agent.add_tool(Box::new(ToolThatInterrupts));
    agent.add_message(UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Text("run".to_owned()),
        timestamp: 1,
    });

    let reply = agent.run(&mut |_| {}).unwrap();

    // 工具执行时置位了中断：run_loop 停止，最终回复是带工具调用的那条消息。
    assert_eq!(reply.stop_reason, StopReason::ToolUse);
    // 消息序列：用户 -> 助手(工具调用) -> 工具结果。没有第二条助手回复。
    assert_eq!(agent.messages().len(), 3);

    pi_ai::interrupt::clear();
}
