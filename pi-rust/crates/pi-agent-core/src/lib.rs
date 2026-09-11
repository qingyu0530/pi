//! Core state and behavior for a Pi agent.
//!
//! pi-ai 定义了「消息」「模型」「Provider」这些零件；
//! pi-agent-core 负责把它们组装成一次真正的对话：
//!   用户消息 -> 组装 Context -> 调 Provider -> 收集流式事件 -> 得到助手消息
//!
//! C++ 对照：
//!   Agent 持有 Box<dyn Provider> ≈ 持有一个抽象基类指针，
//!   运行时才决定具体用哪个子类（FauxProvider 或真实 Provider）。

mod compaction;
mod environment;
mod event;
mod session;
mod tool;
mod tools;
mod truncate;

pub use compaction::{
    CompactionError, CompactionPlan, CompactionResult, CompactionSettings,
    DEFAULT_COMPACTION_SETTINGS, build_summary_prompt, compact, estimate_context_tokens,
    estimate_tokens, plan_compaction, should_compact,
};
pub use environment::{EnvError, Environment, RealEnvironment};
pub use event::AgentEvent;
pub use session::{Entry, Session, SessionError};
pub use tool::{AgentTool, EchoTool, ToolError, ToolResult};
pub use tools::{EditTool, ReadTool, WriteTool};

use pi_ai::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, Context, ConversationMessage, Model,
    Provider, StopReason, ToolCall, ToolResultMessage, ToolResultRole, UserMessage,
    UserMessageContent, UserRole,
};
use serde_json::Value;

/// 单次 `run` 允许的最大轮数，防止 Provider 一直返回工具调用导致死循环。
const MAX_TURNS: usize = 32;

/// Agent 运行过程中可能出现的错误。
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AgentError {
    /// 事件流在结束前没有给出 done 或 error 事件。
    IncompleteStream,
    /// 对话轮数超过上限，可能是模型一直发起工具调用。
    MaxTurnsExceeded { limit: usize },
    /// 上下文压缩失败。
    Compaction(String),
}

/// 一次 Agent 会话的状态与行为。
///
/// 它持有：
/// - provider：把请求变成模型回复的执行者（`Box<dyn Provider>`）。
/// - model：本次会话使用的模型。
/// - system_prompt：每次请求都会带上的系统提示。
/// - tools：本次会话可用的工具。
/// - messages：完整对话记录。
///
/// C++ 对照：`Box<dyn Provider>` 类似 `std::unique_ptr<Provider>`，
/// 其中 `Provider` 是抽象基类；`Box` 表示独占所有权。
pub struct Agent {
    provider: Box<dyn Provider>,
    model: Model,
    system_prompt: Option<String>,
    tools: Vec<Box<dyn AgentTool>>,
    messages: Vec<ConversationMessage>,
}

impl Agent {
    /// 用指定的 Provider 和模型创建一个 Agent。
    #[must_use]
    pub fn new(provider: Box<dyn Provider>, model: Model) -> Self {
        Self {
            provider,
            model,
            system_prompt: None,
            tools: Vec::new(),
            messages: Vec::new(),
        }
    }

    /// 设置系统提示（链式写法，方便创建时一次写完）。
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// 注册一个工具。注册的工具会随请求发给模型，并可用于执行工具调用。
    pub fn add_tool(&mut self, tool: Box<dyn AgentTool>) {
        self.tools.push(tool);
    }

    /// 追加一条消息到对话记录。
    pub fn add_message(&mut self, message: impl Into<ConversationMessage>) {
        self.messages.push(message.into());
    }

    /// 只读访问完整对话记录。
    #[must_use]
    pub fn messages(&self) -> &[ConversationMessage] {
        &self.messages
    }

    /// 单轮对话：把当前消息发给模型，收集流式事件，返回并追加最终的助手消息。
    ///
    /// 过程中会向 `sink` 发出消息级事件：
    /// - `Start` -> `MessageStart`   助手消息开始（收到 Start 时）
    /// - 各种 start/delta/end -> `MessageUpdate`  流式中间更新（text/thinking/toolcall 的 start/delta/end）
    /// - `Done`/`Error` -> `MessageEnd`  消息结束（拿到最终 Done/Error 后）
    ///
    /// C++ 对照：`sink` 是一个回调函数对象（`std::function<void(AgentEvent)>`）。
    pub fn run_once(
        &mut self,
        sink: &mut dyn FnMut(AgentEvent),
    ) -> Result<AssistantMessage, AgentError> {
        // 把 Agent 的当前状态组装成一次请求的输入。
        let context = Context {
            system_prompt: self.system_prompt.clone(),
            messages: self.messages.clone(),
            tools: self.tool_definitions(),
        };

        let mut final_message = None;
        for event in self.provider.stream(&self.model, &context) {
            match &event {
                // 流开始：对应一条助手消息开始。
                AssistantMessageEvent::Start { partial } => {
                    sink(AgentEvent::MessageStart {
                        message: Box::new(partial.clone().into()),
                    });
                }
                // done/error 携带最终消息。
                AssistantMessageEvent::Done { message, .. }
                | AssistantMessageEvent::Error { error: message, .. } => {
                    final_message = Some(message.clone());
                }
                // 其余都是携带 partial 的中间事件：转发给 UI。
                other => {
                    if let Some(partial) = other.partial() {
                        sink(AgentEvent::MessageUpdate {
                            message: Box::new(partial.clone()),
                            event: Box::new(other.clone()),
                        });
                    }
                }
            }
        }

        let message = final_message.ok_or(AgentError::IncompleteStream)?;
        sink(AgentEvent::MessageEnd {
            message: Box::new(message.clone().into()),
        });
        self.messages.push(message.clone().into());
        Ok(message)
    }

    /// 完整对话循环：反复「请求模型 → 执行工具 → 再请求」，
    /// 直到助手不再发起工具调用（或出错/中止），返回最后一条助手消息。
    ///
    /// 过程中会发出 run 级和 turn 级事件。
    /// C++ 对照：类似一个 while 循环，条件由每轮模型的输出决定。
    ///
    /// 拆出 `run_loop` 是为了保证 `agent_end` 一定发出（成功、出错、超限都发），
    /// 让 UI 能正确收尾。
    pub fn run(
        &mut self,
        sink: &mut dyn FnMut(AgentEvent),
    ) -> Result<AssistantMessage, AgentError> {
        // run 只做两件事：发开头、发结尾，中间交给 run_loop
        sink(AgentEvent::AgentStart);
        let result = self.run_loop(sink); // 内部循环
        // 无论成功失败，都发出 agent_end，让 UI 知道运行结束。
        sink(AgentEvent::AgentEnd {
            messages: self.messages.clone(),
        });
        result
    }

    /// `run` 的内部循环，单独抽出来是为了让 `agent_end` 总能发出。
    fn run_loop(
        &mut self,
        sink: &mut dyn FnMut(AgentEvent),
    ) -> Result<AssistantMessage, AgentError> {
        for _ in 0..MAX_TURNS {
            sink(AgentEvent::TurnStart);
            let message = self.run_once(sink)?;

            // 模型出错或被中止：本轮结束，不再继续。
            if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
                sink(AgentEvent::TurnEnd {
                    message: Box::new(message.clone()),
                    tool_results: Vec::new(),
                });
                return Ok(message);
            }

            // 没有工具调用：说明模型给出了最终回答，结束循环。
            let has_tool_calls = message
                .content
                .iter()
                .any(|block| matches!(block, AssistantContent::ToolCall(_)));
            if !has_tool_calls {
                sink(AgentEvent::TurnEnd {
                    message: Box::new(message.clone()),
                    tool_results: Vec::new(),
                });
                return Ok(message);
            }

            // 执行工具，把结果追加进对话，然后进入下一轮请求。
            let tool_results = self.execute_tool_calls(&message, sink);
            sink(AgentEvent::TurnEnd {
                message: Box::new(message.clone()),
                tool_results,
            });
        }

        Err(AgentError::MaxTurnsExceeded { limit: MAX_TURNS })
    }

    /// 执行助手消息里的所有工具调用，追加对应的工具结果消息并返回它们。
    ///
    /// 每个工具调用会发出 `ToolExecutionStart` / `ToolExecutionEnd`，
    /// 工具结果本身作为一条消息发出 `MessageStart` / `MessageEnd`。
    pub fn execute_tool_calls(
        &mut self,
        message: &AssistantMessage,
        sink: &mut dyn FnMut(AgentEvent),
    ) -> Vec<ToolResultMessage> {
        let mut results = Vec::new();
        for block in &message.content {
            if let AssistantContent::ToolCall(call) = block {
                sink(AgentEvent::ToolExecutionStart {
                    tool_call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    args: Value::Object(call.arguments.clone()),
                });

                let (result, is_error) = self.execute_one(call);
                sink(AgentEvent::ToolExecutionEnd {
                    tool_call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    result: Box::new(result.clone()),
                    is_error,
                });

                let result_message = tool_result_message(call, result, is_error);
                sink(AgentEvent::MessageStart {
                    message: Box::new(result_message.clone().into()),
                });
                sink(AgentEvent::MessageEnd {
                    message: Box::new(result_message.clone().into()),
                });
                self.messages.push(result_message.clone().into());
                results.push(result_message);
            }
        }
        results
    }

    /// 执行单个工具调用，返回工具结果和是否失败。
    ///
    /// 工具不存在或执行失败时返回一条错误文本结果，而不是中断整个流程，
    /// 这样模型能看到错误并自行调整。
    fn execute_one(&self, call: &ToolCall) -> (ToolResult, bool) {
        match self.find_tool(&call.name) {
            Some(tool) => match tool.execute(call) {
                Ok(result) => (result, false),
                Err(error) => (ToolResult::text(error.message), true),
            },
            None => (ToolResult::text(format!("未知工具: {}", call.name)), true),
        }
    }

    /// 按名字查找已注册的工具。
    fn find_tool(&self, name: &str) -> Option<&dyn AgentTool> {
        self.tools
            .iter()
            .find(|tool| tool.name() == name)
            .map(AsRef::as_ref)
    }

    /// 把已注册的工具转换成发给模型的工具定义；没有工具时返回 `None`。
    fn tool_definitions(&self) -> Option<Vec<pi_ai::Tool>> {
        if self.tools.is_empty() {
            return None;
        }
        Some(
            self.tools
                .iter()
                .map(|tool| pi_ai::Tool {
                    name: tool.name().to_owned(),
                    description: tool.description().to_owned(),
                    parameters: tool.parameters(),
                    constrained_sampling: None,
                })
                .collect(),
        )
    }

    /// 若上下文超过阈值，把旧消息摘要成一段总结并替换，返回压缩结果。
    ///
    /// 摘要由本 Agent 的 Provider 生成（用系统提示 + 要摘要的旧消息）。
    /// 压缩后消息列表变成「一条摘要消息 + 保留的近期消息」。
    pub fn maybe_compact(
        &mut self,
        settings: CompactionSettings, // 压缩设置（阈值、保留量）
    ) -> Result<Option<CompactionResult>, AgentError> {
        // 把当前所有消息折成 token 估算值。
        let context_tokens = estimate_context_tokens(&self.messages);
        // self.model.context_window：该模型的上下文窗口大小。
        // 没到阈值 → 直接返回 Ok(None)（不压缩）。提前退出。

        if !should_compact(context_tokens, self.model.context_window, settings) {
            return Ok(None);
        }

        // 先克隆，避免闭包借用 self 时与外面的可变借用冲突。
        let model = self.model.clone();
        let messages = self.messages.clone();
        let provider = self.provider.as_ref();
        // 执行压缩 消息、设置、一个闭包作为摘要器
        let result = compact(&messages, settings, |system, prompt| {
            summarize_with_provider(provider, &model, system, prompt)
        }) // ?：失败就提前返回错误
        .map_err(|error| AgentError::Compaction(error.message))?;

        // 用「摘要 + 保留的近期消息」替换原来的全部消息。
        let mut new_messages = Vec::with_capacity(result.retained_tail.len() + 1);// 预分配容量，减少扩容
        new_messages.push(summary_message(&result.summary)); // 第一条放摘要
        new_messages.extend(result.retained_tail.iter().cloned()); // 把保留的近期消息接上去
        self.messages = new_messages; // 整体替换（这就是「替代」）。
        // 返回压缩结果（Some 表示确实压缩了）
        Ok(Some(result))
    }
}

/// 把摘要包成一条用户消息，作为压缩后的上下文。
fn summary_message(summary: &str) -> ConversationMessage {
    UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Text(format!(
            "[Earlier conversation summarized to save context]\n\n{summary}"
        )),
        timestamp: 0,
    }
    .into()
}

/// 用给定 Provider 生成摘要文本。
fn summarize_with_provider(
    provider: &dyn Provider,
    model: &Model,
    system: &str,
    prompt: &str,
) -> Result<String, CompactionError> {
    // 构造请求上下文
    // 把「请你总结这段对话」组织成一次普通模型请求
    let context = Context {
        system_prompt: Some(system.to_owned()),
        messages: vec![
            UserMessage {
                role: UserRole::User,
                content: UserMessageContent::Text(prompt.to_owned()),
                timestamp: 0,
            }
            .into(),
        ],
        tools: None,
    };

    let mut summary = String::new();
    let mut done = false;
    for event in provider.stream(model, &context) {
        match event {
            AssistantMessageEvent::Done { message, .. } => {
                for block in message.content {
                    if let AssistantContent::Text(text) = block {
                        summary.push_str(&text.text);
                    }
                }
                done = true;
            }
            AssistantMessageEvent::Error { error, .. } => {
                return Err(CompactionError::new(
                    error
                        .error_message
                        .unwrap_or_else(|| "摘要生成失败".to_owned()),
                ));
            }
            _ => {}
        }
    }
    if !done {
        return Err(CompactionError::new("摘要流未正常结束"));
    }
    Ok(summary)
}

/// 把一次工具执行的结果组装成工具结果消息。
fn tool_result_message(call: &ToolCall, result: ToolResult, is_error: bool) -> ToolResultMessage {
    ToolResultMessage {
        role: ToolResultRole::ToolResult,
        tool_call_id: call.id.clone(),
        tool_name: call.name.clone(),
        content: result.content,
        details: result.details,
        usage: result.usage,
        added_tool_names: result.added_tool_names,
        is_error,
        timestamp: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{Agent, AgentError, AgentEvent};
    use pi_ai::{
        AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantRole, Context,
        ConversationMessage, FauxProvider, FauxResponse, InputType, Model, ModelCost,
        ModelCostRates, Provider, StopReason, TextContent, ToolCall, ToolResultContent,
        ToolResultMessage, ToolResultRole, Usage, UsageCost, UserMessage, UserMessageContent,
        UserRole,
    };
    use serde_json::{Map, Value};

    /// 一个永远返回空事件流的测试 Provider，用来触发「流不完整」错误。
    struct EmptyProvider;

    impl Provider for EmptyProvider {
        fn id(&self) -> &str {
            "empty"
        }

        fn name(&self) -> &str {
            "Empty Provider"
        }

        fn get_models(&self) -> &[Model] {
            &[]
        }

        fn stream(
            &self,
            _model: &Model,
            _context: &Context,
        ) -> Box<dyn Iterator<Item = AssistantMessageEvent>> {
            Box::new(std::iter::empty())
        }
    }

    /// 一个永远返回工具调用的测试 Provider，用来触发「轮数超限」。
    struct AlwaysToolProvider;

    impl Provider for AlwaysToolProvider {
        fn id(&self) -> &str {
            "always-tool"
        }

        fn name(&self) -> &str {
            "Always Tool Provider"
        }

        fn get_models(&self) -> &[Model] {
            &[]
        }

        fn stream(
            &self,
            _model: &Model,
            _context: &Context,
        ) -> Box<dyn Iterator<Item = AssistantMessageEvent>> {
            let message = tool_call_message("call", "echo", Map::new());
            let events = vec![
                AssistantMessageEvent::Start {
                    partial: message.clone(),
                },
                AssistantMessageEvent::Done {
                    reason: StopReason::ToolUse,
                    message,
                },
            ];
            Box::new(events.into_iter())
        }
    }

    fn faux_model() -> Model {
        Model {
            id: "faux-1".to_owned(),
            name: "Faux 1".to_owned(),
            api: "faux".to_owned(),
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

    fn user_message(text: &str) -> UserMessage {
        UserMessage {
            role: UserRole::User,
            content: UserMessageContent::Text(text.to_owned()),
            timestamp: 1,
        }
    }

    fn empty_usage() -> Usage {
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

    fn tool_call_message(id: &str, name: &str, arguments: Map<String, Value>) -> AssistantMessage {
        AssistantMessage {
            role: AssistantRole::Assistant,
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: id.to_owned(),
                name: name.to_owned(),
                arguments,
                thought_signature: None,
                namespace: None,
            })],
            api: "faux".to_owned(),
            provider: "faux".to_owned(),
            model: "faux-1".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: empty_usage(),
            stop_reason: StopReason::ToolUse,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: Some(false),
            timestamp: 2,
        }
    }

    fn arguments(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn run_once_streams_and_appends_assistant_message() {
        let mut agent = Agent::new(
            Box::new(FauxProvider::new(vec![faux_model()])),
            faux_model(),
        );
        agent.add_message(user_message("你好"));

        let message = agent.run_once(&mut |_| {}).unwrap();

        assert_eq!(message.stop_reason, StopReason::Stop);
        assert_eq!(agent.messages().len(), 2);
        assert!(matches!(
            agent.messages()[1],
            ConversationMessage::Assistant(_)
        ));
        // FauxProvider 会把最后一条用户文本回显回来。
        match &message.content[0] {
            AssistantContent::Text(text) => assert_eq!(text.text, "你好"),
            other => panic!("unexpected content: {other:?}"),
        }
    }

    #[test]
    fn run_once_without_done_or_error_is_incomplete() {
        let mut agent = Agent::new(Box::new(EmptyProvider), faux_model());
        agent.add_message(user_message("hi"));

        assert_eq!(
            agent.run_once(&mut |_| {}),
            Err(AgentError::IncompleteStream)
        );
    }

    #[test]
    fn execute_tool_calls_runs_registered_tool_and_appends_result() {
        let mut agent = Agent::new(Box::new(FauxProvider::new(vec![])), faux_model());
        agent.add_tool(Box::new(super::EchoTool));

        let message = tool_call_message(
            "call_1",
            "echo",
            arguments(serde_json::json!({ "text": "hi" })),
        );
        let results = agent.execute_tool_calls(&message, &mut |_| {});

        assert_eq!(results.len(), 1);
        assert!(!results[0].is_error);
        assert_eq!(results[0].tool_call_id, "call_1");
        assert_eq!(results[0].tool_name, "echo");
        // 结果消息也被追加进了对话记录。
        assert_eq!(agent.messages().len(), 1);
        assert!(matches!(
            agent.messages()[0],
            ConversationMessage::ToolResult(_)
        ));
        match &results[0].content[0] {
            ToolResultContent::Text(text) => assert_eq!(text.text, "hi"),
            other => panic!("unexpected content: {other:?}"),
        }
    }

    #[test]
    fn execute_tool_calls_marks_unknown_tool_as_error() {
        let mut agent = Agent::new(Box::new(FauxProvider::new(vec![])), faux_model());

        let message = tool_call_message("call_1", "missing", Map::new());
        let results = agent.execute_tool_calls(&message, &mut |_| {});

        assert_eq!(results.len(), 1);
        assert!(results[0].is_error);
        match &results[0].content[0] {
            ToolResultContent::Text(text) => assert!(text.text.contains("未知工具")),
            other => panic!("unexpected content: {other:?}"),
        }
    }

    #[test]
    fn run_executes_tool_call_and_continues_until_text() {
        let script = vec![
            FauxResponse::tool_call(
                "call_1",
                "echo",
                arguments(serde_json::json!({ "text": "hi" })),
            ),
            FauxResponse::Text("done".to_owned()),
        ];
        let provider = FauxProvider::with_script(vec![faux_model()], script);
        let mut agent = Agent::new(Box::new(provider), faux_model());
        agent.add_tool(Box::new(super::EchoTool));
        agent.add_message(user_message("echo hi"));

        let final_message = agent.run(&mut |_| {}).unwrap();

        assert_eq!(final_message.stop_reason, StopReason::Stop);
        // user -> assistant(工具调用) -> toolResult -> assistant(最终文本)
        assert_eq!(agent.messages().len(), 4);
        assert!(matches!(
            agent.messages()[1],
            ConversationMessage::Assistant(_)
        ));
        assert!(matches!(
            agent.messages()[2],
            ConversationMessage::ToolResult(_)
        ));
        match &final_message.content[0] {
            AssistantContent::Text(text) => assert_eq!(text.text, "done"),
            other => panic!("unexpected content: {other:?}"),
        }
    }

    #[test]
    fn run_emits_turn_and_message_events() {
        let script = vec![
            FauxResponse::tool_call(
                "call_1",
                "echo",
                arguments(serde_json::json!({ "text": "hi" })),
            ),
            FauxResponse::Text("done".to_owned()),
        ];
        let provider = FauxProvider::with_script(vec![faux_model()], script);
        let mut agent = Agent::new(Box::new(provider), faux_model());
        agent.add_tool(Box::new(super::EchoTool));
        agent.add_message(user_message("echo hi"));

        let mut events = Vec::new();
        agent.run(&mut |event| events.push(event)).unwrap();

        // run 级事件：首尾分别是 agent_start 和 agent_end。
        assert!(matches!(events.first(), Some(AgentEvent::AgentStart)));
        assert!(matches!(events.last(), Some(AgentEvent::AgentEnd { .. })));
        // 两个 turn（一次工具调用轮 + 一次最终文本轮）。
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, AgentEvent::TurnStart))
                .count(),
            2
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, AgentEvent::TurnEnd { .. }))
                .count(),
            2
        );
        // 工具执行事件。
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolExecutionStart { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }))
        );
        // 流式消息更新事件。
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::MessageUpdate { .. }))
        );
    }

    #[test]
    fn run_stops_at_max_turns() {
        let mut agent = Agent::new(Box::new(AlwaysToolProvider), faux_model());
        agent.add_tool(Box::new(super::EchoTool));
        agent.add_message(user_message("loop forever"));

        assert!(matches!(
            agent.run(&mut |_| {}),
            Err(AgentError::MaxTurnsExceeded { .. })
        ));
    }

    #[test]
    fn transcript_round_trips_user_tool_call_and_result() {
        let mut agent = Agent::new(Box::new(FauxProvider::new(vec![])), faux_model());

        agent.add_message(user_message("读取 README"));
        agent.add_message(AssistantMessage {
            role: AssistantRole::Assistant,
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: "call_1".to_owned(),
                name: "read_file".to_owned(),
                arguments: Map::new(),
                thought_signature: None,
                namespace: None,
            })],
            api: "faux".to_owned(),
            provider: "faux".to_owned(),
            model: "faux-1".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: empty_usage(),
            stop_reason: StopReason::ToolUse,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: Some(false),
            timestamp: 2,
        });
        agent.add_message(ToolResultMessage {
            role: ToolResultRole::ToolResult,
            tool_call_id: "call_1".to_owned(),
            tool_name: "read_file".to_owned(),
            content: vec![ToolResultContent::Text(TextContent {
                text: "Hello, Pi".to_owned(),
                text_signature: None,
            })],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 3,
        });

        let messages = agent.messages();
        assert_eq!(messages.len(), 3);
        assert!(matches!(messages[0], ConversationMessage::User(_)));
        assert!(matches!(messages[1], ConversationMessage::Assistant(_)));
        assert!(matches!(messages[2], ConversationMessage::ToolResult(_)));
    }
}
