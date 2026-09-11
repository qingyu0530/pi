use std::io::{self, Write};

use pi_agent_core::{Agent, AgentEvent, EditTool, ReadTool, RealEnvironment, WriteTool};
use pi_ai::{
    AssistantMessageEvent, FauxProvider, FauxResponse, InputType, Model, ModelCost, ModelCostRates,
    OpenAiCompletionsProvider, UreqTransport, UserMessage, UserMessageContent, UserRole,
};
use pi_tui::{PlainRenderer, Renderer};

/// 构造一个 OpenAI-compatible 模型描述。
/// 构造一个 OpenAI-compatible 的 Model（价格先设 0，实际费用靠 parse_usage 算）。
fn openai_model(base_url: &str, model_id: &str) -> Model {
    Model {
        id: model_id.to_owned(),
        name: model_id.to_owned(),
        api: "openai-completions".to_owned(),
        provider: "openai".to_owned(),
        base_url: base_url.to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![InputType::Text, InputType::Image],
        cost: ModelCost {
            rates: ModelCostRates {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            tiers: None,
        },
        context_window: 128_000,
        max_tokens: 4_096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// 给 Agent 注册内置工具。
/// 注册三件套（读/写/编辑），都用真实文件系统。
fn register_tools(agent: &mut Agent) {
    agent.add_tool(Box::new(ReadTool::new(Box::new(RealEnvironment))));
    agent.add_tool(Box::new(WriteTool::new(Box::new(RealEnvironment))));
    agent.add_tool(Box::new(EditTool::new(Box::new(RealEnvironment))));
}
// 选 Provider
fn main() -> io::Result<()> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "用一句话介绍你自己".to_owned());

    let (mut agent, model_label) = match std::env::var("OPENAI_API_KEY") {
        Ok(api_key) => {
            let base_url = std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_owned());
            let model_id =
                std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_owned());
            let model = openai_model(&base_url, &model_id);
            let provider = OpenAiCompletionsProvider::new(Box::new(UreqTransport::new()), api_key);
            (Agent::new(Box::new(provider), model), model_id)
        }
        Err(_) => {
            // 没有 API Key：用 FauxProvider 演示一次「读文件 → 回复」的流程。
            eprintln!("未设置 OPENAI_API_KEY，使用 FauxProvider 演示。");
            let model = openai_model("https://api.openai.com/v1", "faux-1");
            let arguments = serde_json::json!({ "path": "Cargo.toml" })
                .as_object()
                .unwrap()
                .clone();
            let provider = FauxProvider::with_script(
                vec![model.clone()],
                vec![
                    FauxResponse::tool_call("call_1", "read", arguments),
                    FauxResponse::Text("完成".to_owned()),
                ],
            );
            (Agent::new(Box::new(provider), model), "faux-1".to_owned())
        }
    };
    // 运行与流式输出
    register_tools(&mut agent);
    // 把命令行参数当用户消息加入
    agent.add_message(UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Text(prompt),
        timestamp: 0,
    });

    // 边流式输出文本，边打印工具执行情况。
    let reply = agent
        .run(&mut |event| match &event {
            AgentEvent::MessageUpdate { event, .. } => {
                if let AssistantMessageEvent::TextDelta { delta, .. } = event.as_ref() {
                    print!("{delta}");
                    let _ = io::stdout().flush();
                }
            }
            AgentEvent::ToolExecutionEnd {
                tool_name,
                is_error,
                ..
            } => {
                eprintln!("\n[工具 {tool_name} 执行完成, is_error={is_error}]");
            }
            _ => {}
        })
        .map_err(|error| io::Error::other(format!("{error:?}")))?;
    println!();

    let mut renderer = PlainRenderer;
    renderer.render_line(&format!(
        "Pi Rust {} | 模型 {} | {} 条消息 | 停止原因 {:?}",
        env!("CARGO_PKG_VERSION"),
        model_label,
        agent.messages().len(),
        reply.stop_reason
    ))
}
