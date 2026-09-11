use std::io::{self, Write};

use pi_agent_core::{Agent, AgentEvent, EditTool, ReadTool, RealEnvironment, WriteTool};
use pi_ai::{
    AssistantMessageEvent, FauxProvider, FauxResponse, InputType, Model, ModelCost, ModelCostRates,
    OpenAiCompletionsProvider, StopReason, UreqTransport, UserMessage, UserMessageContent,
    UserRole,
};
use pi_tui::{PlainRenderer, Renderer};

/// 构造一个 OpenAI-compatible 模型描述。
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
fn register_tools(agent: &mut Agent) {
    agent.add_tool(Box::new(ReadTool::new(Box::new(RealEnvironment))));
    agent.add_tool(Box::new(WriteTool::new(Box::new(RealEnvironment))));
    agent.add_tool(Box::new(EditTool::new(Box::new(RealEnvironment))));
}

/// 根据环境变量创建 Agent：有 API Key 走真实 Provider，否则回退 Faux 演示。
///
/// 返回 Agent 与用于展示的模型名。
fn build_agent() -> (Agent, String) {
    match std::env::var("OPENAI_API_KEY") {
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
    }
}

/// 执行一轮对话：加入用户消息，运行，并边流式打印结果。
fn run_turn(agent: &mut Agent, prompt: String) -> io::Result<()> {
    agent.add_message(UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Text(prompt),
        timestamp: 0,
    });

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

    if reply.stop_reason == StopReason::Error {
        eprintln!(
            "[本轮出错: {}]",
            reply.error_message.as_deref().unwrap_or("未知错误")
        );
    }
    Ok(())
}

/// 交互式循环：逐行读取用户输入；输入 `exit`/`quit` 或按 Ctrl-D 退出。
fn repl(agent: &mut Agent) -> io::Result<()> {
    let stdin = io::stdin();
    let mut line = String::new();
    loop {
        print!("> ");
        io::stdout().flush()?;

        line.clear();
        // read_line 返回读取到的字节数；0 表示 stdin 结束（Ctrl-D）。
        if stdin.read_line(&mut line)? == 0 {
            println!();
            break;
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input == "exit" || input == "quit" {
            break;
        }
        run_turn(agent, input.to_owned())?;
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let (mut agent, model_label) = build_agent();
    register_tools(&mut agent);

    let mut renderer = PlainRenderer;
    renderer.render_line(&format!(
        "Pi Rust {} | 模型 {} | 输入 exit 退出",
        env!("CARGO_PKG_VERSION"),
        model_label
    ))?;

    // 带命令行参数：一次性提问；不带参数：进入交互式对话。
    match std::env::args().nth(1) {
        Some(prompt) => run_turn(&mut agent, prompt),
        None => repl(&mut agent),
    }
}
