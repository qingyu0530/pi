use std::io::{self, Write};

use pi_agent_core::{Agent, AgentEvent, EditTool, ReadTool, RealEnvironment, Session, WriteTool};
use pi_ai::{
    AssistantMessageEvent, FauxProvider, FauxResponse, InputType, Model, ModelCost, ModelCostRates,
    OpenAiCompletionsProvider, StopReason, UreqTransport, UserMessage, UserMessageContent,
    UserRole,
};
use pi_tui::{PlainRenderer, Renderer};

/// 默认的会话文件（可用 `PI_SESSION` 环境变量覆盖）。
const DEFAULT_SESSION_FILE: &str = "pi-session.jsonl";

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

/// CLI 会话：组合 Agent、会话日志、环境与文件路径。
struct Cli {
    agent: Agent,         // 对话运行时
    session: Session,     // 会话日志
    env: RealEnvironment, // 真实文件系统
    session_path: String, // 会话文件路径
    model_label: String,
    /// 已经保存进 session 的消息数，用来只追加新消息。
    persisted_messages: usize,
}

impl Cli {
    fn new() -> Self {
        let session_path =
            std::env::var("PI_SESSION").unwrap_or_else(|_| DEFAULT_SESSION_FILE.to_owned());
        let env = RealEnvironment;
        // 文件不存在时从空会话开始。
        let session = Session::load(&env, &session_path).unwrap_or_default();
        // 建 Agent、注册工具。
        let (mut agent, model_label) = build_agent();
        register_tools(&mut agent);
        // 把已保存的历史消息回放进 Agent。
        for message in session.messages() {
            agent.add_message(message);
        }
        // 记录初始已保存数
        let persisted_messages = agent.messages().len();

        Self {
            agent,
            session,
            env,
            session_path,
            model_label,
            persisted_messages,
        }
    }

    /// 执行一轮对话并持久化。
    fn run_turn(&mut self, prompt: String) -> io::Result<()> {
        self.agent.add_message(UserMessage {
            role: UserRole::User,
            content: UserMessageContent::Text(prompt),
            timestamp: 0,
        });

        let reply = self
            .agent
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

        self.persist()
    }

    /// 把尚未保存的新消息追加进会话并写盘。
    fn persist(&mut self) -> io::Result<()> {
        for message in &self.agent.messages()[self.persisted_messages..] {
            self.session.push_message(message.clone());
        }
        self.persisted_messages = self.agent.messages().len();
        self.session
            .save(&self.env, &self.session_path)
            .map_err(|error| io::Error::other(format!("保存会话失败: {}", error.message)))
    }

    /// 交互式循环：逐行读取用户输入；`exit`/`quit` 或 Ctrl-D 退出。
    fn repl(&mut self) -> io::Result<()> {
        let stdin = io::stdin();
        let mut line = String::new();
        loop {
            print!("> ");
            io::stdout().flush()?;

            line.clear();
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
            self.run_turn(input.to_owned())?;
        }
        Ok(())
    }
}

fn main() -> io::Result<()> {
    let mut cli = Cli::new();

    let mut renderer = PlainRenderer;
    renderer.render_line(&format!(
        "Pi Rust {} | 模型 {} | 会话 {} | 输入 exit 退出",
        env!("CARGO_PKG_VERSION"),
        cli.model_label,
        cli.session_path
    ))?;

    // 带命令行参数：一次性提问；不带参数：进入交互式对话。
    match std::env::args().nth(1) {
        Some(prompt) => cli.run_turn(prompt),
        None => cli.repl(),
    }
}
/*

会话是一串只追加的 Entry，用 JSONL 存盘；每条带 id/seq/parentId 形成链。
CLI 启动时回放历史，每轮只把新消息追加写盘。

*/
