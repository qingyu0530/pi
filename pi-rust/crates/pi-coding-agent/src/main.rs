use std::io::{self, Write};

use pi_agent_core::{
    Agent, BashTool, DEFAULT_COMPACTION_SETTINGS, EditTool, FindTool, GrepTool, LsTool, ReadTool,
    RealEnvironment, RealShell, Session, WriteTool,
};
use pi_ai::{
    FauxProvider, FauxResponse, InputType, Model, ModelCost, ModelCostRates, ModelRegistry,
    ModelThinkingLevel, OpenAiCompletionsProvider, RequestOptions, StopReason, UreqTransport,
    UserMessage, UserMessageContent, UserRole,
};
use pi_tui::{EventRenderer, PlainRenderer, Renderer};

mod args;

use args::{Args, ListModels, help_text, parse_args};

/// 默认的会话文件（可用 `--session` 或 `PI_SESSION` 覆盖）。
const DEFAULT_SESSION_FILE: &str = "pi-session.jsonl";

/// 演示用的极简模型（没有真实 API Key 时走 Faux）。
fn demo_model() -> Model {
    Model {
        id: "faux-1".to_owned(),
        name: "Faux 1".to_owned(),
        api: "openai-completions".to_owned(),
        provider: "faux".to_owned(),
        base_url: "https://api.openai.com/v1".to_owned(),
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
        context_window: 128_000,
        max_tokens: 4_096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// 构造 Faux 演示 Agent；`scripted` 为真时先调用 read 再回文本。
fn faux_agent(scripted: bool) -> (Agent, String) {
    let model = demo_model();
    let provider = if scripted {
        let arguments = serde_json::json!({ "path": "Cargo.toml" })
            .as_object()
            .unwrap()
            .clone();
        FauxProvider::with_script(
            vec![model.clone()],
            vec![
                FauxResponse::tool_call("call_1", "read", arguments),
                FauxResponse::Text("完成".to_owned()),
            ],
        )
    } else {
        FauxProvider::new(vec![model.clone()])
    };
    (Agent::new(Box::new(provider), model), "faux-1".to_owned())
}

/// 给 Agent 注册内置工具。
fn register_tools(agent: &mut Agent) {
    agent.add_tool(Box::new(ReadTool::new(Box::new(RealEnvironment))));
    agent.add_tool(Box::new(WriteTool::new(Box::new(RealEnvironment))));
    agent.add_tool(Box::new(EditTool::new(Box::new(RealEnvironment))));
    agent.add_tool(Box::new(LsTool::new(Box::new(RealEnvironment))));
    agent.add_tool(Box::new(FindTool::new(Box::new(RealEnvironment))));
    agent.add_tool(Box::new(GrepTool::new(Box::new(RealEnvironment))));
    agent.add_tool(Box::new(BashTool::new(Box::new(RealShell))));
}

/// 根据命令行参数、环境变量与内置模型目录创建 Agent。
///
/// 优先级：命令行参数 > 环境变量 > 默认值。
/// - `--provider` / `PI_PROVIDER`、`--model` / `PI_MODEL` 选择模型（默认 openai/gpt-4o-mini）。
/// - `--api-key` / `OPENAI_API_KEY` 提供密钥；没有则回退 Faux 演示。
///
/// 决定用真实 Provider 还是 Faux，并选出模型。
fn build_agent(args: &Args) -> (Agent, String) {
    let registry = ModelRegistry::builtin();
    let provider_name = args
        .provider
        .clone()
        .or_else(|| std::env::var("PI_PROVIDER").ok())
        .unwrap_or_else(|| "openai".to_owned());
    let model_id = args
        .model
        .clone()
        .or_else(|| std::env::var("PI_MODEL").ok())
        .unwrap_or_else(|| "gpt-4o-mini".to_owned());

    let api_key = args
        .api_key
        .clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok());

    let Some(api_key) = api_key else {
        eprintln!("未提供 API Key（--api-key 或 OPENAI_API_KEY），使用 FauxProvider 演示。");
        return faux_agent(true);
    };

    let Some(model) = registry.get(&provider_name, &model_id).cloned() else {
        eprintln!("模型目录中没有 {provider_name}/{model_id}，使用 FauxProvider 演示。");
        eprintln!("可用 provider: {}", registry.providers().join(", "));
        return faux_agent(false);
    };

    if model.api != "openai-completions" {
        eprintln!("暂不支持的 api: {}，使用 FauxProvider 演示。", model.api);
        return faux_agent(false);
    }

    let provider = OpenAiCompletionsProvider::new(Box::new(UreqTransport::new()), api_key);
    (
        Agent::new(Box::new(provider), model),
        format!("{provider_name}/{model_id}"),
    )
}

/// CLI 会话：组合 Agent、会话日志、环境与文件路径。
struct Cli {
    agent: Agent,         // 对话运行时
    session: Session,     // 会话日志
    env: RealEnvironment, // 真实文件系统
    session_path: String, // 会话文件路径
    model_label: String,
    /// 是否把会话写盘（`--no-session` 时为 false）。
    persist_session: bool,
    /// 已经保存进 session 的消息数，用来只追加新消息。
    persisted_messages: usize,
}

impl Cli {
    fn new(args: &Args) -> io::Result<Self> {
        // 按参数装配整个会话：会话文件、Agent、请求选项、工具、历史回放。
        let session_path = args
            .session
            .clone()
            .or_else(|| std::env::var("PI_SESSION").ok())
            .unwrap_or_else(|| DEFAULT_SESSION_FILE.to_owned());
        let env = RealEnvironment;
        // `--no-session` 时不读盘，从空会话开始。
        let session = if args.no_session {
            Session::new()
        } else {
            Session::load(&env, &session_path).unwrap_or_default()
        };

        let (agent, model_label) = build_agent(args);

        // 解析思考级别（若给了 --thinking）。
        let reasoning_effort = match &args.thinking {
            Some(level) => Some(parse_thinking(level).map_err(io::Error::other)?),
            None => None,
        };

        // 会话亲和：用会话文件路径作为 session id（`--no-session` 时不设）。
        let session_id = if args.no_session {
            None
        } else {
            Some(session_path.clone())
        };
        let mut agent = agent.with_options(RequestOptions {
            session_id,
            reasoning_effort,
            ..RequestOptions::default()
        });

        // 系统提示。
        if let Some(prompt) = &args.system_prompt {
            agent = agent.with_system_prompt(prompt.clone());
        }

        register_tools(&mut agent);
        // 把已保存的历史消息回放进 Agent。
        for message in session.messages() {
            agent.add_message(message);
        }
        // 记录初始已保存数
        let persisted_messages = agent.messages().len();

        Ok(Self {
            agent,
            session,
            env,
            session_path,
            model_label,
            persist_session: !args.no_session,
            persisted_messages,
        })
    }

    /// 执行一轮对话并持久化。
    fn run_turn(&mut self, prompt: String) -> io::Result<()> {
        // 先检查上下文是否过长；需要时压缩后再开始本轮。
        self.compact_if_needed()?;
        // 之后才把用户消息加进去
        self.agent.add_message(UserMessage {
            role: UserRole::User,
            content: UserMessageContent::Text(prompt),
            timestamp: 0,
        });

        // 用 pi-tui 的事件渲染器做增量渲染。
        let mut renderer = EventRenderer::new(Box::new(PlainRenderer));
        let mut render_error: Option<io::Error> = None;
        let reply = self
            .agent
            .run(&mut |event| {
                if let Err(error) = renderer.handle(&event) {
                    render_error.get_or_insert(error);
                }
            })
            .map_err(|error| io::Error::other(format!("{error:?}")))?;
        if let Some(error) = render_error {
            return Err(error);
        }
        println!();

        if reply.stop_reason == StopReason::Error {
            eprintln!(
                "[本轮出错: {}]",
                reply.error_message.as_deref().unwrap_or("未知错误")
            );
        }

        self.persist()
    }

    /// 若上下文超过阈值，用当前 Provider 生成摘要并替换旧消息；随后重建会话文件。
    fn compact_if_needed(&mut self) -> io::Result<()> {
        let result = self
            .agent
            .maybe_compact(DEFAULT_COMPACTION_SETTINGS)
            .map_err(|error| io::Error::other(format!("压缩失败: {error:?}")))?;

        let Some(result) = result else {
            return Ok(());
        };

        eprintln!(
            "\n[已压缩上下文：现有 {} 条消息，压缩前约 {} token]",
            self.agent.messages().len(),
            result.tokens_before
        );
        // 重建会话
        // 压缩改变了消息列表，用当前消息重建会话（丢弃被摘要掉的旧记录）。
        self.session = Session::new();
        for message in self.agent.messages() {
            self.session.push_message(message.clone());
        }
        self.persisted_messages = self.agent.messages().len();
        if !self.persist_session {
            return Ok(());
        }
        self.session // 保存
            .save(&self.env, &self.session_path)
            .map_err(|error| io::Error::other(format!("保存会话失败: {}", error.message)))
    }

    /// 把尚未保存的新消息追加进会话并写盘。
    fn persist(&mut self) -> io::Result<()> {
        for message in &self.agent.messages()[self.persisted_messages..] {
            self.session.push_message(message.clone());
        }
        self.persisted_messages = self.agent.messages().len();
        if !self.persist_session {
            return Ok(());
        }
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

/// 把 `--thinking` 的字符串映射成思考级别。   把 --thinking 的字符串转成 ModelThinkingLevel；非法值返回错误。
fn parse_thinking(level: &str) -> Result<ModelThinkingLevel, String> {
    match level {
        "off" => Ok(ModelThinkingLevel::Off),
        "minimal" => Ok(ModelThinkingLevel::Minimal),
        "low" => Ok(ModelThinkingLevel::Low),
        "medium" => Ok(ModelThinkingLevel::Medium),
        "high" => Ok(ModelThinkingLevel::High),
        "xhigh" => Ok(ModelThinkingLevel::Xhigh),
        "max" => Ok(ModelThinkingLevel::Max),
        other => Err(format!(
            "无效的 thinking 级别: {other}（可选 off/minimal/low/medium/high/xhigh/max）"
        )),
    }
}

/// 打印模型目录，`filter` 可带子串过滤。   
fn print_models(filter: &ListModels) {
    let registry = ModelRegistry::builtin();
    let needle = match filter {
        ListModels::All => None,
        ListModels::Search(text) => Some(text.to_lowercase()),
    };
    for model in registry.models() {
        let label = format!("{}/{}", model.provider, model.id);
        if let Some(needle) = &needle {
            if !label.to_lowercase().contains(needle.as_str()) {
                continue;
            }
        }
        println!("{label}  ({})", model.api);
    }
}
// 解析参数 → 处理 --help/--version/--list-models 这些「只打印」模式 → 否则建会话、跑初始消息、进入 REPL 或退出。
fn main() -> io::Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&raw) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("参数错误: {message}");
            eprint!("{}", help_text());
            return Err(io::Error::other("参数解析失败"));
        }
    };

    // 只打印信息、不启动会话的模式。
    if args.help {
        print!("{}", help_text());
        return Ok(());
    }
    if args.version {
        println!("Pi Rust {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if let Some(filter) = &args.list_models {
        print_models(filter);
        return Ok(());
    }

    let mut cli = Cli::new(&args)?;

    let mut renderer = PlainRenderer;
    renderer.render_line(&format!(
        "Pi Rust {} | 模型 {} | 会话 {} | 输入 exit 退出",
        env!("CARGO_PKG_VERSION"),
        cli.model_label,
        cli.session_path
    ))?;

    // 位置参数先作为初始消息逐条执行。
    for message in &args.messages {
        cli.run_turn(message.clone())?;
    }

    // `--print`：处理完就退出；否则进入交互式对话。
    if args.print {
        return Ok(());
    }
    cli.repl()
}
/*

会话是一串只追加的 Entry，用 JSONL 存盘；每条带 id/seq/parentId 形成链。
CLI 启动时回放历史，每轮只把新消息追加写盘。

*/
