use std::io;

use pi_agent_core::{Agent, EchoTool};
use pi_ai::{
    FauxProvider, FauxResponse, InputType, Model, ModelCost, ModelCostRates, UserMessage,
    UserMessageContent, UserRole,
};
use pi_tui::{PlainRenderer, Renderer};

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

fn main() -> io::Result<()> {
    let model = faux_model();
    // 脚本：先发起一次 echo 工具调用，再回一段最终文本。
    let arguments = serde_json::json!({ "text": "Hello from echo tool" })
        .as_object()
        .unwrap()
        .clone();
    let provider = FauxProvider::with_script(
        vec![model.clone()],
        vec![
            FauxResponse::tool_call("call_1", "echo", arguments),
            FauxResponse::Text("完成".to_owned()),
        ],
    );

    let mut agent = Agent::new(Box::new(provider), model);
    agent.add_tool(Box::new(EchoTool));
    agent.add_message(UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Text("请调用 echo 工具".to_owned()),
        timestamp: 0,
    });

    let reply = agent
        .run()
        .map_err(|error| io::Error::other(format!("{error:?}")))?;

    let mut renderer = PlainRenderer;
    renderer.render_line(&format!(
        "Pi Rust {} ({} messages, final stop reason: {:?})",
        env!("CARGO_PKG_VERSION"),
        agent.messages().len(),
        reply.stop_reason
    ))
}
