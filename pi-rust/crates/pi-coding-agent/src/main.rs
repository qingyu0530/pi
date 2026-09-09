use std::io;

use pi_agent_core::Agent;
use pi_ai::{UserMessage, UserMessageContent, UserRole};
use pi_tui::{PlainRenderer, Renderer};

fn main() -> io::Result<()> {
    let mut agent = Agent::new();
    agent.add_message(UserMessage {
        role: UserRole::User,
        content: UserMessageContent::Text("Hello, Pi".to_owned()),
        timestamp: 0,
    });

    let mut renderer = PlainRenderer;
    renderer.render_line(&format!(
        "Pi Rust {} ({} message initialized)",
        env!("CARGO_PKG_VERSION"),
        agent.messages().len()
    ))
}
