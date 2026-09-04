use std::io;

use pi_agent_core::Agent;
use pi_ai::Role;
use pi_tui::{PlainRenderer, Renderer};

fn main() -> io::Result<()> {
    let mut agent = Agent::new();
    agent.add_message(Role::System, "You are Pi, a coding agent.");

    let mut renderer = PlainRenderer;
    renderer.render_line(&format!(
        "Pi Rust {} ({} message initialized)",
        env!("CARGO_PKG_VERSION"),
        agent.messages().len()
    ))
}
