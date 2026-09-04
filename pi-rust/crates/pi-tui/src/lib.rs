//! Terminal rendering primitives for Pi.

use std::io::{self, Write};

/// Renders text for an interactive frontend.
pub trait Renderer {
    fn render_line(&mut self, line: &str) -> io::Result<()>;
}

/// A renderer that writes plain text to standard output.
#[derive(Debug, Default)]
pub struct PlainRenderer;

impl Renderer for PlainRenderer {
    fn render_line(&mut self, line: &str) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        writeln!(stdout, "{line}")
    }
}
