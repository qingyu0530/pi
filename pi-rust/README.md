# Pi Rust

Rust rewrite of the Pi agent harness.

## Workspace

- `pi-ai`: model and conversation types
- `pi-agent-core`: agent state and conversation loop foundation
- `pi-tui`: terminal rendering abstractions
- `pi-coding-agent`: command-line application

## Development

```bash
cargo fmt --all --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```
