# pi / pi-rust

仓库根下有两个同级项目：

- [`pi/`](pi) —— 原版 **Pi agent harness**（TypeScript monorepo）：`pi-coding-agent`（交互式编码 agent CLI）、`pi-agent-core`（agent 运行时）、`pi-ai`（多 provider LLM API）等。
- [`pi-rust/`](pi-rust) —— 用 Rust 从零重写的 pi-agent，保持与原版一致的 JSON wire format，用于理解这套 agent 框架的运行原理。

## 常用命令

| 目录 | 依赖安装 | 检查 / 测试 |
|---|---|---|
| `pi/` | `cd pi && npm install --ignore-scripts` | `cd pi && npm run check`、`./pi/test.sh` |
| `pi-rust/` | 无需安装 | `cd pi-rust && cargo test` |

## 其它

- 原版项目的详细说明见 [`pi/README.md`](pi/README.md)。
- CI 与发布工作流保留在仓库根的 [`.github/workflows/`](.github/workflows)，通过 `working-directory: pi` 指向原版项目。
- 仓库级协作约定见 [`AGENTS.md`](AGENTS.md)。