# Pi Agent Rust 重写进度总结

> 更新日期：2026-09-11
> Rust workspace：`E:\vscode\pi\pi-rust`
>
> 说明：本文件按当前代码的真实状态维护。早期细节可参考同目录的 `学习总结.md` 和 `pi-agent-rust-rewrite-conversation.md`，本文档为准。

## 1. 项目目标

使用 Rust 重写 Pi Agent，同时通过实际项目学习 Rust。

重写不是把 TypeScript 逐行翻译成 Rust，而是保持原项目的职责划分和外部数据协议，再使用 Rust 的类型系统重新表达这些设计。讲解时主要使用 C++ 的相近概念进行对照。

原 TypeScript 项目的主要分层：

- `packages/ai`：统一的模型、消息、Provider、流式响应和工具调用协议。
- `packages/agent`：Agent 状态、对话循环、工具执行和事件流。
- `packages/tui`：终端界面和渲染组件。
- `packages/coding-agent`：最终的 `pi` 命令行程序。

Rust workspace 按相同职责建立四个 crate：

- `pi-ai`
- `pi-agent-core`
- `pi-tui`
- `pi-coding-agent`

## 2. Rust 工程环境

已完成的基础配置（`Cargo.toml`）：

- Cargo workspace，`resolver = "3"`。
- Rust 2024 edition，最低版本 `rust-version = "1.85"`。
- stable 工具链包含 `rustfmt` 和 Clippy。
- workspace 禁止 `unsafe`（`[workspace.lints.rust] unsafe_code = "forbid"`）。
- Clippy 打开 `all = "warn"`。
- 统一管理依赖：
  - `serde = "=1.0.229"`（derive）
  - `serde_json = "=1.0.151"`
  - `ureq = "=3.4.1"`（`default-features = false`，features `rustls` + `gzip`）
- 保留 `Cargo.lock`；`target` 目录被 `pi-rust/.gitignore` 忽略。

```text
pi-rust/
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml
├── rustfmt.toml
├── crates/
│   ├── pi-ai/
│   ├── pi-agent-core/
│   ├── pi-tui/
│   └── pi-coding-agent/
└── docs/
```

验证命令与当前结果：

```text
cargo test --workspace                   通过，77 个测试成功
cargo fmt --all --check                  通过
cargo clippy --workspace --all-targets   通过（0 warning）
```

## 3. 四个 crate 的当前状态

| crate | 职责 | 当前状态 |
| --- | --- | --- |
| `pi-ai` | 模型、消息、上下文、流式事件、Provider、OpenAI 协议与 HTTP | 数据类型完备；有 FauxProvider 和真实 `OpenAiCompletionsProvider`（ureq 传输，SSE 聚合） |
| `pi-agent-core` | Agent 运行时、工具、执行环境、事件流、会话、压缩 | 工具循环、JSONL 会话持久化、上下文压缩已接入 Agent |
| `pi-tui` | 终端渲染 | 仍只有 `Renderer` trait + `PlainRenderer` 骨架 |
| `pi-coding-agent` | `pi` 可执行程序 | 有真实 Provider 接入、交互式 REPL、会话持久化、自动压缩；无正式参数解析和 TUI |

## 4. `pi-ai` 内容块（`content.rs`）

### 4.1 基础内容结构体

- `TextContent`：文本和可选文本签名。
- `ThinkingContent`：模型推理内容、可选签名和删减标记。
- `ImageContent`：Base64 图片数据和 MIME 类型。
- `ToolCall`：工具调用 ID、名称、JSON 参数对象，以及可选的 `thought_signature`、`namespace`。

### 4.2 按消息来源限制内容类型

```text
UserContent          AssistantContent        ToolResultContent
├── Text             ├── Text                ├── Text
└── Image            ├── Thinking            └── Image
                     └── ToolCall
```

对应 C++ 的 `std::variant`，在编译阶段排除非法组合：用户消息不能包含工具调用，工具结果不能包含思考内容。

### 4.3 JSON 协议

- 用 `type` 字段区分内容块（`#[serde(tag = "type")]`）。
- `ToolCall` 变体在 JSON 中表示为 `toolCall`。
- snake_case 字段转 JSON camelCase；`None` 的可选字段省略。

## 5. `pi-ai` 消息类型（`message.rs`）

- `UserMessage`：固定角色 `UserRole::User`、正文 `UserMessageContent`（`Text` 或 `Blocks`，`#[serde(untagged)]`）、Unix 毫秒时间戳。
- `AssistantMessage`：角色、`Vec<AssistantContent>`、api/provider/model、response model/id、诊断、`Usage`/`UsageCost`、`StopReason`、`DeferredHandle`、错误信息、原始停止原因、`end_turn`、时间戳。
- `ToolResultMessage<TDetails = Value>`：`tool_call_id`、`tool_name`、文本/图片内容、泛型 details、可选 usage 与 `added_tool_names`、`is_error`、时间戳。
- `ConversationMessage`：`User` / `Assistant(Box<..>)` / `ToolResult(Box<..>)`，并实现三种消息的 `From`。
- `StopReason`：Pending / Stop / Length / ToolUse / Error / Aborted / Deferred。

## 6. `pi-ai` 上下文与工具（`context.rs`）

- `Tool<TParameters = Value>`：发给模型的工具定义。
- `Context`：一次调用的完整输入（system_prompt / messages / tools）。
- `ConstrainedSamplingConfig`：`json_schema`（`prefer`/`require`）或 `grammar`（`GrammarVariants`）。

## 7. `pi-ai` 流式事件（`event.rs`）

`AssistantMessageEvent` 用 `#[serde(tag = "type")]` 表达 12 种事件：

- `start`
- text：`text_start` / `text_delta` / `text_end`
- thinking：`thinking_start` / `thinking_delta` / `thinking_end`
- toolcall：`toolcall_start` / `toolcall_delta` / `toolcall_end`
- `done`（携带最终消息） / `error`（携带错误消息）

除 `done` / `error` 外都携带 `partial`（已组装的部分助手消息），用 `partial()` 统一取出。

## 8. `pi-ai` 图片、成本、兼容配置、模型目录

- `images.rs`：图片生成类型（`ImagesContext`、`ImagesStopReason`、`AssistantImages`）。
- `cost.rs`：`ModelCostRates`、`ModelCostTier`、`ModelCost`。原版用 `extends`，Rust 用组合 + `#[serde(flatten)]` 平铺字段。
- `compat.rs`：Provider 兼容配置与路由偏好（OpenAI/Anthropic/Bedrock 兼容、OpenRouter/Vercel 路由等）。
- `model.rs`：`Model`、`ImagesModel`、`ModelCompat`、`ThinkingLevel`、`ThinkingLevelMap`、`InputType`；以及 `calculate_cost`（按价格表/分档/1h 缓存写计费）。

## 9. `pi-ai` Provider 抽象与 FauxProvider（`provider.rs`）

```rust
pub trait Provider {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn get_models(&self) -> &[Model];
    fn stream(&self, model: &Model, context: &Context)
        -> Box<dyn Iterator<Item = AssistantMessageEvent>>;
}
```

- Provider 把 `Model` + `Context` 变成一串流式事件；`stream` 返回装箱迭代器，调用方不关心底层来源。
- `FauxProvider`：不联网、结果可控，按脚本返回 `Text`/`ToolCall`；脚本队列用 `RefCell<VecDeque<_>>` 做内部可变性。

## 10. `pi-ai` API 协议层（`api/`）

目录：`api/mod.rs`、`api/openai_completions.rs`、`api/http.rs`、`api/http_ureq.rs`。

### 10.1 请求方向（`openai_completions.rs:33` 起）

把 `Model` + `Context` 翻译成 `POST /chat/completions` 请求体：

- `ChatRequest`：`model` / `messages` / `stream` / `stream_options.include_usage` / `max_completion_tokens` / `temperature` / `tools`。
- `ChatMessage`（`#[serde(tag = "role")]`）：`system` / `user` / `assistant` / `tool` 四种形状。
- `ChatUserContent`（`untagged`）：纯字符串或 `ChatUserPart` 数组；图片用 `data:<mime>;base64,<data>`。
- `ChatToolCall`：`function.arguments` 是 **JSON 字符串**（协议要求）。
- `ChatTool` / `ChatFunctionDef`：`{type:"function", function:{name, description, parameters}}`。
- 转换函数：`build_request`、`convert_messages`、`convert_user`、`convert_assistant`、`convert_tool_result`、`convert_tool`。

### 10.2 响应方向（`openai_completions.rs:307` 起）

- `ChatCompletionChunk` → `ChunkChoice` → `ChunkDelta` / `ChunkToolCall` / `ChunkFunction`；每个字段都 `#[serde(default)]`，因为流式 chunk 是残缺 JSON。
- `ChunkUsage` / `PromptTokensDetails` / `CompletionTokensDetails`：token 用量。
- `map_stop_reason(Option<&str>)`：`finish_reason` → `StopReason`（`stop`/`length`/`tool_calls`/`content_filter` 等）。
- `parse_usage`：`input = prompt_tokens - cache_read - cache_write`，`output = completion_tokens`，缓存读按 `cached_tokens`/`prompt_cache_hit_tokens`/顶层 `cached_tokens` 兜底；最后调 `calculate_cost`。

### 10.3 流式聚合（`openai_completions.rs:507` 起）

- 内部 `enum Block { Text, Thinking, ToolCall }` 累积碎片。
- `ChatCompletionStream`：有状态聚合器。
  - `handle_chunk`：记录 response id / 实际模型 / usage，处理 `finish_reason`，把 `delta` 分派给文本/思考/工具处理器。
  - `handle_text` / `handle_thinking` / `handle_tool_calls`：按需新建块、发 `*_start`/`*_delta`；`tool_calls` 按 chunk 的 `index` 用 `HashMap<u32, usize>` 分组，参数字符串先累加。
  - `finish`：为每块发 `*_end`（工具调用在这里才把参数字符串解析成对象），推断 stop reason，最后发 `done` / `error`。
  - `snapshot`：把当前累积状态克隆成完整 `AssistantMessage`。
- `aggregate_sse(model, body)`：逐行取 `data:` 后的 JSON，喂给聚合器；遇到 `[DONE]` 停止。

### 10.4 HTTP 传输抽象（`http.rs` / `http_ureq.rs`）

```rust
pub trait HttpTransport {
    fn post(&self, request: &HttpRequest) -> Result<String, HttpError>;
}
```

- `HttpRequest`：url / headers / body。
- `UreqTransport`：基于 `ureq` 的阻塞实现，内部持有 `Agent`（连接池 + 120 秒整体超时，rustls TLS + gzip）。
- 与 `Environment`、`Provider` 同一套「trait + 真实实现 + 测试假实现」的设计。

### 10.5 OpenAiCompletionsProvider（`openai_completions.rs:955` 起）

- 持有 `Box<dyn HttpTransport>` 和 API Key。
- `build_http_request`：`base_url` + `/chat/completions`，头 `Content-Type: application/json` 与 `Authorization: Bearer <key>`。
- `run`：构造请求 → 传输 → `aggregate_sse`；任何失败都收敛成一个 `error` 事件。
- 实现了 `Provider`，因此可以直接替换 `FauxProvider` 接进 `Agent`。

已知限制：当前先把整个响应体读完再聚合，还不是真正的「边收边解析」流式；后续可把 `HttpTransport::post` 换成返回流式 reader。

## 11. `pi-agent-core` Agent 运行时（`lib.rs`）

`Agent` 持有 `Box<dyn Provider>`、`Model`、`system_prompt`、`Vec<Box<dyn AgentTool>>`、`Vec<ConversationMessage>`。

- `run_once`：单轮。组装 `Context` → 调 `provider.stream` → 收集事件 → 追加并返回最终助手消息；没有 `done`/`error` 时返回 `AgentError::IncompleteStream`。
- `run` / `run_loop`：多轮循环，直到助手不再发起工具调用；最多 `MAX_TURNS = 32` 轮。
- `execute_tool_calls` / `execute_one`：执行工具并追加工具结果消息；未知/失败的工具返回 `is_error = true` 的结果，不中断流程。
- `maybe_compact`：上下文超阈值时调用 `compact`，用 Provider 生成摘要，把消息列表替换成「一条摘要用户消息 + 保留的近期消息」，返回 `CompactionResult`。

## 12. `pi-agent-core` 工具系统

- `AgentTool` trait：`name` / `description` / `parameters`（JSON Schema）/ `execute`。
- `ToolResult`：content / details / usage / added_tool_names / terminate；`EchoTool` 为示例。
- 内置工具 `read` / `write` / `edit`：
  - `read`：`offset`（1 起）+ `limit`，按 2000 行 / 50KB 截断并附继续读取提示。
  - `write`：覆盖写入。
  - `edit`：按 `edits[]`（`oldText`/`newText`）精确替换，要求唯一且不重叠，从后往前替换。
- 工具通过 `Environment` 访问文件系统，不直接调用 `std::fs`。

## 13. `pi-agent-core` 执行环境（`environment.rs`）

`Environment` trait（`read_file` / `write_file`）；`RealEnvironment` 用真实文件系统，测试用 `FakeEnv`（内存 `HashMap` + `Rc<RefCell<..>>`）替换。

## 14. `pi-agent-core` 事件流（`event.rs`）

`AgentEvent` 分三层：

- run 级：`AgentStart` / `AgentEnd { messages }`。
- turn 级：`TurnStart` / `TurnEnd { message, tool_results }`。
- 消息与工具级：`MessageStart` / `MessageUpdate { message, event }` / `MessageEnd`，`ToolExecutionStart` / `ToolExecutionUpdate`（暂未发出）/ `ToolExecutionEnd`。

较大的字段统一用 `Box` 装箱。

## 15. `pi-agent-core` 会话持久化（`session.rs`）

对原版会话系统（entries / records / lanes / branches / forks）的最小实现：

- `Entry`（`#[serde(tag = "type", rename_all = "snake_case")]`）：
  - `Message`：id / seq / `parentId` / timestamp / `ConversationMessage`。
  - `Custom`：id / seq / `parentId` / timestamp / `customType` / 可选 `data`。
- `Session`：只追加的记录集合，`parent_id` 串成链、`seq` 递增。
  - `push_message` / `push_custom` 自动填 id / seq / parentId / timestamp。
  - `messages()` 提取所有消息（忽略自定义记录）。
  - `to_jsonl` / `from_jsonl`：每行一条 JSON；加载后恢复 `next_seq` 与 `last_id`。
  - `save` / `load`：通过 `Environment` 读写文件。

## 16. `pi-agent-core` 上下文压缩（`compaction.rs`）

会话过长时把较早的消息摘要成一段总结，只保留最近一段。摘要器由调用方注入，便于测试或接真实模型。

- `CompactionSettings`：`enabled` / `reserve_tokens`（默认 16384）/ `keep_recent_tokens`（默认 20000）。
- `estimate_tokens`：按字符数 / 4 估算（至少 1；图片折算 4800 字符）。
- `estimate_context_tokens`：优先用最近一条有效 assistant 的实测 `usage`，其后的消息再用启发式。
- `should_compact`：`context_tokens > context_window - reserve_tokens`。
- `plan_compaction`：从尾部往前保留到 `keep_recent_tokens`，至少保留一条；其余归入 `to_summarize`。
- `SUMMARIZATION_SYSTEM_PROMPT` / `SUMMARIZATION_PROMPT`：要求输出结构化摘要（Goal / Progress / Next Steps / Critical Context 等）。
- `build_summary_prompt` / `render_message`：把消息渲染成 `role: 内容` 文本。
- `compact(messages, settings, summarize)`：切分 → 构造提示 → 调摘要器 → 返回 `CompactionResult { summary, retained_tail, tokens_before }`。

## 17. `pi-tui` 与 `pi-coding-agent`

- `pi-tui`：仍只有 `Renderer` trait（`render_line`）和 `PlainRenderer`，未实现输入、布局、增量渲染。
- `pi-coding-agent`（`src/main.rs`）：
  - `build_agent`：设置了 `OPENAI_API_KEY` 时用 `OpenAiCompletionsProvider`（`UreqTransport`），模型来自 `OPENAI_BASE_URL` / `OPENAI_MODEL`；否则回退 `FauxProvider` 演示。
  - 注册 `ReadTool` / `WriteTool` / `EditTool`（`RealEnvironment`）。
  - `Cli`：组合 Agent、`Session`、`RealEnvironment`、会话文件路径（`PI_SESSION` 或 `pi-session.jsonl`）。
  - `run_turn`：先 `compact_if_needed`，再加用户消息，`agent.run` 时打印 `TextDelta` 与工具完成事件，最后 `persist`。
  - `persist`：只追加未保存的新消息并写盘。
  - `repl`：逐行读取，`exit`/`quit` 或 Ctrl-D 退出。
  - `main`：带参数时一次性提问，否则进入 REPL。
- 还没有正式参数解析，也没有基于 `AgentEvent` 的 TUI 渲染。

## 18. 已完成的测试

`cargo test --workspace` 共 77 个测试，全部通过。

`pi-ai`（46 个）：

- `tests/content.rs`（2）：工具调用 JSON 形状、内容块序列化往返。
- `tests/message.rs`（11）：用户/助手/工具结果消息与非法消息拒绝、对话消息顺序。
- `tests/context.rs`（3）：工具/上下文、受约束采样、schema 严格度。
- `tests/event.rs`（3）：流式事件形状与 `partial()`。
- `tests/images.rs`（2）：图片生成上下文与结果。
- `tests/model.rs`（3）：成本档位平铺、compat、模型往返。
- `tests/provider.rs`（2）：FauxProvider 文本流与脚本化工具调用。
- `tests/openai_completions.rs`（20）：请求形状、消息/工具转换、chunk 反序列化、stop reason、usage/费用、SSE 聚合与 Provider。

`pi-agent-core`（31 个）：

- `src/lib.rs` 内联（8）：单轮流式、空流报错、工具执行、`run` 循环、turn/message 事件、最大轮数、对话往返。
- `tests/tools.rs`（9）：read / write / edit 的读写与错误路径。
- `tests/session.rs`（5）：会话追加、JSONL 往返、文件保存/加载。
- `tests/compaction.rs`（9）：token 估算、阈值判断、切分、摘要提示、`compact` 结果。

## 19. 已学习和使用的 Rust 概念

- Cargo workspace、crate、模块系统（`mod` / `use` / `pub use` / `crate::` / `super::`）。
- `struct`、携带数据的 `enum`、`match` 与模式绑定。
- `Option<T>`、`Vec<T>`、`HashMap<K, V>`、泛型（默认类型参数 `<T = Value>`）。
- `String` 的所有权、移动、`clone`、`clone_from`、借用。
- trait、`impl`、派生宏、trait object（`Box<dyn Trait>`）与 `dyn` 分发。
- 属性 `#[...]`、Serde 的 `tag` / `untagged` / `flatten` / `rename_all` / `default` / `skip_serializing_if`。
- `Result` 与 `?`、`map_err`、`ok_or_else`；用错误类型替代异常。
- `Box` 装箱（装箱迭代器、装箱 trait object、缩小枚举变体）。
- 内部可变性 `RefCell`、`Rc`；`std::cell` 与借用规则。
- 迭代器（`into_iter` / `rev` / `enumerate` / `flatten` / `next`、`Box<dyn Iterator>`）、`match_indices`、`windows`、`saturating_sub`、`div_ceil`、`with_capacity`。
- 有状态聚合器（`ChatCompletionStream` 状态机）。
- 闭包作为参数（`impl FnOnce`）实现依赖注入（摘要器）。
- 用类型排除非法状态（角色枚举、按来源限内容、泛型 details）。

## 20. 当前代码状态与提交里程碑

- 工作区 `cargo test --workspace`（77 个）全绿；`cargo fmt --all --check` 与 `cargo clippy --workspace --all-targets` 均通过。
- `docs/` 下另有两份历史文档：`学习总结.md`、`pi-agent-rust-rewrite-conversation.md`，内容针对早期阶段。

`pi-rust` 分支提交里程碑：

```text
ca8645708 初始化 Rust workspace 和基础 crate
3e8e50b97 内容块类型、序列化测试
b0db3c44e 用户和助手消息类型
e45a5ff2b 工具结果消息与完整对话消息
0d72c0177 工具、上下文和流式事件类型
a96cc6c25 图片生成类型
4514dcab9 模型成本、provider 兼容配置与模型目录
b93a798f3 内容总结
f8d8c3e34 重写对话与进度文档
858b0ccc0 Agent 接入完整 ConversationMessage 并移除旧消息模型
4daf99c5e Provider trait 与 FauxProvider
a58e06a62 Agent 接入 Provider 并实现工具调用循环
b155ef392 Agent 事件流与 turn 边界
e55725449 执行环境抽象与 read/write 工具
77ef4fb7e edit 工具（精确字符串替换）
1dcf549c2 新增 OpenAI-compatible Chat Completions 请求层
6f1fce41a 清理 edit.rs 的 rustfmt/clippy 告警
442b25fe6 更新 pi-agent-rust 重写进度
ac565c400 新增 OpenAI-compatible 响应解析与费用计算
b57f9abfe 新增 OpenAI-compatible 流式 chunk 聚合器
3052ba108 新增 OpenAI-compatible Provider 与 HTTP 传输抽象
4ce4dbaa6 接入真实 HTTP 传输并让 CLI 使用真实 Provider
eaf480c59 CLI 增加交互式输入循环（REPL）
980310ce8 新增会话持久化（JSONL）并接入 CLI
26c696b05 新增上下文压缩核心逻辑（compaction）
941c86fcb 将上下文压缩接入 Agent 与 CLI
```

## 21. 建议的下一步

1. 模型注册表 / 配置：模型目录（`models.generated`）、API Key 与 base URL 的来源、默认模型选择。
2. 真正的流式传输：把 `HttpTransport::post`（返回整段 body）换成流式 reader，边收边聚合。
3. 兼容层（compat）：`max_tokens` vs `max_completion_tokens`、`developer` role、各种 thinking 字段格式、cache control、OpenRouter 路由等。
4. 工具与 Agent 能力补齐：`glob` / `grep` / `bash` / `powershell`、工具列表管理、终止条件。
5. 会话系统增强：分支、恢复、多会话（对齐原版 harness）。
6. `pi-tui` 与 CLI：正式参数解析、基于 `AgentEvent` 的增量渲染、权限/审批。
7. 更多 Provider 适配器：anthropic-messages、openai-responses、google 等。
