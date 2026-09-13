# Pi Agent Rust 重写进度总结

> 更新日期：2026-09-13
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
│   │   ├── data/models.json
│   │   └── src/...
│   ├── pi-agent-core/
│   ├── pi-tui/
│   └── pi-coding-agent/
└── docs/
```

验证命令与当前结果：

```text
cargo test --workspace                   通过，102 个测试成功
cargo fmt --all --check                  通过
cargo clippy --workspace --all-targets   通过（0 warning）
```

## 3. 四个 crate 的当前状态

| crate | 职责 | 当前状态 |
| --- | --- | --- |
| `pi-ai` | 模型、消息、上下文、流式事件、Provider、OpenAI 协议、兼容层、模型注册表 | 数据类型完备；有 FauxProvider 和真实 `OpenAiCompletionsProvider`；compat 探测/解析与请求接线基本完成；内置模型目录 |
| `pi-agent-core` | Agent 运行时、工具、执行环境、事件流、会话、压缩 | 工具循环、JSONL 会话持久化、上下文压缩已接入 Agent |
| `pi-tui` | 终端渲染 | 仍只有 `Renderer` trait + `PlainRenderer` 骨架 |
| `pi-coding-agent` | `pi` 可执行程序 | 真实 Provider、交互式 REPL、会话持久化、自动压缩、模型注册表；无正式参数解析和 TUI |

## 4. `pi-ai` 内容块（`content.rs`）

- `TextContent` / `ThinkingContent` / `ImageContent` / `ToolCall`：文本、思考、图片、工具调用。
- 按来源限制内容：`UserContent`（Text/Image）、`AssistantContent`（Text/Thinking/ToolCall）、`ToolResultContent`（Text/Image）。
- JSON 用 `type` 区分内容块（`#[serde(tag = "type")]`），`ToolCall` → `toolCall`，字段 camelCase，`None` 省略。

## 5. `pi-ai` 消息类型（`message.rs`）

- `UserMessage`：`UserRole::User` + `UserMessageContent`（`Text` 或 `Blocks`，`untagged`）+ 时间戳。
- `AssistantMessage`：`Vec<AssistantContent>`、api/provider/model、response model/id、诊断、`Usage`/`UsageCost`、`StopReason`、`DeferredHandle`、错误信息、`raw_stop_reason`、`end_turn`、时间戳。
- `ToolResultMessage<TDetails = Value>`：`tool_call_id`、`tool_name`、文本/图片、泛型 details、usage、`added_tool_names`、`is_error`、时间戳。
- `ConversationMessage`：`User` / `Assistant(Box<..>)` / `ToolResult(Box<..>)`，实现 `From`。
- `StopReason`：Pending / Stop / Length / ToolUse / Error / Aborted / Deferred。

## 6. `pi-ai` 上下文与工具（`context.rs`）

- `Tool<TParameters = Value>`、`Context`（system_prompt / messages / tools）。
- `ConstrainedSamplingConfig`：`json_schema`（`prefer`/`require`）或 `grammar`。

## 7. `pi-ai` 流式事件（`event.rs`）

`AssistantMessageEvent`（`#[serde(tag = "type")]`）：`start`、text/thinking/toolcall 的 start/delta/end、`done`、`error`。除 `done`/`error` 外都带 `partial`，用 `partial()` 统一取出。

## 8. `pi-ai` 图片、成本、兼容配置、模型目录、注册表

- `images.rs`：图片生成类型。
- `cost.rs`：`ModelCostRates` / `ModelCostTier` / `ModelCost`，用组合 + `#[serde(flatten)]` 模拟 `extends`。
- `compat.rs`：Provider 兼容配置与路由偏好（`OpenAICompletionsCompat`、`OpenAIResponsesCompat`、`AnthropicMessagesCompat`、`BedrockCompat`、`OpenRouterRouting`、`VercelGatewayRouting`、`ThinkingFormat`、`SessionAffinityFormat`、`CacheControlFormat` 等）。
- `model.rs`：`Model`、`ImagesModel`、`ModelCompat`、`ThinkingLevel`、`ThinkingLevelMap`、`InputType`；`calculate_cost`（价格表/分档/1h 缓存写）。
- `models.rs`：`ModelRegistry`（见第 10.8 节）。

## 9. `pi-ai` Provider 抽象与请求选项（`provider.rs`）

```rust
pub trait Provider {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn get_models(&self) -> &[Model];
    fn stream(&self, model: &Model, context: &Context, options: &RequestOptions)
        -> Box<dyn Iterator<Item = AssistantMessageEvent>>;
}
```

- `RequestOptions`：每次请求的可选参数。
  - `temperature: Option<f64>`
  - `max_tokens: Option<u64>`（覆盖 `Model.max_tokens`）
  - `reasoning_effort: Option<ModelThinkingLevel>`
  - `session_id: Option<String>`（会话亲和）
  - `cache_retention: Option<CacheRetention>`
- `CacheRetention`：`None` / `Short`（默认）/ `Long`。
- `FauxProvider`：不联网、按脚本返回 `Text`/`ToolCall`，脚本队列用 `RefCell<VecDeque<_>>`。

## 10. `pi-ai` API 协议层（`api/`）

目录：`mod.rs`、`openai_completions.rs`、`openai_compat.rs`、`http.rs`、`http_ureq.rs`。

### 10.1 请求方向（`openai_completions.rs`）

`build_request(model, context, options)` 把 `Model` + `Context` + `RequestOptions` 翻译成 `ChatRequest`：

- `ChatMessage`（`#[serde(tag = "role")]`）：`system` / `developer` / `user` / `assistant` / `tool`。
- 正文类型：user 用 `ChatUserContent`（字符串或含图片的 parts）；system/developer/assistant/tool 用 `ChatTextContent`（字符串或 `ChatTextPart`，后者可挂 `cache_control`）。
- `ChatToolCall.function.arguments` 是 **JSON 字符串**。
- compat 驱动的字段：`max_tokens` vs `max_completion_tokens`、`store`、`developer` 角色、工具 `strict`、`reasoning_effort`/`thinking`/`reasoning`、`provider`/`providerOptions` 路由、`prompt_cache_key`/`prompt_cache_retention`、`cache_control` 标记。
- 转换函数：`convert_messages`、`convert_user`、`convert_assistant`、`convert_tool_result`、`convert_tool`。

### 10.2 响应方向

- `ChatCompletionChunk` → `ChunkChoice` → `ChunkDelta` / `ChunkToolCall` / `ChunkFunction`；字段都 `#[serde(default)]`（流式 chunk 残缺）。
- `ChunkUsage` / `PromptTokensDetails` / `CompletionTokensDetails`。
- `map_stop_reason`（`finish_reason` → `StopReason`）、`parse_usage`（输入 = prompt − 缓存，含多种缓存字段兜底，最后 `calculate_cost`）。

### 10.3 流式聚合

- 内部 `enum Block { Text, Thinking, ToolCall }` 累积碎片。
- `ChatCompletionStream`：`handle_chunk` 记录元信息并分派 `handle_text` / `handle_thinking` / `handle_tool_calls`；工具调用按 `index` 分组、参数字符串先累加；`finish` 发 `*_end`（此处解析参数）并推断 stop reason，最后 `done`/`error`；`snapshot` 克隆出完整消息。
- `aggregate_sse(model, body)`：逐行取 `data:` 后的 JSON，遇 `[DONE]` 停止。

### 10.4 兼容层（`openai_compat.rs`）

- `ResolvedOpenAICompletionsCompat`：一组确定值（`bool`/枚举），对应原版 `ResolvedOpenAICompletionsCompat`。
- `detect_openai_completions_compat(model)`：按 `provider` / `base_url` 识别 zai、together、moonshot、openrouter、cloudflare、nvidia、ant-ling、deepseek 等，给出默认能力。
- `resolve_openai_completions_compat(model)`：先探测，再用 `model.compat`（`ModelCompat::OpenaiCompletions`）的显式值覆盖。
- 已接线的字段：`max_tokens_field`、`supports_store`、`supports_usage_in_streaming`、`supports_developer_role`、`requires_tool_result_name`、`requires_assistant_after_tool_result`、`requires_reasoning_content_on_assistant_messages`、`thinking_format`、`supports_strict_mode`、`supports_long_cache_retention`、`cache_control_format`、`thinking_token_budget_field`、`supports_reasoning_effort`、`session_affinity_format`、`send_session_affinity_headers`、`open_router_routing`、`vercel_gateway_routing`。

### 10.5 HTTP 传输（`http.rs` / `http_ureq.rs`）

- `HttpTransport` trait：`post(&HttpRequest) -> Result<String, HttpError>`。
- `UreqTransport`：`ureq` 阻塞实现，连接池 + 120 秒整体超时（rustls + gzip）。
- 与 `Environment`、`Provider` 同一套「trait + 真实实现 + 测试假实现」设计。

### 10.6 OpenAiCompletionsProvider

- 持有 `Box<dyn HttpTransport>` 和 API Key。
- `build_http_request`：拼 URL，头 `Content-Type` / `Authorization`；有 `session_id` 且开启 `send_session_affinity_headers` 时按格式加亲和头（`x-session-id` / `session_id` / `x-client-request-id` / `x-session-affinity`）。
- `run`：构造请求 → 传输 → `aggregate_sse`；失败收敛成一个 `error` 事件。
- 实现 `Provider`，可替换 `FauxProvider`。

已知限制：先读完整个响应体再聚合，还不是真正的「边收边解析」流式。

### 10.7 请求选项与缓存

- `temperature`、`max_tokens`、`reasoning_effort`（按 `thinking_format` 生成 OpenAI 风格 `reasoning_effort` / DeepSeek `thinking` / OpenRouter `reasoning`）。
- OpenAI 提示缓存：`prompt_cache_key`（session id，截断 64 字符）、`prompt_cache_retention`（长缓存 `"24h"`）。
- Anthropic 提示缓存：`cache_control_format == anthropic` 时，给系统提示、最后一个工具、最后一条对话消息打 `{"type":"ephemeral"}`（长缓存带 `ttl:"1h"`）。

### 10.8 模型注册表（`models.rs`）

- 数据形状 `{ api: { modelId: Model } }`，内置数据 `data/models.json` 用 `include_str!` 编译期内嵌。
- `ModelRegistry::from_json(text)`：解析并按 `(provider, id)` 排序。
- `ModelRegistry::builtin()`：用内置数据构造。
- `get(provider, id)` / `models()` / `providers()` / `models_for_provider(provider)`。

## 11. `pi-agent-core` Agent 运行时（`lib.rs`）

`Agent` 持有 `Box<dyn Provider>`、`Model`、`system_prompt`、`Vec<Box<dyn AgentTool>>`、`Vec<ConversationMessage>`、`RequestOptions`。

- `with_system_prompt` / `with_options`：链式设置（builder）。
- `run_once`：组装 `Context` → `provider.stream(&self.model, &context, &self.options)` → 收集事件 → 追加最终助手消息；缺 `done`/`error` 返回 `AgentError::IncompleteStream`。
- `run` / `run_loop`：多轮循环，最多 `MAX_TURNS = 32`。
- `execute_tool_calls` / `execute_one`：执行工具并追加结果；未知/失败返回 `is_error = true` 的结果。
- `maybe_compact`：超阈值时用 Provider 生成摘要，把消息替换成「摘要 + 近期消息」。

## 12. `pi-agent-core` 工具系统

- `AgentTool`：`name` / `description` / `parameters` / `execute`。
- `ToolResult`：content / details / usage / added_tool_names / terminate；`EchoTool` 示例。
- 内置 `read`（offset/limit + 截断）、`write`、`edit`（唯一且不重叠的精确替换）。
- 通过 `Environment` 访问文件系统。

## 13. `pi-agent-core` 执行环境（`environment.rs`）

`Environment`（`read_file`/`write_file`）；`RealEnvironment` 真实实现，测试用内存 `FakeEnv`。

## 14. `pi-agent-core` 事件流（`event.rs`）

`AgentEvent`：run 级（`AgentStart`/`AgentEnd`）、turn 级（`TurnStart`/`TurnEnd`）、消息与工具级（`MessageStart`/`MessageUpdate`/`MessageEnd`、`ToolExecutionStart`/`ToolExecutionUpdate`/`ToolExecutionEnd`）。较大字段用 `Box`。

## 15. `pi-agent-core` 会话持久化（`session.rs`）

- `Entry`（`#[serde(tag = "type", rename_all = "snake_case")]`）：`Message` / `Custom`，都带 id / seq / `parentId` / timestamp。
- `Session`：只追加，`parent_id` 串链、`seq` 递增；`push_message` / `push_custom` / `messages` / `to_jsonl` / `from_jsonl` / `save` / `load`。

## 16. `pi-agent-core` 上下文压缩（`compaction.rs`）

- `CompactionSettings`（`reserve_tokens` 16384、`keep_recent_tokens` 20000）。
- `estimate_tokens` / `estimate_context_tokens`（优先用最近有效 assistant 的实测 usage）。
- `should_compact` / `plan_compaction`（尾部保留、至少一条）。
- `SUMMARIZATION_SYSTEM_PROMPT` / `SUMMARIZATION_PROMPT` / `build_summary_prompt`。
- `compact(messages, settings, summarize)` → `CompactionResult { summary, retained_tail, tokens_before }`。

## 17. `pi-tui` 与 `pi-coding-agent`

- `pi-tui`：仍只有 `Renderer` + `PlainRenderer` 骨架。
- `pi-coding-agent`（`src/main.rs`）：
  - `build_agent`：用 `ModelRegistry::builtin()`，按 `PI_PROVIDER` / `PI_MODEL`（默认 `openai`/`gpt-4o-mini`）选模型；有 `OPENAI_API_KEY` 且 api 为 `openai-completions` 时用 `OpenAiCompletionsProvider`，否则回退 Faux 演示。
  - 注册 `ReadTool` / `WriteTool` / `EditTool`。
  - `Cli`：组合 Agent、`Session`、`RealEnvironment`、会话文件路径（`PI_SESSION` 或 `pi-session.jsonl`）；用会话路径作 session id（`with_options`）。
  - `run_turn`：先 `compact_if_needed`，再加用户消息，`agent.run` 打印 `TextDelta` 与工具事件，最后 `persist`。
  - `repl`：逐行读取，`exit`/`quit` 或 Ctrl-D 退出；带参数时一次性提问。
- 还没有正式参数解析，也没有基于 `AgentEvent` 的 TUI 渲染。

## 18. 已完成的测试

`cargo test --workspace` 共 102 个测试，全部通过。

`pi-ai`（71 个）：

- `tests/content.rs`（2）、`tests/context.rs`（3）、`tests/event.rs`（3）、`tests/images.rs`（2）、`tests/message.rs`（11）、`tests/model.rs`（3）、`tests/provider.rs`（2）。
- `tests/models.rs`（2）：注册表加载/查找、内置目录解析。
- `tests/openai_completions.rs`（43）：请求形状与 compat 字段、消息/工具转换、chunk 反序列化、stop reason、usage/费用、thinking 字段、会话亲和头、prompt cache、cache_control、SSE 聚合与 Provider。

`pi-agent-core`（31 个）：

- `src/lib.rs` 内联（8）：单轮流式、空流报错、工具执行、`run` 循环、turn/message 事件、最大轮数、对话往返。
- `tests/tools.rs`（9）、`tests/session.rs`（5）、`tests/compaction.rs`（9）。

## 19. 已学习和使用的 Rust 概念

- Cargo workspace、crate、模块系统（`mod` / `use` / `pub use` / `crate::` / `super::`）。
- `struct`、携带数据的 `enum`、`match` 与模式绑定；在 enum 上写方法。
- `Option<T>`、`Vec<T>`、`HashMap<K, V>`、`BTreeSet<T>`、泛型（默认类型参数 `<T = Value>`）。
- `String` 的所有权、移动、`clone`、`clone_from`、`std::mem::take`、借用。
- trait、`impl`、派生宏、trait object（`Box<dyn Trait>`）与 `dyn` 分发。
- 属性 `#[...]`、Serde 的 `tag` / `untagged` / `flatten` / `rename` / `rename_all` / `default` / `skip_serializing_if`。
- `Result` 与 `?`、`map_err`、`ok_or_else`；自定义错误类型 + `Display`。
- `let ... else`、`Option::is_some_and`、`bool::then` / `then_some`、`Option::flatten`、`matches!`。
- 枚举 `#[default]` + `unwrap_or_default`。
- `Box` 装箱（装箱迭代器、trait object、缩小枚举变体）。
- 内部可变性 `RefCell`、`Rc`。
- 迭代器（`into_iter` / `into_values` / `rev` / `enumerate` / `flatten` / `flat_map` / `next`）、`sort_by`、`windows`、`saturating_sub`、`div_ceil`、`with_capacity`。
- 有状态聚合器（`ChatCompletionStream` 状态机）。
- 闭包作为参数（`impl FnOnce`）实现依赖注入（摘要器）。
- `include_str!`、类型别名、编译期内嵌数据。
- 用类型排除非法状态（角色枚举、按来源限内容、泛型 details）。

## 20. 当前代码状态与提交里程碑

- 工作区 `cargo test --workspace`（102 个）全绿；`cargo fmt --all --check` 与 `cargo clippy --workspace --all-targets` 均通过。
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
75801cdf9 新增 OpenAI-compatible 兼容层（探测与解析）
c009d7c2d 接入 compat 的路由、工具 strict 与 assistant reasoning_content
f5a59af0d 新增 RequestOptions 并接入 temperature 与 max_tokens
24262f85e 接入 reasoning_effort 与各厂商 thinking 字段
8050a113a 新增 session_id 并发送会话亲和请求头
fafa5601e 接入提示缓存（OpenAI prompt cache 与 Anthropic cache_control）
5dfa387ee 新增模型注册表并从数据加载模型目录
```

## 21. 建议的下一步

1. 工具与 Agent 能力补齐：`glob` / `grep` / `bash` / `powershell`、工具列表管理、终止条件。
2. 真正的流式传输：把 `HttpTransport::post`（返回整段 body）换成流式 reader，边收边聚合。
3. 适配原版完整模型目录：让 `ModelCompat` 接受原版 `compat` 的扁平形状，处理 `thinkingLevelMap` 等，接入 `models.generated` 数据。
4. 剩余 compat：`thinking_format` 的 zai / qwen / together / ant-ling / baseten / chat-template 分支、思考 token 预算。
5. 会话系统增强：分支、恢复、多会话（对齐原版 harness）。
6. `pi-tui` 与 CLI：正式参数解析、基于 `AgentEvent` 的增量渲染、权限/审批。
7. 更多 Provider 适配器：anthropic-messages、openai-responses、google 等。
