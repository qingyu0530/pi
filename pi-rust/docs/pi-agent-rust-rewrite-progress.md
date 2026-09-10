# Pi Agent Rust 重写进度总结

> 更新日期：2026-09-10
> Rust workspace：`E:\vscode\pi\pi-rust`
>
> 说明：本文件已按当前代码的真实状态重写。较早的版本停留在「pi-ai 消息类型、Agent 仍用简化 Message」的阶段，但代码此后已完成 Provider 抽象、工具系统、执行环境和 Agent 多轮循环。早期细节可参考同目录的 `学习总结.md` 和 `pi-agent-rust-rewrite-conversation.md`。

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
- 统一管理 `serde`（`=1.0.229`）和 `serde_json`（`=1.0.151`）。
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
cargo test --workspace                   通过，43 个测试成功
cargo fmt --all --check                  通过
cargo clippy --workspace --all-targets   通过（0 warning）
```

## 3. 四个 crate 的当前状态

| crate | 职责 | 当前状态 |
| --- | --- | --- |
| `pi-ai` | 模型、消息、上下文、流式事件、Provider 抽象 | 数据类型基本完备，含 FauxProvider |
| `pi-agent-core` | Agent 运行时、工具、执行环境、事件流 | 已能跑通「用户消息 → 工具调用 → 工具结果 → 再请求」完整循环 |
| `pi-tui` | 终端渲染 | 仅骨架：`Renderer` trait + `PlainRenderer` |
| `pi-coding-agent` | `pi` 可执行程序 | 已组装 Agent + FauxProvider 脚本，但无参数解析、交互循环、真实网络 |

## 4. `pi-ai` 内容块（`content.rs`）

### 4.1 基础内容结构体

- `TextContent`：文本和可选文本签名。
- `ThinkingContent`：模型推理内容、可选签名和删减标记。
- `ImageContent`：Base64 图片数据和 MIME 类型。
- `ToolCall`：工具调用 ID、名称、JSON 参数对象，以及可选的 `thought_signature`、`namespace`。

### 4.2 按消息来源限制内容类型

使用三个不同枚举限制各类消息允许出现的内容：

```text
UserContent
├── Text
└── Image

AssistantContent
├── Text
├── Thinking
└── ToolCall

ToolResultContent
├── Text
└── Image
```

这种设计对应 C++ 的 `std::variant`，可以在编译阶段排除非法组合：用户消息不能包含工具调用，工具结果不能包含思考内容。

### 4.3 JSON 协议

Serde 属性用于保持与 TypeScript wire format 一致：

- 使用 `type` 字段区分内容块（`#[serde(tag = "type")]`）。
- `ToolCall` 变体在 JSON 中表示为 `toolCall`。
- Rust 的 snake_case 字段转换为 JSON camelCase。
- 值为 `None` 的可选字段在序列化时省略。

## 5. `pi-ai` 消息类型（`message.rs`）

### 5.1 用户消息

`UserMessage`：固定角色 `UserRole::User`、正文 `UserMessageContent`、Unix 毫秒时间戳。

`UserMessageContent` 用 `#[serde(untagged)]` 表达两种正文形式：

- `Text(String)`：纯文本。
- `Blocks(Vec<UserContent>)`：文本和图片内容块数组。

`UserRole` 只有一个变体，从类型上禁止用户消息写成 `assistant` 等角色。

### 5.2 助手消息

`AssistantMessage` 包含：

- 固定角色 `AssistantRole::Assistant`，内容 `Vec<AssistantContent>`。
- API、Provider、请求模型标识。
- Provider 返回的响应模型和响应 ID。
- 诊断记录 `AssistantMessageDiagnostic`（含 `DiagnosticErrorInfo` / `DiagnosticErrorCode`）。
- token 用量 `Usage` 和费用 `UsageCost`。
- 统一停止原因 `StopReason`（Pending / Stop / Length / ToolUse / Error / Aborted / Deferred）。
- 延迟响应句柄 `DeferredHandle`。
- 错误信息、Provider 原始停止原因、回合结束标记、Unix 毫秒时间戳。

### 5.3 工具结果消息

`ToolResultMessage<TDetails = Value>` 包含：

- 固定角色 `ToolResultRole::ToolResult`（JSON 为 `toolResult`）。
- 对应工具调用的 `tool_call_id` 和 `tool_name`。
- 文本或图片结果 `Vec<ToolResultContent>`。
- 泛型的工具附加数据 `TDetails`（类似 C++ `template <typename TDetails = JsonValue>`）。
- 可选的工具自身用量、执行后新增的工具名。
- `is_error`、Unix 毫秒时间戳。

### 5.4 完整对话消息

```rust
pub enum ConversationMessage {
    User(UserMessage),
    Assistant(Box<AssistantMessage>),
    ToolResult(Box<ToolResultMessage>),
}
```

`Assistant` 和 `ToolResult` 用 `Box` 装箱，避免枚举因大变体而整体变胖（Clippy 的 `large_enum_variant` 建议）。已实现三种消息到 `ConversationMessage` 的 `From` 转换。

## 6. `pi-ai` 上下文与工具（`context.rs`）

- `Tool<TParameters = Value>`：发给模型的工具定义（name / description / parameters / constrained_sampling）。
- `Context`：一次模型调用的完整输入（system_prompt / messages / tools）。
- `ConstrainedSamplingConfig`：`json_schema`（严格度 `prefer` / `require`）或 `grammar`（`GrammarVariants`）。
- 辅助类型：`GrammarFormat`、`GrammarVariants`（`HashMap<GrammarFormat, String>`）、`JsonSchemaStrict`。

## 7. `pi-ai` 流式事件（`event.rs`）

`AssistantMessageEvent` 用 `#[serde(tag = "type")]` 表达 12 种流式事件：

- `start`
- text：`text_start` / `text_delta` / `text_end`
- thinking：`thinking_start` / `thinking_delta` / `thinking_end`
- toolcall：`toolcall_start` / `toolcall_delta` / `toolcall_end`
- 结束：`done`（携带最终消息） / `error`（携带错误消息）

除 `done` / `error` 外，其余事件都携带一个 `partial`（「到目前为止已组装好的部分助手消息」）。`partial()` 方法统一从变体里取出这个公共字段，`done` / `error` 返回 `None`。

## 8. `pi-ai` 图片、成本、兼容配置、模型目录

- `images.rs`：`ImagesContext`、`ImagesStopReason`、`AssistantImages`。图片生成走单独调用路径，用 `output: Vec<UserContent>` 承载结果。对应 types.ts 第 469-488 行。
- `cost.rs`：`ModelCostRates`、`ModelCostTier`、`ModelCost`。原版用 `extends`，Rust 用组合 + `#[serde(flatten)]` 让字段平铺，序列化结果与继承一致。对应 types.ts 第 803-818 行。
- `compat.rs`：Provider 兼容配置（`OpenAICompletionsCompat`、`OpenAIResponsesCompat`、`AnthropicMessagesCompat`、`BedrockCompat`）与路由偏好（`OpenRouterRouting`、`VercelGatewayRouting`），以及一组支撑类型。对应 types.ts 第 86-116、307-311、557-801 行。
- `model.rs`：`Model`、`ImagesModel`、`ModelCompat`、`ThinkingLevel`、`ModelThinkingLevel`、`ThinkingLevelMap`、`InputType`。原版 `Model<TApi>` 是泛型，这里简化为非泛型，`api` / `provider` 用 `String`；`ModelCompat` 用带 `tag = "api"` 的枚举表达「按 api 取四种配置之一」。对应 types.ts 第 820-857 行。

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

- 职责：`Model` 描述模型是什么，`Context` 描述一次调用发送什么，Provider 把后者变成一串流式事件。
- `stream` 返回装箱的迭代器对象，调用方只 `next()` 取事件，不关心底层来源（C++ 对照：返回范围的虚函数）。
- `FauxProvider`：不联网、不要 Key、结果可控的测试实现。持有一份模型清单和一个脚本队列，按脚本依次返回 `Text` 或 `ToolCall`；脚本用完后回退为「回显最后一条用户文本」。
- 脚本队列用 `RefCell<VecDeque<FauxResponse>>`：`stream` 只借用 `&self`，但需要推进队列，用内部可变性实现（C++ 对照：`mutable` 成员）。

## 10. `pi-agent-core` Agent 运行时（`lib.rs`）

`Agent` 持有：

- `provider: Box<dyn Provider>`（C++ 对照：`std::unique_ptr<Provider>`，抽象基类指针）。
- `model: Model`、`system_prompt: Option<String>`。
- `tools: Vec<Box<dyn AgentTool>>`、`messages: Vec<ConversationMessage>`。

主要方法：

- `run_once`：单轮。组装 `Context` → 调 `provider.stream` → 收集事件 → 追加并返回最终助手消息。过程中向 sink 发消息级事件（`MessageStart` / `MessageUpdate` / `MessageEnd`）；流在结束前没有 `done` / `error` 时返回 `AgentError::IncompleteStream`。
- `run` / `run_loop`：多轮循环。反复「请求模型 → 执行工具 → 再请求」，直到助手不再发起工具调用（或出错/中止）。最多 `MAX_TURNS = 32` 轮，超出返回 `AgentError::MaxTurnsExceeded`。`run` 只负责发出首尾的 `AgentStart` / `AgentEnd`，中间交给 `run_loop`，保证 `agent_end` 一定发出。
- `execute_tool_calls` / `execute_one`：执行助手消息里的所有工具调用，追加工具结果消息。工具不存在或执行失败时返回一条 `is_error = true` 的文本结果，而不是中断流程。

## 11. `pi-agent-core` 工具系统

### 11.1 工具抽象（`tool.rs`）

```rust
pub trait AgentTool {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value; // JSON Schema
    fn execute(&self, call: &ToolCall) -> Result<ToolResult, ToolError>;
}
```

`pi-ai` 的 `Tool` 是「发给模型的工具定义」；`AgentTool` 在此基础上增加 `execute` 行为（C++ 对照：含纯虚函数 `execute` 的抽象基类）。

`ToolResult` 字段：`content`（文本/图片）、`details`、`usage`、`added_tool_names`、`terminate`。另有示例工具 `EchoTool`。

Rust 没有异常，工具失败用 `Result<ToolResult, ToolError>` 表达，Agent 再转成 `is_error = true` 的工具结果消息。

### 11.2 内置工具（`tools/`）

- `read`：读取文本文件，支持 `offset`（1 起）和 `limit`；结果按 2000 行 / 50KB 截断，超出时在末尾附「用 offset=... 继续读取」的提示。
- `write`：把 `content` 覆盖写入 `path`。
- `edit`：按 `edits[]`（`oldText` / `newText`）做精确替换。要求每个 `oldText` 在原文中唯一出现、各替换区域不重叠；应用时先定位全部区间、排序、查重叠，再从后往前替换，避免改变前面区域的字节下标。

工具不直接调用 `std::fs`，而是通过 `Environment`（见下一节）。

### 11.3 截断（`truncate.rs`）

`truncate_head(content, max_lines, max_bytes)`：行数上限和字节上限谁先到用谁，从头部保留完整行，不返回半行。默认 `DEFAULT_MAX_LINES = 2000`、`DEFAULT_MAX_BYTES = 50 * 1024`。返回 `TruncationResult`（content / truncated / truncated_by / total_lines / output_lines）。

## 12. `pi-agent-core` 执行环境（`environment.rs`）

```rust
pub trait Environment {
    fn read_file(&self, path: &str) -> Result<String, EnvError>;
    fn write_file(&self, path: &str, content: &str) -> Result<(), EnvError>;
}
```

- `RealEnvironment`：真实文件系统实现。
- 工具持有 `Box<dyn Environment>`，测试时用内存实现 `FakeEnv` 替换（C++ 对照：抽象基类 + 依赖注入 / 打桩），与 Provider / FauxProvider 是同一套设计。

## 13. `pi-agent-core` 事件流（`event.rs`）

`AgentEvent` 分三层：

- run 级：`AgentStart` / `AgentEnd { messages }`。
- turn 级：`TurnStart` / `TurnEnd { message, tool_results }`。
- 消息与工具级：`MessageStart` / `MessageUpdate { message, event }` / `MessageEnd`，`ToolExecutionStart` / `ToolExecutionUpdate`（暂未发出）/ `ToolExecutionEnd`。

消息、事件、工具结果较大，相关字段统一用 `Box` 装箱，只占一个指针。

## 14. `pi-tui` 与 `pi-coding-agent`

- `pi-tui`：只有 `Renderer` trait（`render_line`）和写入 stdout 的 `PlainRenderer`。尚未实现输入、布局、增量渲染、组件和交互控制。
- `pi-coding-agent`（`src/main.rs`）：创建 `FauxProvider`，脚本为「调用 read 读取 Cargo.toml → 回一段文本」；注册 `EchoTool`、`ReadTool`、`WriteTool`、`EditTool`（都基于 `RealEnvironment`）；运行 `agent.run`，在 `ToolExecutionEnd` 时打印工具名，最后打印版本、消息数和最终停止原因。

目前还没有参数解析、真实模型调用和交互循环。

## 15. 已完成的测试

`cargo test --workspace` 共 43 个测试，全部通过。

`pi-ai`（26 个）：

- `tests/content.rs`（2）：工具调用 JSON 形状、内容块序列化往返。
- `tests/message.rs`（11）：纯文本/图文用户消息、非法用户消息拒绝、助手消息完整 JSON 形状、诊断与延迟句柄、非法助手消息拒绝、泛型工具详情、图片工具结果与用量、非法工具结果拒绝、对话消息角色与顺序、非法对话消息拒绝。
- `tests/context.rs`（3）：工具/上下文、受约束采样、schema 严格度。
- `tests/event.rs`（3）：流式事件形状与 `partial()`。
- `tests/images.rs`（2）：图片生成上下文与结果。
- `tests/model.rs`（3）：成本档位平铺、compat、模型往返。
- `tests/provider.rs`（2）：FauxProvider 的文本流和脚本化工具调用。

`pi-agent-core`（17 个）：

- `src/lib.rs` 内联（8）：单轮流式并追加助手消息、空流报错、执行已注册工具并追加结果、未知工具标记错误、`run` 执行工具直到文本、`run` 发出 turn/message 事件、达到最大轮数、用户+工具调用+工具结果的对话往返。
- `tests/tools.rs`（9）：read 读取/offset+limit/缺参数/越界，write 后 read 往返，edit 唯一替换/非唯一拒绝/未找到拒绝/多处不重叠替换。

## 16. 已学习和使用的 Rust 概念

- Cargo workspace、crate、模块系统（`mod` / `use` / `pub use` / `crate::` / `super::`）。
- `struct`、携带数据的 `enum`、`match` 与模式绑定。
- `Option<T>`、`Vec<T>`、`HashMap<K, V>`、泛型（默认类型参数 `<T = Value>`）。
- `String` 的所有权、移动、`clone`、借用。
- trait、`impl`、派生宏、trait object（`Box<dyn Trait>`）与 `dyn` 分发。
- 属性 `#[...]`、Serde 序列化 / 反序列化 / `untagged` / `tag` / `flatten` / `rename_all` / `skip_serializing_if`。
- `Result` 与 `?`、`map_err`、`ok_or_else`，用错误类型替代异常。
- `Box` 装箱（装箱迭代器、装箱 trait object、缩小枚举变体）。
- 内部可变性 `RefCell`、`Rc`（测试里共享内存文件系统）、`std::cell` 与借用规则。
- 迭代器（`into_iter` / `rev` / `next`、`Box<dyn Iterator>`）、`match_indices`、`windows`、`saturating_sub`、`usize::from`。
- 用类型排除非法状态（角色枚举、按来源限内容、泛型 details）。

## 17. 当前代码状态

- 工作区处于已提交状态，`cargo test --workspace`（43 个）全绿。
- `cargo fmt --all --check` 与 `cargo clippy --workspace --all-targets` 均通过，无 warning。
- `docs/` 下另有两份历史文档：`学习总结.md`（早期 pi-ai 类型移植）、`pi-agent-rust-rewrite-conversation.md`（与 Codex 的对话整理），内容针对早期阶段，本文档为准。

### 提交里程碑（`pi-rust` 分支）

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
```

## 18. 建议的下一步

1. 实现一个真实 Provider：从 OpenAI-compatible 的 HTTP 请求和 SSE 流解析开始，把 `AssistantMessageEvent` 真正接上网络。
2. 实现模型注册表 / 配置（模型目录、API Key 与 base URL 的来源）。
3. 为 `pi-coding-agent` 加参数解析和交互循环：读取用户输入 → `agent.run` → 按 `AgentEvent` 渲染。
4. 扩充 `pi-tui`：输入、增量渲染、组件与交互控制。

近期最合适的里程碑：把 FauxProvider 替换成一个能真正访问 OpenAI-compatible 端点的 Provider，在 CLI 里完成一次「用户输入 → 模型回复 → 工具调用 → 工具结果」的真实往返。
