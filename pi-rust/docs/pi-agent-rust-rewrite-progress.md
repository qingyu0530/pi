# Pi Agent Rust 重写进度总结

> 更新日期：2026-09-09  
> Rust workspace：`E:\vscode\pi\pi-rust`

## 1. 项目目标

当前工作的目标是使用 Rust 重写 Pi Agent，同时通过实际项目学习 Rust。

重写不是把 TypeScript 逐行翻译成 Rust，而是保持原项目的职责划分和外部数据协议，再使用 Rust 的类型系统重新表达这些设计。学习过程中主要使用 C++ 的相近概念进行对照。

原 TypeScript 项目的主要分层如下：

- `packages/ai`：统一的模型、消息、Provider、流式响应和工具调用协议。
- `packages/agent`：Agent 状态、对话循环、工具执行和事件流。
- `packages/tui`：终端界面和渲染组件。
- `packages/coding-agent`：最终的 `pi` 命令行程序。

Rust workspace 按照相同职责建立了四个 crate：

- `pi-ai`
- `pi-agent-core`
- `pi-tui`
- `pi-coding-agent`

## 2. Rust 工程环境

已经完成以下基础配置：

- 建立 Cargo workspace。
- 使用 Rust 2024 edition。
- 最低 Rust 版本设置为 `1.85`。
- stable 工具链包含 `rustfmt` 和 Clippy。
- workspace 禁止使用 `unsafe`。
- 统一管理 `serde` 和 `serde_json` 依赖版本。
- 生成并保留 `Cargo.lock`。
- 将 `target` 目录加入 `pi-rust/.gitignore`。

workspace 当前包含：

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

## 3. 基础 crate

### 3.1 `pi-ai`

`pi-ai` 是目前主要完成的部分，负责定义模型和 Agent 之间交换的数据。

当前已经实现：

- 内容块类型。
- 用户消息。
- 助手消息。
- 工具结果消息。
- 完整对话消息枚举。
- 与 TypeScript 协议对应的 JSON 序列化和反序列化。
- 对应的集成测试。

### 3.2 `pi-agent-core`

已经建立基础 `Agent` 结构：

- 保存消息数组。
- 添加消息。
- 只读访问消息列表。

这一部分目前仍然使用早期的简化消息类型 `Message { role, content }`，尚未接入 `pi-ai` 中新完成的 `ConversationMessage`。

### 3.3 `pi-tui`

已经建立最小渲染抽象：

- `Renderer` trait 定义逐行渲染接口。
- `PlainRenderer` 将文本写入标准输出。

这只是终端层的骨架，尚未实现输入、布局、增量渲染、组件和交互控制。

### 3.4 `pi-coding-agent`

已经建立名为 `pi` 的二进制程序：

- 初始化 `Agent`。
- 添加一条系统消息。
- 使用 `PlainRenderer` 输出程序版本和初始化消息数量。

目前还没有参数解析、模型调用、交互循环和工具系统。

## 4. `pi-ai` 内容块

`crates/pi-ai/src/content.rs` 已经实现以下类型。

### 4.1 基础内容

- `TextContent`：文本和可选文本签名。
- `ThinkingContent`：模型推理内容、可选签名和删减标记。
- `ImageContent`：Base64 图片数据和 MIME 类型。
- `ToolCall`：工具调用 ID、名称、JSON 参数以及 Provider 附加信息。

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

这种设计对应 C++ 的 `std::variant`。它可以在编译阶段排除非法组合，例如用户消息不能包含工具调用，工具结果不能包含思考内容。

### 4.3 JSON 协议

Serde 属性用于保持与 TypeScript wire format 一致：

- 使用 `type` 字段区分内容块。
- `ToolCall` 在 JSON 中表示为 `toolCall`。
- Rust 的 snake_case 字段转换为 JSON camelCase。
- 值为 `None` 的可选字段在序列化时省略。

例如：

```json
{
  "type": "toolCall",
  "id": "call_1",
  "name": "read_file",
  "arguments": {
    "path": "README.md"
  }
}
```

## 5. `pi-ai` 消息类型

`crates/pi-ai/src/message.rs` 已经实现三种完整消息。

### 5.1 用户消息

`UserMessage` 包含：

- 固定角色 `user`。
- 一段纯文本，或者文本和图片内容块数组。
- Unix 毫秒时间戳。

独立的 `UserRole` 类型保证用户消息不能错误地使用 `assistant` 等角色。

### 5.2 助手消息

`AssistantMessage` 包含：

- 固定角色 `assistant`。
- 文本、思考和工具调用内容块。
- API、Provider 和模型标识。
- Provider 返回的响应模型和响应 ID。
- token 用量和费用。
- 统一停止原因。
- 延迟响应句柄。
- 错误信息和诊断记录。
- Provider 原始停止原因。
- Provider 的回合结束标记。
- Unix 毫秒时间戳。

已经实现的辅助类型包括：

- `Usage`
- `UsageCost`
- `StopReason`
- `DeferredHandle`
- `AssistantMessageDiagnostic`
- `DiagnosticErrorInfo`
- `DiagnosticErrorCode`

### 5.3 工具结果消息

`ToolResultMessage<TDetails>` 包含：

- 固定角色 `toolResult`。
- 对应工具调用的 ID。
- 工具名称。
- 文本或图片结果。
- 泛型的工具附加数据。
- 可选的工具自身用量。
- 工具执行后新增的工具名称。
- 执行是否失败。
- Unix 毫秒时间戳。

泛型参数 `TDetails` 类似 C++ 模板参数；未指定时默认使用 `serde_json::Value`。

### 5.4 完整对话消息

`ConversationMessage` 将三种消息组合为一个枚举：

```rust
pub enum ConversationMessage {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
}
```

它对应原 TypeScript 定义：

```typescript
type Message = UserMessage | AssistantMessage | ToolResultMessage;
```

同时实现了 `From` 转换，使三种具体消息可以方便地转换成 `ConversationMessage`。

## 6. 已完成的测试

`pi-ai` 当前共有 13 个集成测试。

`tests/content.rs` 包含 2 个测试：

- 工具调用生成正确的 TypeScript JSON 形状。
- 用户和助手内容块可以完成序列化往返，并保持数据和顺序。

`tests/message.rs` 包含 11 个测试，覆盖：

- 纯文本用户消息。
- 文本和图片混合用户消息。
- 非法用户消息拒绝。
- 助手消息完整 JSON 形状。
- 诊断记录和延迟响应句柄。
- 非法助手消息拒绝。
- 泛型工具详情。
- 图片工具结果、工具用量和新增工具。
- 非法工具结果消息拒绝。
- 完整对话消息的角色和顺序。
- 未实现的消息角色拒绝。

最近一次验证结果：

```text
cargo check --workspace --all-targets    通过
cargo test --workspace                   通过，13 个测试成功
cargo fmt --all --check                  未通过
cargo clippy --workspace --all-targets   未通过
```

格式检查当前只发现空行和 import 排序问题。

Clippy 当前发现两个问题：

1. `message.rs` 的文档注释中包含 Tab 字符。
2. `ConversationMessage` 的枚举变体大小差异较大，建议对较大的变体增加间接存储，例如 `Box<AssistantMessage>`。

这些问题不会阻止普通编译和测试，但需要在当前消息模型完成前处理。

## 7. 已学习和使用的 Rust 概念

目前代码已经涉及：

- Cargo workspace 和 crate。
- Rust 模块系统以及 `mod`、`use`、`pub use`。
- `struct` 和携带数据的 `enum`。
- `Option<T>`、`Vec<T>` 和泛型。
- `String` 的所有权、移动、克隆和借用。
- trait 和派生宏。
- Rust 属性 `#[...]`。
- Serde 序列化和反序列化。
- 使用 `match` 读取枚举内容。
- 使用 Rust 类型排除非法状态。
- JSON 形状测试和序列化往返测试。

## 8. 当前代码状态

当前存在两套消息模型：

1. `pi-ai/src/lib.rs` 中早期的简化 `Role + Message`。
2. `pi-ai/src/message.rs` 中完整的 `ConversationMessage`。

`pi-agent-core` 仍然依赖第一套简化模型。这意味着消息协议已经基本完成，但 Agent 运行时还没有真正使用它。

此外，当前工作区中新增的工具结果消息、完整对话消息及部分说明文档仍属于未提交修改。继续开发时需要保留这些修改，不能使用会覆盖工作区的 Git 操作。

## 9. 建议的下一步

建议按照以下顺序继续：

1. 修复 `rustfmt` 和 Clippy 报告的问题。
2. 为尚未覆盖的内容类型补充必要测试。
3. 用 `ConversationMessage` 替换 `pi-agent-core` 使用的简化消息模型。
4. 移除不再需要的旧 `Role + Message`，避免两套模型长期并存。
5. 实现模型信息，包括模型 ID、上下文窗口、能力和计费信息。
6. 实现流式事件协议。
7. 定义 Provider trait。
8. 实现一个用于测试的 Faux Provider。
9. 实现 Agent 对话循环和工具执行。
10. 最后逐步扩展 TUI 和 CLI。

当前最合适的近期里程碑是：让 `pi-agent-core::Agent` 使用完整的 `ConversationMessage`，并通过测试完成一轮“用户消息 → 助手工具调用 → 工具结果”的内存态对话流程。
