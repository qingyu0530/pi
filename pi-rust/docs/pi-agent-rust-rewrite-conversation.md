# 用 Rust 重写 Pi Agent：历史对话整理

> 整理时间：2026-09-06  
> 原始记录：`E:\360\.codex\sessions` 和 `E:\360\.codex\archived_sessions`  
> 主要会话：`rollout-2026-09-04T18-51-14-01a06c0b-4adf-7cb2-b61e-4ab5ad24d893.jsonl`

这份文档整理了与“用 Rust 重写 Pi Agent”直接相关的提问和回答。自动审批记录、工具调用细节、重复的界面故障讨论和无关对话已省略。回答经过排版和少量合并，技术含义保持不变。

## 一、目标和学习方式

### 提问

> 我现在想用 Rust 重写一下这个项目，准备从 `packages/ai` 开始写。请先帮我把 Rust 所需要的环境弄好。

后来你进一步明确：

> 我对 Rust 基本不熟悉，主要想通过重写这个项目熟悉 Rust。我熟悉的只有 C++。请用 C++ 对照，带着我一点一点重写，尽量拆分细致，我要学习思路和代码原理。

### 回答

整个任务被确定为两个同时进行的目标：

1. 用 Rust 逐步重写 Pi Agent，首先重写 `pi-ai`。
2. 在实现过程中学习 Rust。每次只引入一小组概念，并用 C++ 做近似对照。

因此不会把原来的 TypeScript 文件机械翻译成一个巨大的 Rust 文件，而是按照职责拆开，先定义稳定的数据类型，再实现消息、Provider 和网络协议。

## 二、Rust 开发环境

### 提问

> 请先帮我把 Rust 的环境装好。

### 回答

当时安装并验证了以下环境：

- Rust stable `1.98.1`
- Cargo `1.98.1`
- MSVC C++ Build Tools 和 Windows SDK
- `rustfmt`
- Clippy
- `rust-analyzer`
- VS Code 的 `rust-lang.rust-analyzer` 扩展

还编译并运行了一个最小 Rust 程序，确认编译器、链接器和 Windows SDK 可以正常协作。Cargo 被加入用户 `PATH`，需要重新打开终端或重启 VS Code 才能让已有终端读取新的环境变量。

## 三、初始化 `pi-rust` workspace

### 提问

> 我在项目里新建了 `pi-rust` 文件夹，准备用 Rust 重写一遍这个项目。直接在这个目录下初始化 Rust workspace 和基础 crate。

### 回答

`pi-rust` 被初始化为 Cargo workspace，并按照原项目的职责建立了四个基础 crate：

- `crates/pi-ai`：消息、模型、Provider 和流式响应
- `crates/pi-agent-core`：Agent 循环、状态和工具调用
- `crates/pi-tui`：终端界面
- `crates/pi-coding-agent`：`pi` CLI 主程序

同时配置了 stable 工具链、`rustfmt`、Clippy、Cargo lockfile 和独立的 `target` 忽略规则。

当时运行并通过：

```text
cargo fmt --all --check
cargo check --workspace
cargo clippy --workspace --all-targets
```

## 四、Git 分支和第一次提交

### 提问

> 先把这一版代码推送到 GitHub，备注是“初始化 Rust workspace 和基础 crate”。新建一个分支，命名为 `pi-rust`，然后推送上去。

### 回答

创建了提交：

```text
ca8645708 feat: 初始化 Rust workspace 和基础 crate
```

随后建立本地 `pi-rust` 分支，并推送到：

```text
https://github.com/qingyu0530/pi.git
```

远程分支为 `origin/pi-rust`。根目录已有的 `.gitignore` 修改没有包含在这次提交中。

## 五、`pi-ai` 的重写路线

### 提问

> 我想先重写 `pi-ai` 这个模块，请带着我一点一点重写。

### 回答

重写被拆分为以下阶段：

1. 内容块：文本、图片、思考、工具调用
2. 消息：用户消息、助手消息、工具结果消息
3. 模型信息：上下文窗口、价格、能力
4. 计费和推理等级
5. 流式事件协议
6. Provider trait
7. Models 注册表和路由
8. OpenAI-compatible HTTP 请求
9. SSE 流解析
10. API Key 和配置
11. 其他供应商适配器

拆分的原因是原版 `packages/ai/src/types.ts` 同时包含很多概念。先把数据结构分清楚，可以用 Rust 类型系统表达约束，并让后面的 Provider 实现建立在稳定的数据模型上。

## 六、第 1.1 步：`content.rs` 内容块

### 提问

> 请继续带着我一步一步用 Rust 重写 `pi-ai`，尽量拆分细致。我想学习思路和代码原理。

### 回答：为什么先定义内容块

一条模型消息不一定只是字符串。助手可能依次产生思考、文本和工具调用：

```json
[
  { "type": "thinking", "thinking": "分析项目结构" },
  { "type": "text", "text": "开始修改" },
  {
    "type": "toolCall",
    "id": "call_1",
    "name": "read_file",
    "arguments": { "path": "README.md" }
  }
]
```

Rust 使用携带数据的枚举表示这些情况：

```rust
pub enum AssistantContent {
    Text(TextContent),
    Thinking(ThinkingContent),
    ToolCall(ToolCall),
}
```

它可以近似理解为 C++：

```cpp
using AssistantContent =
    std::variant<TextContent, ThinkingContent, ToolCall>;
```

一个 `AssistantContent` 值在同一时刻只能是其中一种变体。

代码进一步按照消息来源拆成三个枚举：

```rust
pub enum UserContent {
    Text(TextContent),
    Image(ImageContent),
}

pub enum AssistantContent {
    Text(TextContent),
    Thinking(ThinkingContent),
    ToolCall(ToolCall),
}

pub enum ToolResultContent {
    Text(TextContent),
    Image(ImageContent),
}
```

这样可以在编译阶段排除非法组合。例如 `UserContent` 根本没有 `ToolCall` 变体，所以不能把模型的工具调用伪装成用户内容。

### 提问

> `#[serde(rename = "text")]` 这种带井号的语句是什么意思？C++ 中的 `#define` 是宏定义，Rust 中这种带 `#` 的语句是什么？

### 回答

Rust 的 `#[...]` 叫属性（attribute）。属性附在结构体、枚举、字段或函数上，为编译器或宏提供额外说明。它不是 C++ 预处理器的文本替换。

例如：

```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum UserContent {
    #[serde(rename = "text")]
    Text(TextContent),
}
```

三处属性分别表示：

- `derive` 自动生成 `Serialize` 和 `Deserialize` trait 的实现。
- `serde(tag = "type")` 使用 JSON 的 `type` 字段区分枚举变体。
- `serde(rename = "text")` 将 Rust 变体 `Text` 映射为 JSON 名称 `"text"`。

因此下面的 Rust 值：

```rust
UserContent::Text(TextContent {
    text: "你好".to_owned(),
    text_signature: None,
})
```

会序列化为：

```json
{
  "type": "text",
  "text": "你好"
}
```

Rust 代码中仍然使用 `UserContent::Text`，`rename` 只控制外部数据名称。

### 提问

> `Serialize` 是转换成 JSON 吗？它是自带的，不需要实现吗？

### 回答

`Serialize` 支持将 Rust 值序列化，但它本身不限定 JSON，也可以被 YAML、TOML 等格式库使用。这里需要区分三个东西：

- `serde::Serialize` 是 trait，规定类型如何被序列化。
- `#[derive(Serialize)]` 是派生宏，自动生成这个 trait 的实现。
- `serde_json::to_string(...)` 才是真正生成 JSON 字符串的函数。

例如：

```rust
use serde::Serialize;

#[derive(Serialize)]
struct Person {
    name: String,
    age: u32,
}

let person = Person {
    name: "小明".to_owned(),
    age: 20,
};

let json = serde_json::to_string(&person).unwrap();
```

`derive` 在编译时生成实现，`to_string` 在运行时执行转换。不需要手写实现，是因为派生宏帮忙生成了实现。

### 提问

> Serde 在哪里？是 Rust 自带的吗？

### 回答

Serde 不是 Rust 标准库的一部分，而是第三方 crate，由 Cargo 下载和管理。

workspace 根配置统一声明版本：

```toml
[workspace.dependencies]
serde = { version = "=1.0.229", features = ["derive"] }
serde_json = "=1.0.151"
```

`pi-ai` 再声明自己使用 workspace 中的依赖：

```toml
[dependencies]
serde.workspace = true
serde_json.workspace = true
```

源码中的：

```rust
use serde::{Deserialize, Serialize};
```

只负责把两个名字引入当前作用域。依赖声明和下载由 `Cargo.toml` 与 Cargo 负责。

### 提问

> 我理解 Serde 是一个库，类似 C++ 头文件。`use serde::{Deserialize, Serialize}` 是把库里的两个函数引入进来，后面就能直接用了，对吗？

### 回答

“引入名字后可以直接使用”是对的，但有两点需要修正。

第一，Rust 的 `use` 更接近 C++ 的 `using`，不是 `#include`：

```cpp
using serde::Deserialize;
using serde::Serialize;
```

第二，`Deserialize` 和 `Serialize` 不是普通函数。它们是 trait，同时 Serde 还导出了同名派生宏。`use` 能引入函数、类型、trait 或宏等名字，不能仅凭 `use` 判断某个名字是函数。

真正执行 JSON 转换的是 `serde_json::to_string`、`serde_json::to_value` 和 `serde_json::from_str` 等函数。

### 提问

> trait 是什么？

### 回答

trait 是一组行为的约定，规定一个类型需要提供哪些方法。它可以类比 C++ 的抽象接口，但 Rust 类型不需要通过继承来实现它。

```rust
trait Greet {
    fn greet(&self) -> String;
}

struct Person {
    name: String,
}

impl Greet for Person {
    fn greet(&self) -> String {
        format!("你好，我是{}", self.name)
    }
}
```

`impl Greet for Person` 表示为 `Person` 实现 `Greet` 规定的行为。函数还可以接受任何实现该 trait 的类型：

```rust
fn say_hello(value: &impl Greet) {
    println!("{}", value.greet());
}
```

回到 Serde：`Serialize` 是 trait，`derive(Serialize)` 自动生成 `impl Serialize for 某个类型`，JSON 库据此知道如何读取字段并序列化。

### 提问

> 请从头熟悉 `content.rs`，用 C++ 对照讲解每一行代码。

### 回答：`content.rs` 的核心语法

文件的职责是定义“消息中允许包含哪些数据”以及“这些数据如何转换为 JSON”，不负责请求模型或执行工具。

#### `use`：引入名字

```rust
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
```

花括号表示从同一路径一次引入多个名字。`Map` 表示 JSON 对象，`Value` 表示任意 JSON 值。

#### `TextPhase`：最简单的枚举

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextPhase {
    Commentary,
    FinalAnswer,
}
```

C++ 近似写法：

```cpp
enum class TextPhase {
    Commentary,
    FinalAnswer
};
```

`pub` 表示公开。`rename_all = "snake_case"` 使外部名称变成：

```text
Commentary  -> "commentary"
FinalAnswer -> "final_answer"
```

派生的 trait 提供以下能力：

| Trait | 作用 |
| --- | --- |
| `Clone` | 显式克隆值 |
| `Copy` | 赋值、传参时允许按值复制 |
| `Debug` | 使用 `{:?}` 输出调试信息 |
| `PartialEq` | 支持 `==` 和 `!=` |
| `Eq` | 声明相等关系满足完整等价关系 |
| `Serialize` | 支持序列化 |
| `Deserialize` | 支持反序列化 |

#### `TextSignature`：结构体和 `Option`

```rust
pub struct TextSignature {
    pub v: u8,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<TextPhase>,
}
```

C++ 近似写法：

```cpp
struct TextSignature {
    std::uint8_t v;
    std::string id;
    std::optional<TextPhase> phase;
};
```

Rust 字段采用 `名字: 类型` 的顺序。`pub struct` 只公开类型，字段默认仍然私有，所以每个对外字段也需要 `pub`。

`Option<T>` 表示值可能存在，也可能不存在：

```rust
enum Option<T> {
    None,
    Some(T),
}
```

当 `phase` 是 `None` 时，`skip_serializing_if` 会在 JSON 中省略整个字段，而不是输出 `"phase": null`。

#### `String` 的所有权

C++ 中下面的赋值通常会复制字符串：

```cpp
std::string a = "hello";
std::string b = a;
```

Rust 中，`String` 默认发生所有权移动：

```rust
let a = String::from("hello");
let b = a;
// 此后不能继续使用 a。
```

需要独立副本时显式克隆，只临时读取时借用：

```rust
let b = a.clone();
let reference = &a;
```

这也解释了为什么包含 `String` 的结构体通常派生 `Clone`，但不派生 `Copy`。

#### 文本、思考和图片结构体

```rust
pub struct TextContent {
    pub text: String,
    pub text_signature: Option<String>,
}

pub struct ThinkingContent {
    pub thinking: String,
    pub thinking_signature: Option<String>,
    pub redacted: Option<bool>,
}

pub struct ImageContent {
    pub data: String,
    pub mime_type: String,
}
```

`#[serde(rename_all = "camelCase")]` 使 Rust 的 `text_signature`、`mime_type` 等字段在 JSON 中变成 `textSignature`、`mimeType`。

`Option<bool>` 有三种状态：`Some(true)`、`Some(false)` 和 `None`。`ImageContent.data` 保存 Base64 文本，但类型本身不验证 Base64 是否有效，也不检查数据是否确实是所声明的图片格式。

#### `ToolCall`：任意 JSON 参数对象

```rust
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Map<String, Value>,
    pub thought_signature: Option<String>,
    pub namespace: Option<String>,
}
```

`Map<String, Value>` 可以近似理解为：

```cpp
std::map<std::string, JsonValue>
```

最外层必须是 JSON 对象，但各个值可以是字符串、数字、布尔值、数组、对象或 `null`：

```json
{
  "path": "README.md",
  "offset": 10,
  "recursive": true
}
```

`ToolCall` 只描述模型希望执行的工具调用。它本身不会打开文件或执行命令。

#### 用 `match` 读取枚举中的数据

```rust
match &content {
    UserContent::Text(text) => {
        println!("文本内容：{}", text.text);
    }
    UserContent::Image(image) => {
        println!("图片类型：{}", image.mime_type);
    }
}
```

`&content` 表示借用，避免取走内容的所有权。`match` 可以类比为更强的 `switch`：它既判断变体，也安全地提取变体携带的数据，并由编译器检查是否覆盖了所有情况。

## 七、`content.rs` 的逻辑测试

### 提问

> `content.rs` 有没有测试？我知道肯定运行了 `cargo test`，但代码逻辑呢？如果测试了，请告诉我测试内容。

### 回答

测试不只检查是否能编译，还验证了 JSON 转换的实际行为。测试文件是 `pi-rust/crates/pi-ai/tests/content.rs`，当时有两个测试。

### 测试一：工具调用的 JSON 形状

先构造：

```rust
AssistantContent::ToolCall(ToolCall {
    id: "call_1".to_owned(),
    name: "read_file".to_owned(),
    arguments,
    thought_signature: None,
    namespace: None,
})
```

再用 `serde_json::to_value` 转换，并断言结果等于：

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

它验证：

- `type` 标记正确存在。
- `ToolCall` 被重命名为 `toolCall`。
- 工具参数被完整保留。
- 值为 `None` 的可选字段被省略。
- JSON 与原 TypeScript 协议一致。

### 测试二：序列化往返

测试构造一个用户图片块，以及一个依次包含思考和文本的助手内容数组，然后执行：

```text
原始 Rust 值 -> JSON 字符串 -> 还原后的 Rust 值
```

最后使用 `assert_eq!` 比较还原后的值与原始值。它检查类型、正文、图片数据、签名和数组顺序是否丢失或改变。

“往返一致”不能单独证明 JSON 与原协议一致，因为编码器和解码器可能共享同一个错误。第一个测试因此使用手写的预期 JSON 做额外验证。

当时也明确记录了未覆盖的内容：

- 没有逐一测试 `TextPhase`、`TextSignature` 和 `ToolResultContent`。
- 没有覆盖所有可选字段组合。
- 没有验证 Base64 能否解码成真实图片。
- 没有真正执行 `read_file`；`ToolCall` 只是数据结构。

## 八、第 1.2 步：`message.rs` 用户消息

### 提问

> 好的，请继续带着我用 Rust 重构下一段代码。

### 回答：为什么需要消息类型

`TextContent` 只能表达一块文本：

```json
{"type":"text","text":"你好"}
```

它没有说明是谁发送的，也没有时间戳。一条完整用户消息需要：

```json
{
  "role": "user",
  "content": "你好",
  "timestamp": 1700000000123
}
```

原 TypeScript 类型近似为：

```typescript
export interface UserMessage {
    role: "user";
    content: string | (TextContent | ImageContent)[];
    timestamp: number;
}
```

因此新增 `message.rs`，把上一节的内容块组合成带角色和时间戳的消息。

### 模块引用

```rust
//! 消息类型：将内容块组合成带角色和时间戳的消息。

use serde::{Deserialize, Serialize};
use crate::content::UserContent;
```

`//!` 是当前模块的文档注释；`///` 通常给紧随其后的类型、字段或函数写文档。

`crate::content::UserContent` 从当前 `pi-ai` crate 的 `content` 模块引用 `UserContent`，近似于：

```cpp
using pi_ai::content::UserContent;
```

### `UserRole`：角色只能是用户

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum UserRole {
    #[serde(rename = "user")]
    User,
}
```

如果 `role` 使用 `String`，调用者就可能错误地写入 `"assistant"`。使用只有一个变体的 `UserRole` 后，只能正常构造：

```rust
role: UserRole::User
```

读取 JSON 时，Serde 也会拒绝错误角色。这把 TypeScript 中 `role: "user"` 的限制落实到了 Rust 类型中。

### `UserMessageContent`：正文的两种形式

```rust
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserMessageContent {
    Text(String),
    Blocks(Vec<UserContent>),
}
```

C++ 近似写法：

```cpp
using UserMessageContent = std::variant<
    std::string,
    std::vector<UserContent>
>;
```

这里存在两层枚举：

| 类型 | 表达的问题 |
| --- | --- |
| `UserMessageContent` | 整条正文是字符串，还是内容块数组？ |
| `UserContent` | 数组里的某一个内容块是文本，还是图片？ |

`#[serde(untagged)]` 表示外层直接使用变体内部的数据，不输出 `Text` 或 `Blocks` 变体名称：

```rust
UserMessageContent::Text(String::from("你好"))
```

对应：

```json
"你好"
```

而内容块数组对应：

```json
[
  {"type":"text","text":"你好"}
]
```

外层没有额外包装，但数组里的 `UserContent` 仍然保留自己的 `type` 字段。

### `UserMessage`：组装角色、正文和时间戳

```rust
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct UserMessage {
    pub role: UserRole,
    pub content: UserMessageContent,
    pub timestamp: u64,
}
```

C++ 近似写法：

```cpp
struct UserMessage {
    UserRole role;
    UserMessageContent content;
    std::uint64_t timestamp;
};
```

`timestamp` 是 Unix 纪元起的非负整数毫秒数，由调用方提供，这个结构体本身不会读取系统时间。

构造示例：

```rust
let message = UserMessage {
    role: UserRole::User,
    content: UserMessageContent::Text(String::from("你好，Pi")),
    timestamp: 1_700_000_000_123,
};
```

Rust 数字中的下划线只是可读性分隔符，相当于 C++ 的：

```cpp
1'700'000'000'123
```

序列化结果为：

```json
{"role":"user","content":"你好，Pi","timestamp":1700000000123}
```

### 在 `lib.rs` 中公开消息类型

```rust
mod message;
pub use message::{UserMessage, UserMessageContent, UserRole};
```

`mod message` 告诉编译器加载同目录的 `message.rs`。`pub use` 将三个类型从 crate 根重新导出，因此其他 crate 可以直接写：

```rust
use pi_ai::{UserMessage, UserMessageContent, UserRole};
```

### `message.rs` 的测试

当时新增了三个测试：

1. 纯文本消息与 TypeScript JSON 形状一致，并可以反序列化还原。
2. 文本和图片混合内容保持顺序、签名、媒体类型和图片数据。
3. 拒绝错误角色、缺失字段、负时间戳、数字正文、思考块和工具调用块等非法用户消息。

对话记录中的验证结果是三个新测试通过；加上 `content.rs` 的两个测试，共五个测试通过。该步骤当时尚未提交或推送。

## 九、历史节点和当前工作区状态

截至这批旧对话结束时：

- `ca8645708`：初始化 Rust workspace 和基础 crate，已推送。
- `dd8ae8d93`：添加 Rust 内容块、序列化测试和学习注释，已推送到 `origin/pi-rust`。
- `message.rs`、相关 `lib.rs` 修改和 `tests/message.rs` 当时尚未提交。

整理本文时，本地工作区仍能看到未提交的 `message.rs`、`tests/message.rs` 和 `lib.rs` 修改。因此下一步可以从理解并复查用户消息类型继续，然后再实现 `AssistantMessage` 和 `ToolResultMessage`。

## 十、目前形成的 Rust 学习脉络

到 `message.rs` 为止，已经接触了这些 Rust 概念：

- Cargo workspace 与 crate
- `use` 路径和模块系统
- `struct`、携带数据的 `enum` 与 `match`
- `Option<T>` 和 `Vec<T>`
- trait、`impl` 与派生宏
- 属性 `#[...]`
- `String` 的所有权、移动、克隆和借用
- Serde 序列化与反序列化
- 用类型限制非法消息
- 用明确 JSON 和序列化往返测试协议行为

接下来的自然步骤是完成另外两种消息：

```text
AssistantMessage
ToolResultMessage
```

三种消息齐备后，再用一个总消息枚举将它们组合起来，并逐步替换当前过于简单的 `Role + String` 消息模型。
