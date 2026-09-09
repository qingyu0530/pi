// 上一个文件是 content.rs，定义了文本 图片等内容
// 但是文本块本身没有说明
//  谁发送了内容
// 一条消息包含什么
// 消息是什么时间产生的

// message.rs 负责把内容块组合成一条完整的用户消息
/*
UserMessage
├── role: UserRole
├── content: UserMessageContent
└── timestamp: u64

*/

//! 消息类型：将内容块组合成带角色和时间戳的消息。

use serde::{Deserialize, Serialize};
// 引用 content.rs 中的类型
// crate        当前 crate，即 pi-ai
// content      content 模块
// UserContent  模块中的 UserContent 类型
// 可以近似理解为 C++：
// using pi_ai::content::UserContent;
// 这里的 crate 是 Rust 的固定关键字，表示“当前 crate 的根
use crate::content::{AssistantContent, ToolResultContent, UserContent};
use serde_json::{Map, Number, Value};

/// 用户消息的固定角色。不能使用 Assistant 或 Tool 等其他角色。
/// PartialEq 可以使用 == 和 !=
//  Eq	声明它具有完整的相等关系
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
// 只有一个变体
// 这是一个专门用于 UserMessage 的角色类型，所以它只允许：
// UserRole::User
// 如果将角色定义成字符串：
// pub role: String,
// 调用者就可能写出：
// role: String::from("assistant")
// 这样会产生一条“角色是助手的用户消息”。
// 使用 UserRole 后，这种非法状态无法正常构造。
// 为什么不能直接省略 role
// 因为发送给模型的 JSON 协议仍然需要：
// {
//   "role": "user"
// }
// Rust 内部用枚举保证类型安全，Serde 再把枚举转换成协议要求的字符串。
pub enum UserRole {
    #[serde(rename = "user")]
    User,
}

/// 用户消息的正文：纯文本，或者按顺序排列的文本、图片内容块。
/// C++ 对照：std::variant<std::string, std::vector<UserContent>>。
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
// 告诉 Serde：序列化时直接输出变体携带的数据，不要输出枚举变体名称。
// 只序列化枚举内部的数据；
// 不序列化 Text 或 Number 这些 Rust 变体名称。
#[serde(untagged)]
// Rust 没有 TypeScript 这种直接写联合类型的语法，因此定义一个枚举表达两种情况
// 纯文本 或者 内容块数组
pub enum UserMessageContent {
    // content: string | (TextContent | ImageContent)[];

    // untagged 使 JSON 直接保存字符串，不添加 "Text" 包装。
    Text(String),
    // Vec<T> 类似 std::vector<T>。数组中的每个块仍有自己的 type 标记。
    Blocks(Vec<UserContent>),
}

/// 对应原版 types.ts 中的 UserMessage。
/// 完整用户消息
/*
为什么这里没有 Copy
UserMessage 内部最终可能包含：
String
Vec<UserContent>
这些类型拥有动态内存，不能进行简单的按位复制，所以 UserMessage 只有 Clone，没有 Copy。


*/
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct UserMessage {
    pub role: UserRole,
    pub content: UserMessageContent,
    /// 自 Unix 纪元起的非负整数毫秒数，由调用方提供，不自动读取时钟。
    /// u64 类似 C++ 的 std::uint64_t。
    pub timestamp: u64,
}

// 字段都带 pub: pub struct UserMessage
// 只表示结构体类型公开。
/*
Rust 的字段默认私有，因此字段还要单独写 pub：

pub role: UserRole,
pub content: UserMessageContent,
pub timestamp: u64,

C++ 的 struct 字段默认是 public，这一点与 Rust 不同。

*/

/*
前半部分的枚举回答“用户消息中的字段允许使用什么数据”：

UserRole
// 用户消息的角色，只能是 User，对应 JSON 中的 "user"

UserMessageContent
// 用户消息的正文，可以是：
// 1. 一段纯文本 String
// 2. 一组文本、图片内容块 Vec<UserContent>

后半部分的结构体回答“一条完整的用户消息由什么组成”：

UserMessage
// role：消息角色
// content：消息正文
// timestamp：消息产生时的 Unix 毫秒时间戳

而 #[derive(...)] 让这些类型支持复制、比较以及序列化和反序列化；
#[serde(rename = "user")] 将 UserRole::User 转换成 JSON 中的 "user"；
#[serde(untagged)] 让正文直接转换成字符串或数组，不额外添加枚举变体名称。


*/

// -----------------------------------------------------------------------------
// 新增：助手消息类型
// 用户消息只需要记录“用户说了什么”；
// 助手消息还必须记录“哪个模型回答、回答内容、消耗多少 token、为何停止”。
// -----------------------------------------------------------------------------

/// 助手消息的固定角色。只能使用 Assistant，不能写成 User 或 ToolResult。
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AssistantRole {
    // 它与前面的 UserRole 相同，限制助手消息的角色只能是
    #[serde(rename = "assistant")]
    Assistant,
}

/// 一次模型调用各类 token 的价格。
/// 模型调用费用
/// 费用使用 f64，因为价格可以是小数；因此包含它的 Usage 只能派生 PartialEq，不能派生 Eq。
/// 没有 Eq。
/// 因为 f64 对应 C++ 的 double，浮点数可能有 NaN。而：
/// NaN != NaN
/// 不满足严格相等关系，所以 Rust 不允许包含 f64 的类型实现 Eq。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
// 其实就是花了多少钱  一次调用的费用明细
// Usage：这次调用“用了多少 token”
// UsageCost：这些 token “花了多少钱”
/*
Usage     = 用了多少分钟、多少流量
UsageCost = 这些分钟和流量分别花了多少钱

UsageCost 结构体的字段含义：
输入 token 花费：       0.001
输出 token 花费：       0.002
读取缓存 token 花费：   0.0001
写入缓存 token 花费：   0.0002
总费用：                0.0033

*/
pub struct UsageCost {
    pub input: f64,  // 表示输入 token 的费用
    pub output: f64, // 和 input 一样，表示模型输出 token 的费用。
    // cache_read 表示从模型 Provider 缓存中读取 token 所产生的费用
    #[serde(rename = "cacheRead")]
    pub cache_read: f64,
    // 向 Provider 缓存写入内容的费用
    #[serde(rename = "cacheWrite")]
    pub cache_write: f64,
    // 这次请求的总费用
    pub total: f64,
}

/// 一次模型调用消耗的 token 数量和费用。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
// 用了多少 token，以及总费用是多少
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    // 一小时缓存写入了多少 token 是可选的
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u64>,
    // 模型推理过程使用的 token 数
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    pub total_tokens: u64,
    pub cost: UsageCost,
}
/*
Usage 里面的是  调度这个模型 用了多少token
UsageCost 里面的就是 对应的token是多少钱 以及总费用

*/

/// 模型停止生成的统一原因。
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    Pending,  // 流式响应尚未完成
    Stop,     // 模型正常结束
    Length,   // 模型达到最大长度限制
    ToolUse,  // 模型调用了工具
    Error,    // 模型调用过程中出现错误
    Aborted,  // 调用方主动中止模型调用
    Deferred, // 模型调用被延迟处理，后续可继续获取结果
}

/// Provider 提供的延迟响应标识，用于之后继续获取同一次响应。
/// 延迟响应的凭据
/// 这次模型请求暂时没有最终结果；
// Provider 给我们一个“句柄”；
// 之后拿着这个句柄继续查询同一次请求。
// 它不是 API Key，也不代表登录凭据。更像 C++ 异步任务中的任务 ID 或 Future 的外部句柄。
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredHandle {
    // 哪一个服务商返回的句柄 比如 openai
    pub provider: String,
    // 请求使用的模型标识 比如gpt-5
    pub model_id: String,
    // api协议或者适配器 比如 openai-responses
    pub api: String,
    /// Provider 为这次延迟请求分配的唯一标识。
    /// 后续查询时，程序会把它传回对应 Provider。
    pub id: String,
    // 句柄失效的时间，单位是 Unix 毫秒时间戳
    // 表示 Provider 规定这个句柄到某个时间后不能再查询
    // expires_at: None,   表示 Provider 没有告知过期时间
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    // 建议多久后再查询，单位是毫秒
    // poll_after_ms: Some(1_000), 意思是等待一秒钟。
    // 它不是强制等待规则，只是 Provider 给出的建议。
    pub poll_after_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    // 不同 Provider 可能还需要保存自己的额外 JSON 数据
    pub data: Option<Value>,
}

/// 错误的可序列化信息。Provider 抛出的实际错误对象不会直接保存到消息中。
/// 模型调用出错时，不能直接把 Rust 的错误对象塞进 JSON。
/// 错误对象可能包含运行时状态、函数调用信息，无法直接序列化。
//  因此提取出可保存的字段。
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticErrorInfo {
    // 错误类型名 例如 name: Some("HttpError".to_owned()),
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    // 错误描述，是唯一必填字段。
    // 比如 message: "Too many requests".to_owned(),
    // 因为任何诊断记录至少要说明“发生了什么”，所以这里不用 Option<String>。
    pub message: String,
    // 错误调用栈。它可以帮助开发者调试，但 Provider 返回的错误通常没有本地调用栈，
    // 而且某些环境不适合保存完整堆栈，所以它是可选字段。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    // 错误码
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<DiagnosticErrorCode>,
}

/// Provider 的错误代码可以是字符串或 JSON 数字。
/// 错误码有两种类型
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DiagnosticErrorCode {
    // 表示字符串错误码：
    Text(String),
    // JSON 数字错误码
    Number(Number),
    // 如果没有它，Serde 默认会在 JSON 中保留枚举变体名
}

/// 保存已脱敏的 Provider 或运行时诊断信息。
/// 一条完整诊断记录
/// 它不是“错误本身”，而是一条“这次调用发生过什么异常情况”的记录
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AssistantMessageDiagnostic {
    // Rust 里不能直接把字段叫做
    // 因为 type 是 Rust 关键字
    // 所以 Rust 内部使用：kind
    #[serde(rename = "type")]
    pub kind: String,
    // 这条诊断信息产生的时间
    // 它不是模型回复完成的时间，而是错误或恢复事件发生的时间。
    pub timestamp: u64,
    // 有些诊断只是提示信息，没有真正错误
    // 例如“正在重试”可能只需要：  error: None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<DiagnosticErrorInfo>,
    // 保存其他诊断数据。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Map<String, Value>>,
}

/// 对应原版 types.ts 中的 AssistantMessage。
///
/// 与 UserMessage 不同，助手消息还要记录模型来源、token 用量和停止原因。
/// 完整的助手消息
/*

回复内容：
role、content

模型来源：
api、provider、model、response_model、response_id
api: String：采用什么 API 协议。
pub provider: String, 哪个模型服务商。
pub model: String,请求时指定的模型名称。
pub response_model: Option<String>,实际回复时 Provider 告知的具体模型
pub response_id: Option<String>,Provider 返回的这次回复 ID。


调用统计：
usage、stop_reason、timestamp、end_turn
pub usage: Usage,本次请求消耗的 token 和费用。 必填字段  即使模型调用失败，上层也可以填零用量。
pub stop_reason: StopReason,模型这次为什么停止。

异常或等待状态：
diagnostics、deferred、error_message、raw_stop_reason
diagnostics: Option<Vec<AssistantMessageDiagnostic>>,一条助手消息可能包含多条诊断记录


*/
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub role: AssistantRole,
    pub content: Vec<AssistantContent>,
    pub api: String,
    pub provider: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<AssistantMessageDiagnostic>>,
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deferred: Option<DeferredHandle>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_turn: Option<bool>,
    pub timestamp: u64,
}

// -----------------------------------------------------------------------------
// 新增：工具结果消息类型
// 工具执行后，需要把结果和原先的 ToolCall 关联起来，交还给模型。
// -----------------------------------------------------------------------------

/// 工具结果消息的固定角色。JSON 协议中使用 "toolResult"。
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
// 工具执行结果
pub enum ToolResultRole {
    #[serde(rename = "toolResult")]
    ToolResult,
}

/// 对应原版 types.ts 中的 ToolResultMessage。
///
/// TDetails 是工具特有的附加结果类型；未指定时默认保存任意 JSON 值 Value。
/// C++ 对照：template <typename TDetails = JsonValue>。
/// 工具执行完成后交回模型的结果。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
// 模型请求调用一个工具后，程序实际执行工具，并把执行结果返回给模型。
pub struct ToolResultMessage<TDetails = Value> {
    pub role: ToolResultRole,
    /// 与之前 AssistantContent::ToolCall(ToolCall) 中 ToolCall.id 对应。
    // 对应此前模型发出的 ToolCall.id
    // 这个字段很关键，因为模型一次回复可能请求多个工具
    pub tool_call_id: String,
    pub tool_name: String,
    /// 工具只能返回文本或图片，不能返回助手的思考或新的工具调用。
    /// 工具实际返回给模型的内容。
    pub content: Vec<ToolResultContent>,
    // 工具专用的附加结果，不一定需要发送给模型作为主要内容，
    // 但程序内部可能需要保存
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<TDetails>,
    /// 工具自身的用量，不计入模型上下文的 token 用量。
    #[serde(skip_serializing_if = "Option::is_none")]
    // 执行工具本身的资源统计
    // 例如某个远程搜索工具也可能消耗 token、请求次数或费用，可以记录
    // 普通本地读取文件工具通常没有模型 token 用量：
    pub usage: Option<Usage>,
    /// 这次工具执行后新增、可供模型在下一轮调用的工具名称。
    #[serde(skip_serializing_if = "Option::is_none")]
    // 表示执行当前工具后，有哪些新工具可以在下一轮提供给模型
    pub added_tool_names: Option<Vec<String>>,
    // 工具是否执行失败
    pub is_error: bool,
    // 工具结果产生的 Unix 毫秒时间戳
    pub timestamp: u64,
}

// -----------------------------------------------------------------------------
// 新增：完整对话消息类型
// 原版 TypeScript：Message = UserMessage | AssistantMessage | ToolResultMessage。
// -----------------------------------------------------------------------------

/// 对话中允许出现的三种完整消息。
///
/// 使用 untagged 后，JSON 直接保存每种消息本身，不额外包裹 User、Assistant 或 ToolResult。
/// C++ 对照：std::variant<UserMessage, AssistantMessage, ToolResultMessage>。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConversationMessage {
    User(UserMessage),
    Assistant(Box<AssistantMessage>),
    ToolResult(Box<ToolResultMessage>),
}

impl From<UserMessage> for ConversationMessage {
    fn from(message: UserMessage) -> Self {
        Self::User(message)
    }
}

impl From<AssistantMessage> for ConversationMessage {
    fn from(message: AssistantMessage) -> Self {
        Self::Assistant(Box::new(message))
    }
}

impl From<ToolResultMessage> for ConversationMessage {
    fn from(message: ToolResultMessage) -> Self {
        Self::ToolResult(Box::new(message))
    }
}
/*
哪个工具调用？
tool_call_id

调用的是什么工具？
tool_name

工具返回了什么？
content

有没有额外内部数据？
details

工具是否消耗资源？
usage

是否新增工具？
added_tool_names

成功还是失败？
is_error

什么时候产生？
timestamp

*/
