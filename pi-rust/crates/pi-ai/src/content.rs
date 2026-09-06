// serde 不是 Rust 自带的，是第三方库。
// Rust 把这种库称为 crate，由包管理工具 Cargo 下载和管理
// serde 负责 Rust 类型和外部数据之间的转换，serde_json 负责 JSON。

// 把 serde 提供的名字引入当前文件
// Serialize 用来支持转换成 JSON，但它本身不限定格式，也能配合其他库输出 YAML、TOML 等。
// 这种其实就是 c++ 里面的 std::sort

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// #[...] 是属性；derive 根据类型定义自动生成指定 trait 的实现。
// PartialEq 相等、不相等比较
// Copy 允许在赋值、传参时隐式复制值
// 不要把 Copy 理解成 C++ 的自定义复制构造函数。
// Rust 的 Copy 表示值可以通过简单复制来复制，不能在复制时执行你自定义的逻辑
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
// rename_all = "snake_case":告诉 Serde：转换数据时，把枚举变体名称转成下划线命名：
// Commentary  → "commentary"
// FinalAnswer → "final_answer"
#[serde(rename_all = "snake_case")]
// pub：公开这个类型，允许其他模块在可访问的路径下使用它。
// enum：定义枚举,就只能是其中的一种 要么是Commentary 要么就是FinalAnswer
// TextPhase：枚举类型名
// Commentary：一个枚举变体，表示过程说明。
// FinalAnswer：另一个枚举变体，表示最终回答。
// println! 后面的 ! 表示这是宏。{:?} 表示使用 Debug 格式输出
pub enum TextPhase {
    Commentary,
    FinalAnswer,
}

// 这个结构体存储供应商元数据，供后续请求回传使用。
// derive。作用与前面相同，但这次没有 Copy
// 原因是这个结构体包含 String。Rust 的 String 拥有自己的字符串缓冲区，
// 不能直接使用 Copy 复制，否则两个值可能错误地共同拥有同一块需要释放的内存。
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
// 字段声明顺序
// Rust：名字: 类型
// C++ ：类型 名字

// 可见性区别：只公开类型。Rust 的字段默认私有，
// 所以字段也需要单独写 pub。 C++ 的 struct 字段则默认公开。
pub struct TextSignature {
    // v 表示版本
    // 当前类型并没有保证它只能为 1，这一点以后可以根据需要加强约束。
    pub v: u8, // 对应c++的std::uint8_t v;
    // String 大致对应 C++ 的 std::string，
    // 但 Rust 的 String 保证保存有效的 UTF-8 文本。
    pub id: String,
    // 序列化时，调用 Option::is_none 检查字段；如果返回 true，就省略这个字段。
    // 没有值时：
    // {"v":1,"id":"msg_1"}
    // 不加这个省略属性时，None 通常会输出为 "phase":null。
    #[serde(skip_serializing_if = "Option::is_none")]
    // Option<T> 对应“可能存在一个 T”，可以类比 C++ 的 std::optional<T>。
    // Rust 标准库中的定义概念上是：
    // enum Option<T> {    None,    Some(T),}
    // 因此这个字段可以是 None
    // 也可以是：Some(TextPhase::FinalAnswer)
    pub phase: Option<TextPhase>,
}

// 一个纯文本内容块
// 自动生成克隆、比较、调试、序列化和反序列化实现。
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
// 序列化字段名称时使用小驼峰命名。
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    pub text: String,
    // 下一个字段为 None 时，不输出它。
    #[serde(skip_serializing_if = "Option::is_none")]
    // 可选的文本签名字符串。
    pub text_signature: Option<String>,
}

/// 这是供应商返回的推理内容块。
/// 为这个结构体生成相同的六种 trait 实现。
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
// 序列化字段名称时使用小驼峰命名。
// text_signature → textSignature
// Rust 内部仍然使用 text_signature。
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    // 保存推理文
    pub thinking: String,
    //可选签名，没有时省略
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    // 可选的删减标记，没有时省略
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted: Option<bool>,
}
/*
补充一个 C++ 开发者最容易遇到的区别：字符串赋值
这个文件里的多个结构体都包含 String，因此现在理解这点很有用。
C++：
std::string a = "hello";
std::string b = a; // 复制
// a 仍然可以使用
Rust：
let a = String::from("hello");
let b = a; // 所有权移动给 b
// println!("{}", a); // 编译错误：a 已被移动

从使用效果看，第二种更接近 C++ 的移动语义，但 Rust 会在编译期禁止你继续使用已移动的原值。
需要两份独立的数据时：
let a = String::from("hello");
let b = a.clone();

// a 和 b 都可使用
同理，TextContent 派生了 Clone：
let other = text.clone();
生成的实现会克隆它的字段，包括其中的字符串。
只是临时读取而不想移动时，用借用：
let json = serde_json::to_string(&text).unwrap();

// to_string 借用了 text，text 仍然可用。
println!("{}", text.text);
这里暂时记住三个动作就够了：
let b = a;         // 对 String 等非 Copy 类型：移动
let b = a.clone(); // 显式克隆
let b = &a;        // 借用，不取得所有权


*/

/// A base64-encoded image content block.
// 图片数据经过 Base64 编码。
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    // 保存 Base64 文本
    // 类型定义本身不检查 Base64 是否有效，也不检查图片内容是否真的符合声明的格式。
    pub data: String,
    pub mime_type: String, // 保存媒体类型，例如 "image/png"
}

// 描述模型请求执行某个工具的数据
/// A request from the model to execute a named tool.
/// 工具调用请求
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    // 是这次调用的标识，用来关联之后的执行结果
    pub id: String,
    // name 是工具名字，例如 "read_file"。
    pub name: String,
    // arguments 是工具参数对象。
    // Map<String, Value>
    /*
    Value 能表示 JSON 的多种类型
    {
    "path": "README.md",
    "offset": 10,
    "recursive": true
    }
    最外层使用 Map，保证它是对象；每个字段的值使用 Value，允许它们有不同类型。

     */
    pub arguments: Map<String, Value>,
    // 可选的供应商签名，不存在时省略。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    // 可选的命名空间，不存在时省略。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

/// Content accepted in a user message.
/// 用户消息接受的内容块
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
// 使用 type 字段区分枚举变体
#[serde(tag = "type")]
// 将内容块拆分成三种内容来枚举
// 文本 图片
// Rust 中这种 #[...] 写法叫属性（attribute）。
// 它附在结构体、枚举、函数等代码上，为编译器或宏提供额外说明
// 它与 C++ 的 #define 不同：它不是文本替换，也不是预处理指令。
// #[derive(Serialize, Deserialize)]：让 Serde 的派生宏自动生成序列化和反序列化代码。
// #[serde(tag = "type")]：告诉 Serde，JSON 使用 type 字段区分枚举变体。
// #[serde(rename = "text")]：告诉 Serde，Rust 中的 Text 在 JSON 中使用名称 "text"
// UserContent 自己只有文本、图片两个变体。
// 同一个 UserContent 值，在某一时刻只保存其中一种情况
pub enum UserContent {
    // 一个叫 Text 的枚举变体，内部携带一个 TextContent 值。
    // 把 Rust 变体 Text 的外部名称设为 "text"。
    #[serde(rename = "text")]
    Text(TextContent),
    // 一个叫 Image 的枚举变体，内部携带一个 ImageContent 值。
    // 把 Image 的外部名称设为 "image"。
    #[serde(rename = "image")]
    Image(ImageContent),
}
/*
与 tag = "type" 配合，前面的值会序列化成：
{
  "type": "text",
  "text": "你好"
}
注意分工：
- UserContent::Text 决定 "type":"text"。
- 内部 TextContent 提供 "text":"你好"。
没有额外的 "TextContent": {...} 嵌套层。
反序列化时，Serde 先读取 type，看到 "text"，就知道应当按 TextContent 的规则读取其余字段。



如何读取枚举内部的数据：match
虽然这个文件没有写 match，但理解枚举后，需要知道怎么使用它。
match &content {
    UserContent::Text(text) => {
        println!("文本内容：{}", text.text);
    }
    UserContent::Image(image) => {
        println!("图片类型：{}", image.mime_type);
    }
}
逐行解释：
match &content {
对借用的 content 做模式匹配。可以先理解成比 C++ switch 更强的分支语句。
UserContent::Text(text) =>
如果当前变体是 Text，就把内部数据的引用绑定到变量 text。
这里的 text 是新起的变量名，可以改成其他名字。
UserContent::Image(image) =>
如果是图片，就把图片数据的引用绑定到 image。
=> 分隔“匹配条件”和“执行内容”。
这种写法的重要作用是：编译器知道每个分支里的数据类型，并检查是否处理了所有情况。
C++ 使用 std::variant 时，可以通过 std::visit 或 std::get_if 完成类似操作。



*/
// 助手内容枚举
/// Content produced by an assistant message.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
// 使用 JSON 的 type 字段区分枚举
#[serde(tag = "type")]
// 文本 思考 工具调用
pub enum AssistantContent {
    // 文本变体，JSON 标记为 "text"
    #[serde(rename = "text")]
    Text(TextContent),
    // 思考变体，JSON 标记为 "thinking"。
    #[serde(rename = "thinking")]
    Thinking(ThinkingContent),
    // Rust 的 ToolCall 对应 JSON 中的 "toolCall"
    // 工具调用变体，JSON 标记为 "toolCall"。
    // 左边 ToolCall：枚举变体名
    // 括号内部 ToolCall：前面定义的结构体类型
    #[serde(rename = "toolCall")]
    ToolCall(ToolCall),
    /*
    使用时就能看出区别：
    AssistantContent::ToolCall(call)
    这里 call 是一个已经构造好的 ToolCall 结构体值。
    一个 AssistantContent 只包含一种内容
    如果一条助手消息同时含有思考和文本，使用的是多个内容块：
    let blocks: Vec<AssistantContent> = vec![
        AssistantContent::Thinking(ThinkingContent {
            thinking: String::from("先检查项目结构"),
            thinking_signature: None,
            redacted: None,
        }),
        AssistantContent::Text(TextContent {
            text: String::from("检查完成"),
            text_signature: None,
        }),
    ];
     */
}

/// Content returned after executing a tool.
/// 工具结果内容
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
// 文本 图片
// 定义公开枚举 ToolResultContent。
pub enum ToolResultContent {
    /*
    两者分别表达：
    - 用户发送了一段文本。
    - 工具执行后返回了一段文本。
    后续消息类型可以明确要求相应的内容类型。
    比如截图工具可以返回图片，读取文件工具可以返回文本。
     */
    #[serde(rename = "text")]
    Text(TextContent),
    #[serde(rename = "image")]
    Image(ImageContent),
}

/*
前半部分的结构体回答“数据具体是什么”：
TextContent
ThinkingContent
ImageContent
ToolCall
后半部分的枚举回答“某个位置允许出现哪些数据”：
UserContent       // 文本、图片
AssistantContent  // 文本、思考、工具调用
ToolResultContent // 文本、图片
而 #[derive(...)] 与 #[serde(...)] 让这些 Rust 类型能够按约定转换成 JSON。


*/
