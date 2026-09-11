//! HTTP 传输抽象。
//!
//! 和 `Environment`（文件系统）、`Provider`（模型）一样，这里把「发 HTTP 请求」
//! 抽象成一个 trait，真实实现走网络，测试实现返回固定文本。
//!
//! C++ 对照：`HttpTransport` ≈ 抽象基类，`RealTransport` / 测试假实现是子类。

/// 一次 HTTP 请求。
#[derive(Clone, Debug, PartialEq)]
pub struct HttpRequest {
    /// 完整 URL，例如 `https://api.openai.com/v1/chat/completions`。
    pub url: String,
    /// 请求头（名字, 值）。
    pub headers: Vec<(String, String)>,
    /// 请求体（JSON 字符串）。
    pub body: String,
}

/// HTTP 传输失败时的错误。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpError {
    pub message: String,
}

impl HttpError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// 能发送 HTTP 请求并拿回响应体的传输层。
///
/// 目前返回完整响应体文本（SSE 文本），实现简单、便于测试；
/// 真正的「边收边解析」可以在以后把返回类型换成流式 reader。
pub trait HttpTransport {
    /// 发送 POST 请求，返回响应体文本。
    fn post(&self, request: &HttpRequest) -> Result<String, HttpError>;
}
