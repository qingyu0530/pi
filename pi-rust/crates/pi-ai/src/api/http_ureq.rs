//! 基于 `ureq` 的真实 HTTP 传输实现。
//!
//! `ureq` 是纯 Rust 的阻塞式 HTTP 客户端（默认启用 rustls TLS），
//! 与本项目同步的设计一致。

use std::time::Duration;

use ureq::Agent;

use crate::api::http::{HttpError, HttpRequest, HttpTransport};
// ureq 的连接池对象，复用 TCP/TLS 连接，比每次新建快。
/// 用 `ureq` 实现的传输层。
///
/// 内部持有一个 `Agent`：它复用连接池，并设置了整体超时。
pub struct UreqTransport {
    agent: Agent,
}

impl UreqTransport {
    /// 创建默认传输层（整体超时 120 秒）。
    #[must_use]
    pub fn new() -> Self {
        let config = Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(120))) // 整体超时 120 秒（含连接、发送、接收）。没有超时的话，网络卡住会永久挂起。
            .build();
        Self {
            agent: Agent::new_with_config(config),
        }
    }
}

impl Default for UreqTransport {
    fn default() -> Self {
        Self::new()
    }
}
// 这就是「把 HttpRequest 真正发出去、拿回文本」的地方
impl HttpTransport for UreqTransport {
    fn post(&self, request: &HttpRequest) -> Result<String, HttpError> {
        // 逐个加请求头。`header` 是 `self -> Self` 的建造者方法，所以要重新赋值。
        let mut builder = self.agent.post(&request.url);
        for (name, value) in &request.headers {
            builder = builder.header(name.as_str(), value.as_str());
        }

        let mut response = builder
            .send(request.body.as_str())
            .map_err(|error| HttpError::new(format!("HTTP 请求失败: {error}")))?;

        response
            .body_mut()
            .read_to_string()
            .map_err(|error| HttpError::new(format!("读取响应体失败: {error}")))
    }
}
