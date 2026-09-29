//! 与具体线上 API 交互的协议层。
//!
//! `provider.rs` 定义 Provider 抽象（一次调用变成事件流）；
//! 这一层放某个 API 的 wire format：请求 JSON 长什么样、响应 chunk 怎么读。
//! 目前有 OpenAI-compatible 的 Chat Completions 与 Anthropic Messages。

pub mod anthropic_messages;
pub mod http;
pub mod http_ureq;
pub mod openai_compat;
pub mod openai_completions;
