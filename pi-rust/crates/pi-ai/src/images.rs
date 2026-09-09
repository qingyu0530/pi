//! 图片生成相关类型。
//!
//! 前面的 content.rs / message.rs / context.rs 都是“对话/文本模型”用的；
//! 图片生成（images API）走另一条调用路径，返回的是图片内容块。
//! 原版：types.ts 第 469-488 行。

use serde::{Deserialize, Serialize};

use crate::content::UserContent;
use crate::message::Usage;

/// 图片生成请求的上下文输入。
/// 原版：export interface ImagesContext { input: ImagesInputContent[] }
/// ImagesInputContent = TextContent | ImageContent，和已定义的 UserContent 完全相同，
/// 所以这里直接复用 UserContent，不用再定义一遍。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImagesContext {
    /// 输入内容块，顺序保存。
    pub input: Vec<UserContent>,
}

/// 图片生成的停止原因。
/// 原版：export type ImagesStopReason = "stop" | "error" | "aborted";
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ImagesStopReason {
    #[serde(rename = "stop")]
    Stop,
    #[serde(rename = "error")]
    Error,
    #[serde(rename = "aborted")]
    Aborted,
}

/// 图片生成返回的助手结果。
/// 原版：export interface AssistantImages
/// 与 AssistantMessage 结构相似，但没有内容“角色”，而是 output 里的图片内容块。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantImages {
    /// 采用的图片生成 API 协议，例如 "openrouter-images"。
    pub api: String,
    /// 哪个模型服务商。
    pub provider: String,
    /// 请求时指定的模型名称。
    pub model: String,
    /// 生成的图片内容块，顺序保存。
    pub output: Vec<UserContent>,
    /// Provider 返回的这次响应 ID。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// 这次请求的 token 用量和费用。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// 模型这次为什么停止。
    pub stop_reason: ImagesStopReason,
    /// 出错时的错误信息。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// 结果产生的 Unix 毫秒时间戳。
    pub timestamp: u64,
}
