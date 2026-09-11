//! 上下文压缩（compaction）：会话太长时，把较早的消息摘要成一段总结，只保留最近的一段。
//!
//! 原版 `packages/agent/src/harness/compaction/compaction.ts` 很完整
//! （切点、拆分回合、文件操作提取、迭代摘要）。这里做简化版：
//! 估算 token → 判断是否该压缩 → 切分「要摘要的旧消息」和「保留的近期消息」
//! → 调摘要器 → 返回结果。摘要器由调用方提供，便于测试（也便于接真实模型）。

use pi_ai::{
    // 需要遍历各种消息的内容来估算 token。
    AssistantContent,
    ConversationMessage,
    StopReason,
    ToolResultContent,
    UserContent,
    UserMessageContent,
};

/// 压缩相关的错误。   和其他错误类型同款
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactionError {
    pub message: String,
}

impl CompactionError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for CompactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// 压缩设置。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactionSettings {
    /// 是否启用自动压缩。
    pub enabled: bool,
    /// 给摘要提示与输出预留的 token。
    pub reserve_tokens: usize,
    /// 压缩后大致保留的近期 token 数。
    pub keep_recent_tokens: usize,
}

/// 默认设置（对齐原版）。
pub const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = CompactionSettings {
    enabled: true,
    reserve_tokens: 16_384,
    keep_recent_tokens: 20_000,
};

/// 粗略的字符/token 比例。
const CHARS_PER_TOKEN: usize = 4;
/// 一张图片大致折算的字符数。
const ESTIMATED_IMAGE_CHARS: usize = 4_800;

/// 估算一条消息的字符数。   实际含义：把一条消息的所有文本折算成「字符数」。
fn message_chars(message: &ConversationMessage) -> usize {
    match message {
        ConversationMessage::User(user) => match &user.content {
            // 用户消息：纯文本或内容块（图片按固定字符数折算）。
            UserMessageContent::Text(text) => text.chars().count(),
            UserMessageContent::Blocks(blocks) => blocks
                .iter()
                .map(|block| match block {
                    UserContent::Text(text) => text.text.chars().count(),
                    UserContent::Image(_) => ESTIMATED_IMAGE_CHARS,
                })
                .sum(),
        },
        ConversationMessage::Assistant(assistant) => {
            assistant // 助手消息：文本 + 思考 + 工具调用（工具名 + 参数 JSON）。
                .content
                .iter()
                .map(|block| match block {
                    AssistantContent::Text(text) => text.text.chars().count(),
                    AssistantContent::Thinking(thinking) => thinking.thinking.chars().count(),
                    AssistantContent::ToolCall(call) => {
                        call.name.chars().count()
                            + serde_json::to_string(&call.arguments)
                                .map(|json| json.chars().count())
                                .unwrap_or(0)
                    }
                })
                .sum()
        }
        ConversationMessage::ToolResult(result) => result // 工具结果：文本 + 图片。
            .content
            .iter()
            .map(|block| match block {
                ToolResultContent::Text(text) => text.text.chars().count(),
                ToolResultContent::Image(_) => ESTIMATED_IMAGE_CHARS,
            })
            .sum(),
    }
}

/// 估算一条消息的 token 数（保守估计，至少 1）。
#[must_use]
pub fn estimate_tokens(message: &ConversationMessage) -> usize {
    message_chars(message).div_ceil(CHARS_PER_TOKEN).max(1) // 非空消息至少算 1 token，避免空消息被当 0
}

/// 把用量换算成「上下文 token 总数」。
fn usage_context_tokens(usage: &pi_ai::Usage) -> usize {
    if usage.total_tokens > 0 {
        // 优先用服务端给的 total_tokens；没有就各项相加。
        usage.total_tokens as usize
    } else {
        (usage.input + usage.output + usage.cache_read + usage.cache_write) as usize
    }
}

/// 估算整个消息列表的上下文 token 数。
///
/// 优先用最近一条有效 assistant 消息的 `usage`（服务端实测），
/// 其后的消息再用字符启发式估算。
#[must_use]
pub fn estimate_context_tokens(messages: &[ConversationMessage]) -> usize {
    // 从后往前找最近一条有效的 assistant 消息，它的 usage 是服务端实测的上下文大小。
    for (index, message) in messages.iter().enumerate().rev() {
        if let ConversationMessage::Assistant(assistant) = message {
            let valid = !matches!(
                // 排除出错/中止的消息（它们的 usage 不可信）。
                assistant.stop_reason,
                StopReason::Error | StopReason::Aborted
            );
            let usage_tokens = usage_context_tokens(&assistant.usage);
            if valid && usage_tokens > 0 {
                //                  这条 usage 之后的新消息，用字符启发式再估算。
                let trailing: usize = messages[index + 1..].iter().map(estimate_tokens).sum();
                return usage_tokens + trailing; // 找到就返回 实测 + 后续估算；一条都没有就全部用启发式
            }
        }
    }
    messages.iter().map(estimate_tokens).sum()
}

/// 判断上下文是否超过压缩阈值。
#[must_use]
pub fn should_compact(
    context_tokens: usize,
    context_window: u64,
    settings: CompactionSettings,
) -> bool {
    settings.enabled // 条件：已启用 且 当前 token > 上下文窗口 − 预留。
        //                                  saturating_sub：防止下溢。
        //                                      预留 reserve_tokens 是留给「摘要提示 + 摘要输出」的空间
        && context_tokens > (context_window as usize).saturating_sub(settings.reserve_tokens)
}

/// 压缩计划：哪些消息要摘要，哪些保留。
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionPlan {
    /// 将被摘要进总结的旧消息。
    pub to_summarize: Vec<ConversationMessage>,
    /// 压缩后保留的近期消息。
    pub retained_tail: Vec<ConversationMessage>,
}

/// 从尾部往前保留，直到累计 token 超过 `keep_recent_tokens`；其余归入要摘要的部分。
///
/// 至少保留最后一条消息，避免把全部历史都摘要掉。
#[must_use]
pub fn plan_compaction(
    messages: &[ConversationMessage],
    settings: CompactionSettings,
) -> CompactionPlan {
    let mut retained_reversed = Vec::new();
    let mut tokens = 0usize;
    let mut cut = messages.len();
    //                      enumerate().rev()：反向遍历（拿到下标）
    for (index, message) in messages.iter().enumerate().rev() {
        let message_tokens = estimate_tokens(message);
        // 从尾部往前累积，直到超过 keep_recent_tokens
        // !retained_reversed.is_empty() && ...：保证至少保留一条（否则最后一条如果自己就超限会被丢）
        if !retained_reversed.is_empty() && tokens + message_tokens > settings.keep_recent_tokens {
            break;
        }
        tokens += message_tokens;
        cut = index; // cut 记录保留区的起始下标
        // messages[..cut]：要摘要的旧消息
        retained_reversed.push(message.clone());
    }
    // retained_reversed.reverse() 恢复正序。
    retained_reversed.reverse();

    CompactionPlan {
        to_summarize: messages[..cut].to_vec(),
        retained_tail: retained_reversed,
    }
}

/// 摘要用的系统提示。   系统提示：只做摘要、不继续对话
pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. \
Read a conversation between a user and an AI assistant, then produce a structured summary. \
Do NOT continue the conversation. Do NOT answer questions in it. ONLY output the summary.";

/// 摘要指令（附在对话文本后面）。
// 用户提示：规定结构化格式（Goal / Progress / Next Steps 等），并强调保留文件路径、函数名、错误信息。
pub const SUMMARIZATION_PROMPT: &str = "The messages above are a conversation to summarize. \
Create a structured context checkpoint summary that another LLM will use to continue the work.\n\n\
Use this format:\n\n\
## Goal\n[What the user is trying to accomplish]\n\n\
## Constraints & Preferences\n- [constraints or (none)]\n\n\
## Progress\n### Done\n- [x] [completed]\n### In Progress\n- [ ] [current work]\n### Blocked\n- [blockers or (none)]\n\n\
## Key Decisions\n- **[decision]**: [rationale]\n\n\
## Next Steps\n1. [ordered next steps]\n\n\
## Critical Context\n- [files, names, errors needed to continue]\n\n\
Keep each section concise. Preserve exact file paths, function names, and error messages.";

/// 把一条消息渲染成可读文本（用于放进摘要提示）。
fn render_message(message: &ConversationMessage) -> String {
    match message {
        // 把消息渲染成 role: 内容 的可读文本，供摘要模型阅读。
        // 图片用 [image] 占位；思考用 [thinking]；工具调用用 [tool_call] name(args)。
        ConversationMessage::User(user) => match &user.content {
            UserMessageContent::Text(text) => format!("user: {text}"),
            UserMessageContent::Blocks(blocks) => {
                let parts: Vec<String> = blocks
                    .iter()
                    .map(|block| match block {
                        UserContent::Text(text) => text.text.clone(),
                        UserContent::Image(_) => "[image]".to_owned(),
                    })
                    .collect();
                format!("user: {}", parts.join("\n"))
            }
        },
        ConversationMessage::Assistant(assistant) => {
            let mut parts = Vec::new();
            for block in &assistant.content {
                match block {
                    AssistantContent::Text(text) => parts.push(text.text.clone()),
                    AssistantContent::Thinking(thinking) => {
                        parts.push(format!("[thinking] {}", thinking.thinking));
                    }
                    AssistantContent::ToolCall(call) => parts.push(format!(
                        "[tool_call] {}({})",
                        call.name,
                        serde_json::to_string(&call.arguments).unwrap_or_default()
                    )),
                }
            }
            format!("assistant: {}", parts.join("\n"))
        }
        ConversationMessage::ToolResult(result) => {
            let parts: Vec<String> = result
                .content
                .iter()
                .map(|block| match block {
                    ToolResultContent::Text(text) => text.text.clone(),
                    ToolResultContent::Image(_) => "[image]".to_owned(),
                })
                .collect();
            format!("tool({}): {}", result.tool_name, parts.join("\n"))
        }
    }
}

/// 把要摘要的消息拼成一段文本，末尾附上摘要指令。
#[must_use]
pub fn build_summary_prompt(messages: &[ConversationMessage]) -> String {
    let mut output = String::new();
    for message in messages {
        output.push_str(&render_message(message));
        output.push('\n');
    }
    output.push('\n');
    output.push_str(SUMMARIZATION_PROMPT);
    output
}

/// 压缩结果。
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionResult {
    /// 生成的摘要。
    pub summary: String,
    /// 保留的近期消息。
    pub retained_tail: Vec<ConversationMessage>,
    /// 压缩前的上下文 token 估算。
    pub tokens_before: usize,
}

/// 执行压缩：切分消息、构造摘要提示、调用摘要器、返回结果。
///
/// `summarize` 接收 `(系统提示, 用户提示)`，返回摘要文本。
/// 把摘要器做成参数，既能接真实模型，也能在测试里注入固定结果。
pub fn compact(
    messages: &[ConversationMessage],
    settings: CompactionSettings,
    summarize: impl FnOnce(&str, &str) -> Result<String, CompactionError>,
) -> Result<CompactionResult, CompactionError> {
    let plan = plan_compaction(messages, settings);
    let prompt = build_summary_prompt(&plan.to_summarize);
    let summary = summarize(SUMMARIZATION_SYSTEM_PROMPT, &prompt)?;
    Ok(CompactionResult {
        summary,
        retained_tail: plan.retained_tail,
        tokens_before: estimate_context_tokens(messages),
    })
}
/*
超过阈值 → 从最新往前数出「保留的近期消息」和「要摘要的旧消息」→ 只把旧消息发给模型 →
模型返回一段摘要文本 → 用「摘要 + 近期消息」替换掉原来的全部消息。


1. estimate_context_tokens(messages)          估算当前用了多少 token
        │
2. should_compact(...) ?                      超过阈值？
        │ 是
3. plan_compaction(messages, settings)        切分：
        ├─ to_summarize   = 旧消息（会被摘要）
        └─ retained_tail  = 近期消息（原样保留）
        │
4. build_summary_prompt(to_summarize)         只把【旧消息】拼成提示
        │
5. summarize(system, prompt) ──► summary      调模型，得到【一段摘要文本】
        │
6. CompactionResult { summary, retained_tail, tokens_before }


送去模型摘要的只有旧消息；近期消息不发送，原样留着。
模型产出的是摘要文本，不是消息。


*/
