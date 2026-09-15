//! Terminal rendering primitives for Pi.
//!
//! 这一层只负责「把文本写到终端」，不关心 Agent 逻辑。
//! 上层（coding-agent）把 `AgentEvent` 交给 [`EventRenderer`] 完成增量渲染。

use std::io::{self, Write};

use pi_agent_core::AgentEvent;
use pi_ai::AssistantMessageEvent;

/// 把文本渲染到某个输出。
/// 规定「渲染文本」的三个动作——增量写、整行写、刷新
///
/// 分成三个方法，是为了支持「流式」：`write` 不换行，可以一小段一小段追加；
/// `render_line` 写整行；`flush` 立刻把缓冲推出去。
/// C++ 对照：类似一个带 `<<` 和 `flush` 的输出流抽象。
pub trait Renderer {
    /// 写一段文本，不自动换行（用于流式增量）。
    fn write(&mut self, text: &str) -> io::Result<()>;

    /// 写一整行（自动换行）。
    fn render_line(&mut self, line: &str) -> io::Result<()>;

    /// 刷新缓冲，让已写内容立刻可见。
    fn flush(&mut self) -> io::Result<()>;
}

/// 一个把文本写到标准输出的渲染器。
#[derive(Debug, Default)]
pub struct PlainRenderer;
/// 把文本直接写到标准输出。
impl Renderer for PlainRenderer {
    fn write(&mut self, text: &str) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        write!(stdout, "{text}")?;
        stdout.flush()
    }

    fn render_line(&mut self, line: &str) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        writeln!(stdout, "{line}")
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stdout().lock().flush()
    }
}
/// 一个「有状态的翻译器」——把 AgentEvent 变成渲染动作。
/// 把 `AgentEvent` 增量渲染到某个 [`Renderer`]。
///
/// 它维护「当前是否停在一行中间」的状态：
/// - 文本/思考的 delta 用 `write` 连续追加，不换行；
/// - 遇到工具事件或消息结束时，先补换行再输出整行。
///
/// C++ 对照：类似一个把事件流翻译成打印动作的有状态翻译器。
pub struct EventRenderer {
    renderer: Box<dyn Renderer>,
    /// 是否正停在一行中间（正在流式写文本，还没换行）。
    mid_line: bool,
    /// 当前这一行是否是思考内容（用于决定要不要加 `[思考]` 前缀）。
    in_thinking: bool,
}
/// 构造函数，接收底层渲染器，两个状态标志初始为 false。
impl EventRenderer {
    /// 用给定的底层渲染器创建。
    #[must_use]
    pub fn new(renderer: Box<dyn Renderer>) -> Self {
        Self {
            renderer,
            mid_line: false,
            in_thinking: false,
        }
    }
    /// 接收一个 AgentEvent，按事件类型决定怎么渲染
    /// 处理一个 Agent 事件并渲染。
    pub fn handle(&mut self, event: &AgentEvent) -> io::Result<()> {
        match event {
            AgentEvent::MessageUpdate { event, .. } => self.handle_message_event(event),
            AgentEvent::ToolExecutionStart { tool_name, .. } => {
                self.end_line()?;
                self.renderer
                    .render_line(&format!("[工具 {tool_name} 开始]"))
            }
            AgentEvent::ToolExecutionEnd {
                tool_name,
                is_error,
                ..
            } => {
                self.end_line()?;
                let status = if *is_error { "失败" } else { "完成" };
                self.renderer
                    .render_line(&format!("[工具 {tool_name} {status}]"))
            }
            // 一条消息结束：若还停在半行，补一个换行。
            AgentEvent::MessageEnd { .. } => self.end_line(),
            _ => Ok(()),
        }
    }

    /// 处理流式消息事件：文本和思考的增量。
    fn handle_message_event(&mut self, event: &AssistantMessageEvent) -> io::Result<()> {
        match event {
            AssistantMessageEvent::TextDelta { delta, .. } => {
                // 从思考切到正文时先换行。
                if self.in_thinking {
                    self.end_line()?;
                }
                self.renderer.write(delta)?;
                self.mid_line = true;
                self.renderer.flush()
            }
            AssistantMessageEvent::ThinkingDelta { delta, .. } => {
                if !self.in_thinking {
                    self.end_line()?;
                    self.renderer.write("[思考] ")?;
                    self.in_thinking = true;
                }
                self.renderer.write(delta)?;
                self.mid_line = true;
                self.renderer.flush()
            }
            _ => Ok(()),
        }
    }

    /// 如果正停在一行中间，补一个换行。
    fn end_line(&mut self) -> io::Result<()> {
        if self.mid_line {
            self.renderer.render_line("")?;
            self.mid_line = false;
        }
        self.in_thinking = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::{EventRenderer, Renderer};
    use pi_agent_core::AgentEvent;
    use pi_ai::{
        AssistantMessage, AssistantMessageEvent, AssistantRole, StopReason, Usage, UsageCost,
    };
    use serde_json::Value;

    /// 把输出收集到共享字符串里的测试渲染器。
    #[derive(Clone, Default)]
    struct CapturingRenderer {
        buffer: Rc<RefCell<String>>,
    }

    impl Renderer for CapturingRenderer {
        fn write(&mut self, text: &str) -> std::io::Result<()> {
            self.buffer.borrow_mut().push_str(text);
            Ok(())
        }

        fn render_line(&mut self, line: &str) -> std::io::Result<()> {
            let mut buffer = self.buffer.borrow_mut();
            buffer.push_str(line);
            buffer.push('\n');
            Ok(())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn assistant_message() -> AssistantMessage {
        AssistantMessage {
            role: AssistantRole::Assistant,
            content: Vec::new(),
            api: "faux".to_owned(),
            provider: "faux".to_owned(),
            model: "faux-1".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage {
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                cache_write_1h: None,
                reasoning: None,
                total_tokens: 0,
                cost: UsageCost {
                    input: 0.0,
                    output: 0.0,
                    cache_read: 0.0,
                    cache_write: 0.0,
                    total: 0.0,
                },
            },
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
        }
    }

    fn text_delta(delta: &str) -> AgentEvent {
        AgentEvent::MessageUpdate {
            message: Box::new(assistant_message()),
            event: Box::new(AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: delta.to_owned(),
                partial: assistant_message(),
            }),
        }
    }

    fn thinking_delta(delta: &str) -> AgentEvent {
        AgentEvent::MessageUpdate {
            message: Box::new(assistant_message()),
            event: Box::new(AssistantMessageEvent::ThinkingDelta {
                content_index: 0,
                delta: delta.to_owned(),
                partial: assistant_message(),
            }),
        }
    }

    fn setup() -> (EventRenderer, Rc<RefCell<String>>) {
        let capture = CapturingRenderer::default();
        let buffer = capture.buffer.clone();
        (EventRenderer::new(Box::new(capture)), buffer)
    }

    #[test]
    fn streams_text_deltas_without_newlines() {
        let (mut renderer, buffer) = setup();

        renderer.handle(&text_delta("你好")).unwrap();
        renderer.handle(&text_delta("，世界")).unwrap();

        assert_eq!(*buffer.borrow(), "你好，世界");
    }

    #[test]
    fn tool_events_end_the_text_line() {
        let (mut renderer, buffer) = setup();

        renderer.handle(&text_delta("开始")).unwrap();
        renderer
            .handle(&AgentEvent::ToolExecutionStart {
                tool_call_id: "c1".to_owned(),
                tool_name: "read".to_owned(),
                args: Value::Null,
            })
            .unwrap();
        renderer
            .handle(&AgentEvent::ToolExecutionEnd {
                tool_call_id: "c1".to_owned(),
                tool_name: "read".to_owned(),
                result: Box::new(pi_agent_core::ToolResult::text("ok")),
                is_error: false,
            })
            .unwrap();

        assert_eq!(
            *buffer.borrow(),
            "开始\n[工具 read 开始]\n[工具 read 完成]\n"
        );
    }

    #[test]
    fn thinking_is_prefixed_and_separated_from_text() {
        let (mut renderer, buffer) = setup();

        renderer.handle(&thinking_delta("想一下")).unwrap();
        renderer.handle(&text_delta("答案")).unwrap();

        assert_eq!(*buffer.borrow(), "[思考] 想一下\n答案");
    }

    #[test]
    fn message_end_finishes_the_line() {
        let (mut renderer, buffer) = setup();

        renderer.handle(&text_delta("hi")).unwrap();
        renderer
            .handle(&AgentEvent::MessageEnd {
                message: Box::new(assistant_message().into()),
            })
            .unwrap();

        assert_eq!(*buffer.borrow(), "hi\n");
    }
}
