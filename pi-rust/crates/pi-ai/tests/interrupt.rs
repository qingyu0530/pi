//! 中断（Ctrl+C）行为测试（流式部分）。
//!
//! 中断标志是进程级全局静态；本文件所有场景放在**一个测试函数**里
//! 顺序执行，避免并行测试互相看到别人的标志。集成测试文件各自是
//! 独立进程，所以也不会影响其它测试文件。

use std::io::Read;

use pi_ai::api::http::{HttpError, HttpRequest, HttpTransport};
use pi_ai::api::openai_completions::OpenAiCompletionsProvider;
use pi_ai::{
    AssistantMessageEvent, Context, InputType, Model, ModelCost, ModelCostRates, Provider,
    RequestOptions, StopReason, UserMessage, UserMessageContent, UserRole,
};

struct FakeTransport {
    body: String,
}

impl HttpTransport for FakeTransport {
    fn post(&self, _request: &HttpRequest) -> Result<Box<dyn Read>, HttpError> {
        // 模拟「永远读不完」的慢流：每次 read 只吐一点。
        Ok(Box::new(SlowReader {
            data: self.body.clone().into_bytes(),
            position: 0,
        }))
    }
}

/// 每次只吐 8 字节的 reader（模拟网络慢速分片）。
struct SlowReader {
    data: Vec<u8>,
    position: usize,
}

impl Read for SlowReader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let take = (self.data.len() - self.position).min(8).min(buffer.len());
        if take == 0 {
            return Ok(0);
        }
        buffer[..take].copy_from_slice(&self.data[self.position..self.position + take]);
        self.position += take;
        Ok(take)
    }
}

fn model() -> Model {
    Model {
        id: "gpt-4o-mini".to_owned(),
        name: "GPT-4o mini".to_owned(),
        api: "openai-completions".to_owned(),
        provider: "openai".to_owned(),
        base_url: "https://api.openai.com/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![InputType::Text],
        cost: ModelCost {
            rates: ModelCostRates {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            tiers: None,
        },
        context_window: 128_000,
        max_tokens: 1_024,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn context() -> Context {
    Context {
        system_prompt: None,
        messages: vec![
            UserMessage {
                role: UserRole::User,
                content: UserMessageContent::Text("hi".to_owned()),
                timestamp: 1,
            }
            .into(),
        ],
        tools: None,
    }
}

/// 无限长的 SSE 流：拉不完，只能靠中断结束。
fn endless_body() -> String {
    "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}

"
    .repeat(1000)
}

#[test]
fn interrupt_aborts_stream_and_clears() {
    pi_ai::interrupt::clear();
    let provider = OpenAiCompletionsProvider::new(
        Box::new(FakeTransport {
            body: endless_body(),
        }),
        "test-key",
    );
    let mut events = provider.stream(&model(), &context(), &RequestOptions::default());
    // 先拉出 start 和几个 delta（确认流真的在走）。
    assert!(matches!(
        events.next(),
        Some(AssistantMessageEvent::Start { .. })
    ));
    let mut deltas = 0;
    while deltas < 3 {
        match events.next() {
            Some(AssistantMessageEvent::TextDelta { .. }) => deltas += 1,
            // start 之后先来一个 TextStart 是正常的。
            Some(AssistantMessageEvent::TextStart { .. }) => {}
            other => panic!("expected text delta, got {other:?}"),
        }
    }

    // 模拟 Ctrl+C。
    pi_ai::interrupt::request();

    // 继续拉：流应以 Done(Aborted) 结束，不再无限。
    let mut saw_aborted_done = false;
    for event in events.by_ref() {
        if let AssistantMessageEvent::Done { reason, message } = event {
            assert_eq!(reason, StopReason::Aborted);
            assert_eq!(message.stop_reason, StopReason::Aborted);
            assert!(
                message
                    .error_message
                    .as_deref()
                    .is_some_and(|text| text.contains("中断"))
            );
            saw_aborted_done = true;
            break;
        }
    }
    assert!(saw_aborted_done, "中断后流应结束而不是无限继续");
    // 已结束的流迭代到尽头返回 None。
    assert!(events.next().is_none());

    // 中断累积的文本保留在消息里（部分输出不丢失）。
    // （start 事件的 partial 无内容；Done 里的 message 至少含已收 delta。）

    // clear 恢复。
    pi_ai::interrupt::clear();
    assert!(!pi_ai::interrupt::is_requested());

    // clear 后同样的流可以正常走完。
    let provider = OpenAiCompletionsProvider::new(
        Box::new(FakeTransport {
            body: "data: [DONE]

"
            .to_owned(),
        }),
        "test-key",
    );
    let events: Vec<AssistantMessageEvent> = provider
        .stream(&model(), &context(), &RequestOptions::default())
        .collect();
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AssistantMessageEvent::Done { .. }))
    );
}
