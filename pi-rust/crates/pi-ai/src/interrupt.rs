//! 全局中断标志。
//!
//! Ctrl+C 的信号处理器在别的线程上把标志置位；
//! 各 Provider 的流式迭代器和 Agent 的工具循环在安全点检查它，
//! 检测到中断就走「提前收尾」路径（停止原因 = Aborted），不再继续。
//!
//! 为什么用全局静态而不是传参：中断是**进程级**的横切关注点，
//! 穿透 Provider → 迭代器 → Agent 好几层；一个原子布尔最简单。
//! C++ 对照：`static std::atomic<bool> g_interrupted;`。
//!
//! 注意：测试里用完必须 `clear()`，否则会泄漏给同进程的其它测试
//! （不同集成测试文件是独立进程，互不影响）。

use std::sync::atomic::{AtomicBool, Ordering};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// 请求中断（信号处理器调用）。
pub fn request() {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

/// 清除中断标志（每轮开始前调用）。
pub fn clear() {
    INTERRUPTED.store(false, Ordering::SeqCst);
}

/// 是否已请求中断。
#[must_use]
pub fn is_requested() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}
