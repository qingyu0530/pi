//! 进程执行抽象。
//!
//! 和 `Environment` 同一套思路：工具不直接调 `std::process`，而是通过 `Shell` trait 执行命令。
//! 好处是测试时用假实现替换，不用真的启动进程。
//!
//! C++ 对照：`Shell` ≈ 抽象基类；`RealShell` / 测试里的假实现是子类。
// 把「执行一条 shell 命令」抽象成 Shell trait，并提供一个用真实进程实现的 RealShell。和 Environment 同一套设计：工具只依赖 trait，测试用假实现。本文件还定义了执行结果 ShellOutput 和执行错误 ShellError。
use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// 执行命令失败时的错误（只表示「没能跑起来」，命令本身的非 0 退出码不算错误）。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShellError {
    pub message: String,
}

impl ShellError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ShellError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// 一次命令执行的结果。
#[derive(Clone, Debug, PartialEq)]
pub struct ShellOutput {
    /// 合并后的 stdout + stderr。
    pub output: String,
    /// 正常退出的退出码；被信号杀死（含超时）时为 `None`。
    pub exit_code: Option<i32>,
    /// 是否因为超过超时时间被强制结束。
    pub timed_out: bool,
}

/// 执行 shell 命令的能力。   规定「执行命令」的能力。
pub trait Shell { 
    /// 执行一条命令，返回合并输出、退出码和是否超时。
    ///
    /// `timeout` 为 `None` 表示不限时。
    fn run(&self, command: &str, timeout: Option<Duration>) -> Result<ShellOutput, ShellError>;
}

/// 使用真实进程的实现（通过 `bash -c` 执行）。  用真实进程实现 Shell。
#[derive(Clone, Copy, Debug, Default)]
pub struct RealShell;

impl Shell for RealShell {
    fn run(&self, command: &str, timeout: Option<Duration>) -> Result<ShellOutput, ShellError> {
        // stdin 置空，stdout/stderr 用管道接出来。
        let mut child = Command::new("bash")
            .arg("-c")
            .arg(command)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| ShellError::new(format!("启动 shell 失败: {error}")))?;

        // 取出管道句柄，交给两个线程并发读取。
        // 必须并发读：管道缓冲区满了子进程会卡住，如果主线程只等退出就会死锁。
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ShellError::new("无法获取 stdout 管道"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| ShellError::new("无法获取 stderr 管道"))?;
        let stdout_handle = thread::spawn(move || read_to_string(stdout));
        let stderr_handle = thread::spawn(move || read_to_string(stderr));

        let mut timed_out = false;
        let status = match timeout {
            Some(limit) => {
                let start = Instant::now();
                // 轮询等待，超时就杀掉子进程。
                loop {
                    match child.try_wait() {
                        Ok(Some(status)) => break Some(status),
                        Ok(None) => {
                            if start.elapsed() >= limit {
                                let _ = child.kill();
                                timed_out = true;
                                break child.wait().ok();
                            }
                            thread::sleep(Duration::from_millis(20));
                        }
                        Err(error) => {
                            return Err(ShellError::new(format!("等待进程失败: {error}")));
                        }
                    }
                }
            }
            None => Some(
                child
                    .wait()
                    .map_err(|error| ShellError::new(format!("等待进程失败: {error}")))?,
            ),
        };

        // 子进程结束后管道关闭，两个读线程会自然返回。
        let stdout = stdout_handle.join().unwrap_or_default();
        let stderr = stderr_handle.join().unwrap_or_default();

        // 合并输出：stderr 非空时接在 stdout 后面。
        let mut output = stdout;
        if !stderr.is_empty() {
            if !output.is_empty() && !output.ends_with('\n') {
                output.push('\n');
            }
            output.push_str(&stderr);
        }

        Ok(ShellOutput {
            output,
            exit_code: status.and_then(|status| status.code()),
            timed_out,
        })
    }
}

/// 把 reader 读到结尾并转成字符串（非法 UTF-8 用替换字符兜底）。
fn read_to_string(mut reader: impl Read) -> String {
    let mut buffer = Vec::new();
    let _ = reader.read_to_end(&mut buffer);
    String::from_utf8_lossy(&buffer).into_owned()
}
