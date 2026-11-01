//! 进程执行抽象。
//!
//! 和 `Environment` 同一套思路：工具不直接调 `std::process`，而是通过 `Shell` trait 执行命令。
//! 好处是测试时用假实现替换，不用真的启动进程。
//!
//! Windows 说明（对齐原版 `utils/shell.ts` / `harness/env/nodejs.ts`）：
//! - bash 解析顺序：Git Bash 已知位置 → PATH 上的 bash.exe → 兜底 `bash`；
//! - 解析到 WSL 旧版启动器（System32/Sysnative 的 bash.exe）时，命令经 stdin
//!   传给 `bash -s`（argv 传输会有转义问题）；
//! - 解析到 WSL bash（含 WindowsApps 商店桩）时，命令里的 Windows 盘符路径
//!   （`E:\x` / `E:/x`）会先转换成 `/mnt/e/x`，否则 `ls E:\...` 在 WSL 里找不到路径。
//!
//! C++ 对照：`Shell` ≈ 抽象基类；`RealShell` / `PowerShellShell` / 测试里的假实现是子类。

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

use regex::Regex;

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

/// 启动进程并收集输出（bash / powershell 共用的执行核）。
///
/// `stdin_text` 为 `Some` 时把文本写进孩子的 stdin；为 `None` 时置空 stdin。
fn run_process(
    program: &str,
    args: &[&str],
    stdin_text: Option<&str>,
    timeout: Option<Duration>,
) -> Result<ShellOutput, ShellError> {
    // stdin 按需接管道，stdout/stderr 用管道接出来。
    let mut child = Command::new(program)
        .args(args)
        .stdin(if stdin_text.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| ShellError::new(format!("启动 {program} 失败: {error}")))?;

    // 命令走 stdin：写完立刻丢掉句柄关掉管道，孩子进程读到 EOF 才会结束。
    if let Some(text) = stdin_text {
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(text.as_bytes());
        }
    }

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

/// 把 reader 读到结尾并转成字符串（非法 UTF-8 用替换字符兜底）。
fn read_to_string(mut reader: impl Read) -> String {
    let mut buffer = Vec::new();
    let _ = reader.read_to_end(&mut buffer);
    String::from_utf8_lossy(&buffer).into_owned()
}

/// 解析后的 bash 启动配置。
#[derive(Clone, Debug)]
struct BashConfig {
    /// bash 可执行文件（Windows 上是完整路径，Unix 上就是 "bash"）。
    program: String,
    /// WSL 旧版启动器：命令经 stdin 传给 `bash -s`。
    stdin_transport: bool,
    /// WSL bash：命令里的盘符路径要先转换成 /mnt 形式。
    translate_paths: bool,
}

/// 解析 bash 启动配置（进程级缓存，首次使用时决定一次）。
fn bash_config() -> &'static BashConfig {
    static CONFIG: OnceLock<BashConfig> = OnceLock::new();
    CONFIG.get_or_init(resolve_bash)
}

/// 决定用哪个 bash：Windows 上 Git Bash 优先，其次 PATH 上的 bash.exe。
fn resolve_bash() -> BashConfig {
    #[cfg(windows)]
    {
        // 1. Git Bash 的已知安装位置（对 Windows 路径最友好）。
        let git_bash_candidates = [
            std::env::var("ProgramFiles")
                .ok()
                .map(|dir| format!("{dir}\\Git\\bin\\bash.exe")),
            std::env::var("ProgramFiles(x86)")
                .ok()
                .map(|dir| format!("{dir}\\Git\\bin\\bash.exe")),
        ];
        for candidate in git_bash_candidates.into_iter().flatten() {
            if std::fs::metadata(&candidate).is_ok() {
                return BashConfig {
                    program: candidate,
                    stdin_transport: false,
                    translate_paths: false,
                };
            }
        }

        // 2. PATH 上的 bash.exe（Cygwin / MSYS2 / WSL 等）。
        if let Some(found) = find_executable_on_path("bash.exe") {
            return BashConfig {
                stdin_transport: is_legacy_wsl_bash(&found),
                translate_paths: is_wsl_bash(&found),
                program: found,
            };
        }

        // 3. 兜底：直接叫 bash（spawn 失败时报错信息由调用方看到）。
        BashConfig {
            program: "bash".to_owned(),
            stdin_transport: false,
            translate_paths: false,
        }
    }
    #[cfg(not(windows))]
    {
        BashConfig {
            program: "bash".to_owned(),
            stdin_transport: false,
            translate_paths: false,
        }
    }
}

/// 在 PATH 上找可执行文件（`where.exe` 的第一个真实存在的结果）。
#[cfg(windows)]
fn find_executable_on_path(name: &str) -> Option<String> {
    let output = Command::new("where.exe")
        .arg(name)
        .stdin(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let first = text.trim().lines().next()?.trim();
    if first.is_empty() {
        return None;
    }
    // where 可能列出并不存在的路径，验证一下。
    std::fs::metadata(first).ok()?;
    Some(first.to_owned())
}

/// 是否 WSL 旧版启动器（System32 / Sysnative 的 bash.exe）。
#[cfg(windows)]
fn is_legacy_wsl_bash(path: &str) -> bool {
    let normalized = path.replace('/', "\\").to_lowercase();
    normalized.ends_with("\\windows\\system32\\bash.exe")
        || normalized.ends_with("\\windows\\sysnative\\bash.exe")
}

/// 是否 WSL bash（含 WindowsApps 商店桩和旧版启动器）。
#[cfg(windows)]
fn is_wsl_bash(path: &str) -> bool {
    let normalized = path.replace('/', "\\").to_lowercase();
    normalized.ends_with("\\windowsapps\\bash.exe") || is_legacy_wsl_bash(path)
}

/// 把命令文本里的 Windows 盘符路径（`E:\a\b`、`E:/a/b`）转换成 WSL 的 `/mnt/e/a/b`。
///
/// 启发式转换（只在解析到 WSL bash 时启用）：
/// - 只处理「单字母盘符 + 冒号 + 分隔符」开头的连续路径段；
/// - 盘符前的字符不能是字母数字（避开 `http://e` 这类误匹配）；
/// - 路径段遇到空白、引号、`; & | < > ( )`、`$`、反引号就停（那些是 shell 语法）。
fn translate_windows_paths(command: &str) -> String {
    static WINDOWS_PATH: OnceLock<Regex> = OnceLock::new();
    let pattern = WINDOWS_PATH.get_or_init(|| {
        // 分组：1 = 盘符前的前缀字符（或行首空串），2 = 盘符，3 = 路径剩余部分。
        Regex::new(r#"(?i)(^|[^A-Za-z0-9_\\/:.])([A-Za-z]):[\\/]([^;&|<>()\s'"`$]*)"#).unwrap()
    });
    pattern
        .replace_all(command, |caps: &regex::Captures| {
            let prefix = &caps[1];
            let drive = caps[2].to_lowercase();
            let rest = caps[3].replace('\\', "/");
            format!("{prefix}/mnt/{drive}/{rest}")
        })
        .into_owned()
}

/// 用真实 bash 进程执行的实现。
///
/// Windows 上按 `resolve_bash` 的结果选 bash；WSL 时自动转换盘符路径。
#[derive(Clone, Copy, Debug, Default)]
pub struct RealShell;

impl Shell for RealShell {
    fn run(&self, command: &str, timeout: Option<Duration>) -> Result<ShellOutput, ShellError> {
        let config = bash_config();
        let translated;
        let command = if config.translate_paths {
            translated = translate_windows_paths(command);
            translated.as_str()
        } else {
            command
        };
        if config.stdin_transport {
            run_process(&config.program, &["-s"], Some(command), timeout)
        } else {
            run_process(&config.program, &["-c", command], None, timeout)
        }
    }
}

/// 用 Windows PowerShell 执行命令的实现（对齐原版 powershell 工具）。
///
/// 优先 PowerShell 7（pwsh.exe），回退系统自带 powershell.exe；
/// 每条命令前加一段设置 UTF-8 输出编码的前缀，避免非 ASCII 输出乱码。
#[derive(Clone, Copy, Debug, Default)]
pub struct PowerShellShell;

/// PowerShell 命令前缀：把控制台输出编码切到 UTF-8。
const UTF8_OUTPUT_PREFIX: &str =
    "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\n";

impl Shell for PowerShellShell {
    fn run(&self, command: &str, timeout: Option<Duration>) -> Result<ShellOutput, ShellError> {
        #[cfg(windows)]
        {
            let script = format!("{UTF8_OUTPUT_PREFIX}{command}");
            run_process(
                powershell_program(),
                &[
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    &script,
                ],
                None,
                timeout,
            )
        }
        #[cfg(not(windows))]
        {
            let _ = (command, timeout);
            Err(ShellError::new("powershell 工具只在 Windows 上可用"))
        }
    }
}

/// 解析 PowerShell 可执行文件（pwsh 优先），进程级缓存。
#[cfg(windows)]
fn powershell_program() -> &'static str {
    static PROGRAM: OnceLock<String> = OnceLock::new();
    PROGRAM
        .get_or_init(|| {
            find_executable_on_path("pwsh.exe")
                .or_else(|| find_executable_on_path("powershell.exe"))
                .unwrap_or_else(|| "powershell.exe".to_owned())
        })
        .as_str()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{PowerShellShell, RealShell, Shell, bash_config, translate_windows_paths};

    #[test]
    fn translates_drive_letter_paths() {
        assert_eq!(
            translate_windows_paths("ls E:\\vscode\\pi"),
            "ls /mnt/e/vscode/pi"
        );
        assert_eq!(
            translate_windows_paths("cd E:/vscode/pi && cargo test"),
            "cd /mnt/e/vscode/pi && cargo test"
        );
        assert_eq!(
            translate_windows_paths("cat \"C:\\temp\\文件.txt\""),
            "cat \"/mnt/c/temp/文件.txt\""
        );
        // 盘符根：E:\ -> /mnt/e/。
        assert_eq!(translate_windows_paths("cd E:\\"), "cd /mnt/e/");
        // 多个盘符一起换。
        assert_eq!(
            translate_windows_paths("cp E:\\a D:\\b"),
            "cp /mnt/e/a /mnt/d/b"
        );
        // 大写盘符转小写挂载点。
        assert_eq!(translate_windows_paths("ls C:/Users"), "ls /mnt/c/Users");
    }

    #[test]
    fn leaves_urls_and_bare_drives_alone() {
        // URL 不动：盘符前的字符是字母/斜杠时不匹配。
        assert_eq!(
            translate_windows_paths("curl http://example.com/a"),
            "curl http://example.com/a"
        );
        // 相对路径、环境变量不动。
        assert_eq!(translate_windows_paths("ls ./src $HOME"), "ls ./src $HOME");
        // 裸盘符（没有跟分隔符）不动。
        assert_eq!(translate_windows_paths("cd E:"), "cd E:");
    }

    #[cfg(windows)]
    #[test]
    fn classifies_wsl_bash_paths() {
        use super::{is_legacy_wsl_bash, is_wsl_bash};

        // 旧版启动器：System32 / Sysnative。
        assert!(is_legacy_wsl_bash("C:\\Windows\\System32\\bash.exe"));
        assert!(is_legacy_wsl_bash("C:\\windows\\sysnative\\bash.exe"));
        // 商店桩：WindowsApps（是 WSL，但不是旧版）。
        assert!(!is_legacy_wsl_bash(
            "C:\\Users\\u\\AppData\\Local\\Microsoft\\WindowsApps\\bash.exe"
        ));
        assert!(is_wsl_bash(
            "C:\\Users\\u\\AppData\\Local\\Microsoft\\WindowsApps\\bash.exe"
        ));
        assert!(is_wsl_bash("C:\\Windows\\System32\\bash.exe"));
        // Git Bash 两者都不是。
        assert!(!is_wsl_bash("C:\\Program Files\\Git\\bin\\bash.exe"));
    }

    #[cfg(windows)]
    #[test]
    fn powershell_shell_runs_write_output() {
        let shell = PowerShellShell;
        let output = shell
            .run("Write-Output 'pi-ps-smoke'", Some(Duration::from_secs(30)))
            .unwrap();
        assert_eq!(output.exit_code, Some(0));
        assert!(output.output.contains("pi-ps-smoke"));
    }

    #[cfg(not(windows))]
    #[test]
    fn powershell_shell_rejects_non_windows() {
        let error = PowerShellShell.run("Get-Date", None).unwrap_err();
        assert!(error.message.contains("Windows"));
    }

    /// 本机解析到 WSL bash 时，盘符路径转换后应真的能列出文件；
    /// Git Bash / 其他 bash 的机器上跳过。
    #[cfg(windows)]
    #[test]
    fn wsl_bash_lists_windows_drive_path() {
        if !bash_config().translate_paths {
            return;
        }
        let dir = std::env::temp_dir().join("pi-wsl-path-test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("hello.txt"), "hi").unwrap();
        let win_path = dir.to_string_lossy().into_owned();

        let output = RealShell
            .run(&format!("ls \"{win_path}\""), Some(Duration::from_secs(30)))
            .unwrap();

        assert_eq!(output.exit_code, Some(0));
        assert!(output.output.contains("hello.txt"));
    }
}
