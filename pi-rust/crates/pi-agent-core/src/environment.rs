//! 执行环境抽象。
//!
//! 工具不应该直接调用 `std::fs`，而是通过 `Environment` trait 访问文件系统。
//! 好处：测试时用内存实现替换真实文件系统，不需要真的建文件。
//!
//! C++ 对照：`Environment` ≈ 抽象基类；`RealEnvironment` / 测试里的假实现是子类。
//! 这与 Provider / FauxProvider 是同一套设计（依赖注入 / 打桩）。

/// 环境操作失败时的错误。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnvError {
    pub message: String,
}
/// 构造函数
impl EnvError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
/// 实现 Display（打印）
impl std::fmt::Display for EnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// 工具可使用的环境能力。   核心 trait
pub trait Environment {
    /// 读取文件为 UTF-8 文本。
    fn read_file(&self, path: &str) -> Result<String, EnvError>;

    /// 覆盖写入文件。
    fn write_file(&self, path: &str, content: &str) -> Result<(), EnvError>;
}

/// 使用真实文件系统的实现。 用真实文件系统实现 trait
#[derive(Clone, Copy, Debug, Default)]
pub struct RealEnvironment;

impl Environment for RealEnvironment {
    fn read_file(&self, path: &str) -> Result<String, EnvError> {
        std::fs::read_to_string(path)
            .map_err(|error| EnvError::new(format!("读取文件 `{path}` 失败: {error}")))
    }

    fn write_file(&self, path: &str, content: &str) -> Result<(), EnvError> {
        std::fs::write(path, content)
            .map_err(|error| EnvError::new(format!("写入文件 `{path}` 失败: {error}")))
    }
}
