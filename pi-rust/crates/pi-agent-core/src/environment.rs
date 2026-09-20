//! 执行环境抽象。
//!
//! 工具不应该直接调用 `std::fs`，而是通过 `Environment` trait 访问文件系统。
//! 好处：测试时用内存实现替换真实文件系统，不需要真的建文件。
//!
//! C++ 对照：`Environment` ≈ 抽象基类；`RealEnvironment` / 测试里的假实现是子类。
//! 这与 Provider / FauxProvider 是同一套设计（依赖注入 / 打桩）。

use std::io::Write;

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

/// 目录中的一个条目。
///
/// 只保留列目录需要的两个字段：名字和「是否为目录」。
/// 不存完整路径，因为调用方已经知道父目录，拼起来即可。
///
/// C++ 对照：类似 `std::filesystem::directory_entry` 的精简版。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirEntry {
    /// 条目名（不含父目录部分）。
    pub name: String,
    /// 是否为目录。
    pub is_dir: bool,
}

/// 工具可使用的环境能力。   核心 trait
pub trait Environment {
    /// 读取文件为 UTF-8 文本。
    fn read_file(&self, path: &str) -> Result<String, EnvError>;

    /// 覆盖写入文件。
    fn write_file(&self, path: &str, content: &str) -> Result<(), EnvError>;

    /// 列出目录内容（不递归）。
    ///
    /// 路径不存在、不是目录或无权访问时返回 `Err`。
    /// C++ 对照：类似用 `std::filesystem::directory_iterator` 把条目一次性收进 `vector`。
    fn read_dir(&self, path: &str) -> Result<Vec<DirEntry>, EnvError>;

    /// 追加写入文件（文件不存在则创建）。
    ///
    /// 默认实现是「读出来 + 拼接 + 整体写回」，对内存实现足够；
    /// 真实文件系统会覆盖成真正的追加写，避免每次重写整个文件。
    fn append_file(&self, path: &str, content: &str) -> Result<(), EnvError> {
        let mut existing = self.read_file(path).unwrap_or_default();
        existing.push_str(content);
        self.write_file(path, &existing)
    }
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
    // 用操作系统的真正追加写——只写新内容，不碰已有字节
    fn append_file(&self, path: &str, content: &str) -> Result<(), EnvError> {
        // 以「创建 + 追加」模式打开，只写新内容，不重写整个文件。
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|error| EnvError::new(format!("打开文件 `{path}` 追加失败: {error}")))?;
        file.write_all(content.as_bytes())
            .map_err(|error| EnvError::new(format!("追加写入文件 `{path}` 失败: {error}")))
    }

    fn read_dir(&self, path: &str) -> Result<Vec<DirEntry>, EnvError> {
        // read_dir 返回一个迭代器，每个元素是 Result<DirEntry, io::Error>。
        let entries = std::fs::read_dir(path)
            .map_err(|error| EnvError::new(format!("读取目录 `{path}` 失败: {error}")))?;

        let mut result = Vec::new();
        for entry in entries {
            // 单个条目也可能读失败（例如权限问题），这里直接向上报错。
            let entry = entry
                .map_err(|error| EnvError::new(format!("读取目录 `{path}` 的条目失败: {error}")))?;
            // 文件名可能不是合法 UTF-8，用 lossy 转换避免报错（Unix 上文件名是任意字节）。
            let name = entry.file_name().to_string_lossy().into_owned();
            // file_type 可能失败，失败时保守地当成「不是目录」。
            let is_dir = entry
                .file_type()
                .map(|file_type| file_type.is_dir())
                .unwrap_or(false);
            result.push(DirEntry { name, is_dir });
        }
        Ok(result)
    }
}
