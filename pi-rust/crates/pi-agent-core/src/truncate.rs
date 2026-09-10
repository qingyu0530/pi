//! 工具输出的截断工具。
//!
//! 两个独立上限，谁先到用谁：
//! - 行数上限（默认 2000 行）
//! - 字节上限（默认 50KB）
//!
//! 从头部保留完整行，不返回半行。

/// 默认最大行数。
pub const DEFAULT_MAX_LINES: usize = 2000;
/// 默认最大字节数（50KB）。
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;

/// 触发截断的限制。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TruncatedBy {
    /// 达到行数上限。
    Lines,
    /// 达到字节上限。
    Bytes,
}

/// 截断结果。
#[derive(Clone, Debug, PartialEq)]
pub struct TruncationResult {
    /// 截断后的内容。
    pub content: String,
    /// 是否发生了截断。
    pub truncated: bool,
    /// 因哪个限制被截断；未截断时为 `None`。
    pub truncated_by: Option<TruncatedBy>,
    /// 原始内容总行数。
    pub total_lines: usize,
    /// 截断后保留的行数。
    pub output_lines: usize,
}

/// 从头部保留内容，直到遇到行数或字节上限。
///
/// `str::len()` 返回的是字节数，正合字节上限的需要。
#[must_use]
pub fn truncate_head(content: &str, max_lines: usize, max_bytes: usize) -> TruncationResult {
    let lines: Vec<&str> = content.split('\n').collect(); // split('\n') 按换行切成一段段，collect() 收集成 Vec<&str>（每个元素是指向原文的切片，不复制）。
    let total_lines = lines.len(); // 总行数
    let mut selected: Vec<&str> = Vec::new(); // 保存保留的行
    let mut byte_count = 0usize; // 已累计的字节数
    let mut truncated_by = None;
    // 同时拿到下标 index 和行 line
    for (index, line) in lines.iter().enumerate() {
        // 如果已经保留满 max_lines 行，标记「因行数截断」，break 跳出。Some(...) 把值包进 Option。
        if index >= max_lines {
            truncated_by = Some(TruncatedBy::Lines);
            break;
        }
        let separator = usize::from(!selected.is_empty()); // 除第一行外，每行前面还有一个换行符。
        if byte_count + separator + line.len() > max_bytes { // 已用字节 + 分隔符 + 本行字节」超过 max_bytes，标记「因字节截断」并跳出。
            truncated_by = Some(TruncatedBy::Bytes);
            break;
        }
        byte_count += separator + line.len(); // 累计字节数。line.len() 返回字节数（不是字符数），正是字节上限需要的。
        selected.push(line); // 把这一行收进 selected
    }

    let output_lines = selected.len(); // 实际保留的行数
    TruncationResult {
        content: selected.join("\n"), // 把保留的行用换行拼回一个字符串。
        truncated: truncated_by.is_some(),
        truncated_by,
        total_lines,
        output_lines,
    }
}
