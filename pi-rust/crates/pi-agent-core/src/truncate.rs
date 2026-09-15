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
/// grep 匹配行的最大字符数。
pub const GREP_MAX_LINE_LENGTH: usize = 500;

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
        if byte_count + separator + line.len() > max_bytes {
            // 已用字节 + 分隔符 + 本行字节」超过 max_bytes，标记「因字节截断」并跳出。
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

/// 从尾部保留内容，直到遇到行数或字节上限。
///
/// 和 `truncate_head` 相反：命令输出往往「结尾才重要」（错误、结果），
/// 所以保留末尾。同样只保留完整行，不返回半行。
#[must_use]
pub fn truncate_tail(content: &str, max_lines: usize, max_bytes: usize) -> TruncationResult {
    let lines: Vec<&str> = content.split('\n').collect();
    let total_lines = lines.len();
    let mut selected: Vec<&str> = Vec::new(); // 先按「从后往前」收集
    let mut byte_count = 0usize;
    let mut truncated_by = None;

    for (index, line) in lines.iter().rev().enumerate() {
        if index >= max_lines {
            truncated_by = Some(TruncatedBy::Lines);
            break;
        }
        let separator = usize::from(!selected.is_empty());
        if byte_count + separator + line.len() > max_bytes {
            truncated_by = Some(TruncatedBy::Bytes);
            break;
        }
        byte_count += separator + line.len();
        selected.push(line);
    }

    // 收集时是从后往前，拼回时要反过来恢复原顺序。
    selected.reverse();

    let output_lines = selected.len();
    TruncationResult {
        content: selected.join("\n"),
        truncated: truncated_by.is_some(),
        truncated_by,
        total_lines,
        output_lines,
    }
}

/// 把单行截断到 `max_chars` 个字符，超出时追加 `... [truncated]`。  超出就截断并加后缀
///
/// 返回 `(截断后的文本, 是否发生了截断)`。
/// 用 `chars().count()` 按 Unicode 字符数（不是字节数）计算。
#[must_use]
pub fn truncate_line(line: &str, max_chars: usize) -> (String, bool) {
    if line.chars().count() <= max_chars {
        (line.to_owned(), false)
    } else {
        let kept: String = line.chars().take(max_chars).collect();
        (format!("{kept}... [truncated]"), true)
    }
}

#[cfg(test)]
mod tests {
    use super::{TruncatedBy, truncate_tail};

    #[test]
    fn tail_keeps_last_lines() {
        let result = truncate_tail("a\nb\nc\nd", 2, 1024);
        assert_eq!(result.content, "c\nd");
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(result.total_lines, 4);
        assert_eq!(result.output_lines, 2);
    }

    #[test]
    fn tail_keeps_last_bytes() {
        // 每行 3 字节 + 换行；给 4 字节只能容纳最后一行 "cc"。
        let result = truncate_tail("aa\nbb\ncc", 100, 4);
        assert_eq!(result.content, "cc");
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
    }

    #[test]
    fn tail_without_truncation_returns_all() {
        let result = truncate_tail("a\nb", 100, 1024);
        assert_eq!(result.content, "a\nb");
        assert!(!result.truncated);
        assert_eq!(result.truncated_by, None);
    }
}
