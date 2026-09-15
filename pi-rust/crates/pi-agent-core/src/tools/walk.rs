//! 目录遍历辅助：`find` 和 `grep` 共用。
//!
//! 遍历只依赖 `Environment`，不直接碰真实文件系统，所以测试里能用内存实现替换。
// 深度优先遍历一个目录树下所有文件  给 find 和 grep 复用
use crate::environment::{EnvError, Environment}; 

/// 遍历时跳过的目录名：体积大且通常不是搜索目标。  列出遍历时要跳过的目录名，避免钻进 .git、node_modules 这种又大又没用的目录。
pub(crate) const SKIP_DIRS: [&str; 2] = [".git", "node_modules"];

/// 深度优先遍历 `root` 下的所有文件。
///
/// 每遇到一个文件就调用 `visit(relative_path, file_name)`。
/// `visit` 返回 `false` 时停止整个遍历（用于提前达到结果上限）。
///
/// 用显式栈而不是递归函数：Rust 没有尾调用优化，深目录递归可能爆栈。
/// C++ 对照：等价于手动维护一个 `std::vector<std::string>` 当工作栈。
pub(crate) fn walk_files(
    env: &dyn Environment,
    root: &str,
    mut visit: impl FnMut(&str, &str) -> bool,
) -> Result<(), EnvError> {
    // 栈里存「相对 root 的目录路径」，空字符串表示 root 本身。
    let mut stack = vec![String::new()];

    'walk: while let Some(relative_dir) = stack.pop() {
        let dir = join_dir(root, &relative_dir);
        for entry in env.read_dir(&dir)? {
            let relative_path = join_path(&relative_dir, &entry.name);
            if entry.is_dir {
                if SKIP_DIRS.contains(&entry.name.as_str()) {
                    continue;
                }
                stack.push(relative_path);
            } else if !visit(&relative_path, &entry.name) {
                break 'walk;
            }
        }
    }

    Ok(())
}

/// 拼接「搜索根 + 相对目录」得到传给环境的目录路径。
///
/// 当根是 `.` 或空串时直接返回相对目录，避免产生 `./src` 这种带前缀的路径。
pub(crate) fn join_dir(root: &str, relative: &str) -> String {
    if relative.is_empty() {
        root.to_owned()
    } else if root.is_empty() || root == "." {
        relative.to_owned()
    } else {
        format!("{root}/{relative}")
    }
}

/// 拼接相对路径：空串表示「没有这一段」。
pub(crate) fn join_path(dir: &str, name: &str) -> String {
    if dir.is_empty() {
        name.to_owned()
    } else if name.is_empty() {
        dir.to_owned()
    } else {
        format!("{dir}/{name}")
    }
}
