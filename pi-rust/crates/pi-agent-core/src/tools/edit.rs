//! `edit` 工具：对文件做精确字符串替换。

use pi_ai::ToolCall; // 模型的工具调用类型
use serde_json::Value;

use crate::environment::Environment;
use crate::tool::{AgentTool, ToolError, ToolResult};

/// 一次精确替换。
struct Replacement {
    /// 要被替换的原文（必须在原文件中唯一出现）。
    old_text: String,
    /// 替换后的文本。
    new_text: String,
}

/// 编辑文件的工具。
pub struct EditTool {
    env: Box<dyn Environment>,
}
/// 构造
impl EditTool {
    #[must_use]
    pub fn new(env: Box<dyn Environment>) -> Self {
        Self { env }
    }
}

impl AgentTool for EditTool {
    fn name(&self) -> &str { // 工具名
        "edit"
    }
    // 给模型看的说明。
    fn description(&self) -> &str {
        "用精确文本替换来编辑一个文件。每个 edits[].oldText 必须在原文件中唯一出现，\
         且各替换区域不能重叠。所有替换都基于原始文件内容，不是逐条叠加。"
    }
    // path 是字符串；edits 是数组，数组元素是带 oldText/newText 的对象。这是发给模型的参数格式
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "要编辑的文件路径" },
                "edits": {
                    "type": "array",
                    "description": "一组精确替换。oldText 必须唯一且互不重叠。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": { "type": "string", "description": "要被替换的精确文本" },
                            "newText": { "type": "string", "description": "替换后的文本" }
                        },
                        "required": ["oldText", "newText"]
                    }
                }
            },
            "required": ["path", "edits"]
        })
    }
    // execute 接收工具调用，返回结果或错误
    fn execute(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let path = call
            .arguments
            .get("path") // 取必填字符串 path。链式：get → and_then(as_str) → ok_or_else，末尾 ? 失败即返回。
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::new("缺少字符串参数 path"))?;
        let edits_value = call
            .arguments
            .get("edits") // 取 edits（这里先只取到 &Value，还没解析），缺失就报错。
            .ok_or_else(|| ToolError::new("缺少参数 edits"))?;



        let edits = parse_edits(edits_value)?; // 调用解析、读写、返回
        let content = self // 读文件
            .env
            .read_file(path)
            .map_err(|error| ToolError::new(error.message))?;
        let new_content = apply_edits(&content, &edits)?; // 应用替换，得到新内容
        self.env // 写回文件
            .write_file(path, &new_content)
            .map_err(|error| ToolError::new(error.message))?;
        // 返回成功文本
        Ok(ToolResult::text(format!(
            "已在 {path} 中替换 {} 处",
            edits.len()
        )))
    }
}
/// 把 JSON 的 edits 数组解析成 `Replacement` 列表。
fn parse_edits(value: &Value) -> Result<Vec<Replacement>, ToolError> {
    let array = value
        .as_array() // 要求是数组，否则报错。
        .ok_or_else(|| ToolError::new("edits 必须是数组"))?;
    if array.is_empty() { // 空数组也报错
        return Err(ToolError::new("edits 至少要有一项"));
    }

    let mut edits = Vec::new();
    for (index, item) in array.iter().enumerate() {
        let old_text = item
            .get("oldText")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::new(format!("edits[{index}] 缺少字符串 oldText")))?;
        let new_text = item
            .get("newText")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::new(format!("edits[{index}] 缺少字符串 newText")))?;
        edits.push(Replacement {
            old_text: old_text.to_owned(),
            new_text: new_text.to_owned(),
        });
    }
    Ok(edits)
}

/// 在原文上应用所有替换，返回新内容。
///
/// 规则：
/// - 每个 `old_text` 必须在**原文**中恰好出现一次（唯一）。
/// - 各替换区域不能重叠。
/// - 从后往前替换，避免前面的替换改变后面区域的字节下标。

// apply_edits 先在原文里为每个 oldText 定位（要求唯一），
// 把替换区间记下、排序、检查不重叠，然后从后往前逐段替换
fn apply_edits(content: &str, edits: &[Replacement]) -> Result<String, ToolError> {
    // 收集每处替换：起始下标、结束下标、替换文本。
    let mut spans: Vec<(usize, usize, &str)> = Vec::new();
    for edit in edits {
        let mut matches = content.match_indices(&edit.old_text); // 逐个给出 (字节下标, 匹配到的 &str
        let first = matches // 取第一个匹配；一个都没有 → 报「未找到」
            .next()
            .ok_or_else(|| ToolError::new(format!("未找到 oldText:\n{}", edit.old_text)))?;
        if matches.next().is_some() { // 再取第二个；还有 → 说明出现多次，报「不唯一」。（is_some() 是「还有下一个」）
            return Err(ToolError::new(format!(
                "oldText 不唯一（出现多次）:\n{}",
                edit.old_text
            )));
        }
        let start = first.0; // 起止下标
        let end = start + edit.old_text.len(); // old_text.len() 返回字节长度，start + 字节长度 = 结束位置。
        spans.push((start, end, edit.new_text.as_str())); // 记录这一处替换
    }

    // 按起始位置排序，并检查相邻区域是否重叠。
    spans.sort_by_key(|span| span.0); // 按起始位置排序
    // windows(2) 得到所有相邻的一对 (pair[0], pair[1])。
    // 若前一处的结束 pair[0].1 大于后一处的开始 pair[1].0，说明重叠 → 报错。
    for pair in spans.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(ToolError::new("编辑区域重叠，请合并为一次替换"));
        }
    }

    // 从后往前替换，这样前面的下标不受影响。
    let mut result = content.to_owned(); // 复制原文成可变的 result
    for (start, end, new_text) in spans.into_iter().rev() { // 从后往前遍历替换
        // 为什么要从后往前：替换会改变后面文本的下标；先改后面的，前面的下标就不会被影响。
        result.replace_range(start..end, new_text);
    }
    Ok(result)
}
