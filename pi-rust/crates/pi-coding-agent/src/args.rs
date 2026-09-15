//! 命令行参数解析。
//!   
//! 手写解析，不引入额外依赖；只覆盖当前 Rust 版本支持的能力子集。   把命令行 argv 解析成一个 Args 结构，供 main 决定用哪个模型、会话存哪、是不是非交互模式等。手写解析，不依赖第三方参数库。
//! C++ 对照：类似自己写一个 `argv` 循环，而不是用现成的参数库。 

/// `--list-models` 的两种形式。   表达 --list-models 的两种形态——不带搜索词、带搜索词。
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ListModels {
    /// 不带搜索词，列出全部。
    All,
    /// 带一个子串搜索词。
    Search(String),
}

/// 解析后的命令行参数。保存所有解析结果。每个字段对应一类选项。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Args {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub system_prompt: Option<String>,
    pub session: Option<String>,
    pub no_session: bool,
    pub print: bool,
    pub list_models: Option<ListModels>,
    pub thinking: Option<String>,
    pub help: bool,
    pub version: bool,
    /// 位置参数：要发给模型的初始消息。
    pub messages: Vec<String>,
}

/// 解析命令行参数（不含程序名）。
/// 从左到右扫描参数数组，把每个选项写进 Args。遇到 -- 停止解析选项，剩下全当消息。遇到未知选项或缺少值返回错误。
/// 出错时返回错误信息。
pub fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut result = Args::default();
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        match arg {
            // `--` 之后全部当位置参数，不再当选项。
            "--" => {
                result.messages.extend(args[index + 1..].iter().cloned());
                break;
            }
            "--help" | "-h" => result.help = true,
            "--version" | "-v" => result.version = true,
            "--print" | "-p" => result.print = true,
            "--no-session" => result.no_session = true,
            "--list-models" => {
                // 后面若是普通词（不以 '-' 开头）就当作搜索词。
                let next = args.get(index + 1);
                if let Some(value) = next.filter(|value| !value.starts_with('-')) {
                    result.list_models = Some(ListModels::Search(value.clone()));
                    index += 1;
                } else {
                    result.list_models = Some(ListModels::All);
                }
            }
            "--provider" => result.provider = Some(take_value(args, &mut index, "--provider")?),
            "--model" => result.model = Some(take_value(args, &mut index, "--model")?),
            "--api-key" => result.api_key = Some(take_value(args, &mut index, "--api-key")?),
            "--system-prompt" => {
                result.system_prompt = Some(take_value(args, &mut index, "--system-prompt")?);
            }
            "--session" => result.session = Some(take_value(args, &mut index, "--session")?),
            "--thinking" => result.thinking = Some(take_value(args, &mut index, "--thinking")?),
            other if other.starts_with('-') => return Err(format!("未知选项: {other}")),
            other => result.messages.push(other.to_owned()),
        }
        index += 1;
    }
    Ok(result)
}

/// 取当前选项后面的值；没有就报错。
fn take_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    let value = args
        .get(*index + 1)
        .ok_or_else(|| format!("{flag} 需要一个值"))?;
    *index += 1;
    Ok(value.clone())
}

/// 帮助文本。  用 format! 把版本号插进去
#[must_use]
pub fn help_text() -> String {
    format!(
        "Pi Rust {version} - Rust 重写的 Pi coding agent

用法:
  pi [选项] [--] [消息...]

选项:
  --provider <name>        模型 provider（默认 openai，或环境变量 PI_PROVIDER）
  --model <id>             模型 id（默认 gpt-4o-mini，或环境变量 PI_MODEL）
  --api-key <key>          API Key（默认读 OPENAI_API_KEY）
  --system-prompt <text>   系统提示
  --thinking <level>       思考级别: off, minimal, low, medium, high, xhigh, max
  --session <path>         会话文件（默认 pi-session.jsonl，或环境变量 PI_SESSION）
  --no-session             不保存会话
  --print, -p              非交互：处理消息后退出
  --list-models [search]   列出可用模型（可选子串过滤）
  --help, -h               显示帮助
  --version, -v            显示版本
  --                       后面的参数都当作消息
",
        version = env!("CARGO_PKG_VERSION")
    )
}
// 验证解析行为
#[cfg(test)]
mod tests {
    use super::{Args, ListModels, parse_args};

    fn parse(args: &[&str]) -> Args {
        let owned: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
        parse_args(&owned).unwrap()
    }

    #[test]
    fn parses_provider_and_model() {
        let args = parse(&["--provider", "openai", "--model", "gpt-4o"]);

        assert_eq!(args.provider.as_deref(), Some("openai"));
        assert_eq!(args.model.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn parses_print_flag_and_messages() {
        let args = parse(&["-p", "hello", "world"]);

        assert!(args.print);
        assert_eq!(args.messages, vec!["hello", "world"]);
    }

    #[test]
    fn double_dash_stops_option_parsing() {
        let args = parse(&["--", "--not-a-flag", "-x"]);

        assert_eq!(args.messages, vec!["--not-a-flag", "-x"]);
    }

    #[test]
    fn list_models_with_and_without_search() {
        assert_eq!(parse(&["--list-models"]).list_models, Some(ListModels::All));
        assert_eq!(
            parse(&["--list-models", "gpt"]).list_models,
            Some(ListModels::Search("gpt".to_owned()))
        );
    }

    #[test]
    fn missing_value_is_an_error() {
        let owned = vec!["--model".to_owned()];
        let error = parse_args(&owned).unwrap_err();

        assert!(error.contains("需要一个值"));
    }

    #[test]
    fn unknown_option_is_an_error() {
        let owned = vec!["--nope".to_owned()];
        let error = parse_args(&owned).unwrap_err();

        assert!(error.contains("未知选项"));
    }

    #[test]
    fn parses_flags() {
        let args = parse(&["-h", "-v", "--no-session", "--thinking", "high"]);

        assert!(args.help);
        assert!(args.version);
        assert!(args.no_session);
        assert_eq!(args.thinking.as_deref(), Some("high"));
    }
}
