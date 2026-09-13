//! 模型目录（registry）：从 JSON 数据加载模型清单。
//!
//! 数据形状与原版 provider data 一致：`{ api: { modelId: Model } }`。
//! 一个文件可以包含多个 provider 的模型（`Model` 自带 `provider` 字段）。
//!
//! C++ 对照：`ModelRegistry` 类似一个从配置加载的 `std::vector<Model>` + 查找表。

use std::collections::{BTreeSet, HashMap};

use crate::model::Model;

/// 内置模型目录（编译期内嵌，运行时无需读文件）。
const BUILTIN_MODELS: &str = include_str!("../data/models.json");

/// 原始目录数据：api -> (model id -> Model)。
type RawCatalog = HashMap<String, HashMap<String, Model>>;

/// 加载/解析模型目录时的错误。
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistryError {
    pub message: String,
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// 模型注册表：一份可按 (provider, id) 查找的模型清单。
#[derive(Clone, Debug, Default)]
pub struct ModelRegistry {
    models: Vec<Model>,
}

impl ModelRegistry {
    /// 从 JSON 文本加载；按 (provider, id) 排序，保证顺序稳定。
    pub fn from_json(text: &str) -> Result<Self, RegistryError> {
        let raw: RawCatalog = serde_json::from_str(text).map_err(|error| RegistryError {
            message: format!("解析模型目录失败: {error}"),
        })?;
        let mut models: Vec<Model> = raw
            .into_values()
            .flat_map(|by_id| by_id.into_values())
            .collect();
        models.sort_by(|a, b| (&a.provider, &a.id).cmp(&(&b.provider, &b.id)));
        Ok(Self { models })
    }

    /// 内置目录。
    ///
    /// 数据是编译期内嵌的，解析失败属于开发错误，所以这里直接 panic（由测试覆盖）。
    #[must_use]
    pub fn builtin() -> Self {
        Self::from_json(BUILTIN_MODELS).expect("内置 models.json 必须有效")
    }

    /// 按 (provider, id) 查找模型。
    #[must_use]
    pub fn get(&self, provider: &str, id: &str) -> Option<&Model> {
        self.models
            .iter()
            .find(|model| model.provider == provider && model.id == id)
    }

    /// 全部模型。
    #[must_use]
    pub fn models(&self) -> &[Model] {
        &self.models
    }

    /// 所有 provider 名字（去重、排序）。
    #[must_use]
    pub fn providers(&self) -> Vec<&str> {
        let names: BTreeSet<&str> = self
            .models
            .iter()
            .map(|model| model.provider.as_str())
            .collect();
        names.into_iter().collect()
    }

    /// 某个 provider 下的所有模型。
    #[must_use]
    pub fn models_for_provider(&self, provider: &str) -> Vec<&Model> {
        self.models
            .iter()
            .filter(|model| model.provider == provider)
            .collect()
    }
}
