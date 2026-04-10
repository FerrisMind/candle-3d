use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    Pi3,
    Pi3x,
    #[serde(rename = "triposr")]
    TripoSr,
}

impl ModelFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pi3 => "pi3",
            Self::Pi3x => "pi3x",
            Self::TripoSr => "triposr",
        }
    }
}

impl std::fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
