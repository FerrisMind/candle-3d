use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    Pi3,
    Pi3x,
    #[serde(rename = "triposr")]
    TripoSr,
    #[serde(rename = "triposg")]
    TripoSg,
    Vggt,
    /// TRELLIS.2-4B image-to-3D (Microsoft). The `oxide-lab` repository id is
    /// reserved for a future canonical-package mirror; it must not be uploaded
    /// to before the weight licence is cleared for redistribution.
    #[serde(rename = "trellis2")]
    Trellis2,
}

impl ModelFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pi3 => "pi3",
            Self::Pi3x => "pi3x",
            Self::TripoSr => "triposr",
            Self::TripoSg => "triposg",
            Self::Vggt => "vggt",
            Self::Trellis2 => "trellis2",
        }
    }

    /// Repository that hosts the canonical package (the four files required by
    /// [`crate::weights::load_canonical_weights`]). For VGGT this is the
    /// `oxide-lab` mirror, not the raw upstream `facebook/VGGT-1B` checkpoint.
    /// For TRELLIS.2 the id is reserved and intentionally unused for now.
    pub const fn huggingface_repo_id(self) -> &'static str {
        match self {
            Self::Pi3 => "oxide-lab/Pi3",
            Self::Pi3x => "oxide-lab/Pi3X",
            Self::TripoSr => "oxide-lab/TripoSR",
            Self::TripoSg => "oxide-lab/TripoSG",
            Self::Vggt => "oxide-lab/VGGT-1B",
            Self::Trellis2 => "oxide-lab/TRELLIS.2-4B",
        }
    }
}

impl std::fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
