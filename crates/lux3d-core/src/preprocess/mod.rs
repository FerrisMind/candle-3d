mod pi3;
mod pi3x;
mod trellis2;
mod triposg;
mod triposr;
mod vggt;

pub use pi3::Pi3PreprocessStage;
pub use pi3x::{Pi3xPreprocessStage, Pi3xRawInput};
pub use trellis2::{
    TRELLIS2_ALPHA_THRESHOLD, TRELLIS2_IMAGE_NET_MEAN, TRELLIS2_IMAGE_NET_STD,
    TRELLIS2_MAX_INPUT_SIDE, TRELLIS2_SUPPORTED_RESOLUTIONS, Trellis2CropInfo, Trellis2Preprocess,
    alpha_foreground_bbox, normalized_pixel_values_from_u8, python_round, reference_crop_box,
    square_crop_premultiplied,
};
pub use triposg::{TripoSgPreparedInputs, TripoSgPreprocessStage};
pub use triposr::TripoPreprocessStage;
pub use vggt::{VggtPreparedInputs, VggtPreprocessStage};

/// Crate-internal re-export: the cascade's `run_prepared` entry resizes the
/// prepared 512 crop to the 1024 DINOv3 input (T6).
pub(crate) use trellis2::resize_lanczos_u8;
