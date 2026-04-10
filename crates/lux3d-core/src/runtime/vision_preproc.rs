#![cfg_attr(not(feature = "vision-preproc"), allow(dead_code))]

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use image::DynamicImage;

#[cfg(feature = "vision-preproc")]
use mistralrs_vision::{
    ApplyTensorTransforms, ApplyTransforms, InterpolateResize, Normalize, TensorTransforms,
    ToTensor, Transforms,
};

pub(super) const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub(super) const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub(super) fn tensorize_rgb_image(image: &DynamicImage, device: &Device) -> CandleResult<Tensor> {
    #[cfg(feature = "vision-preproc")]
    {
        image.apply(
            Transforms {
                input: &ToTensor,
                inner_transforms: &[],
            },
            device,
        )
    }

    #[cfg(not(feature = "vision-preproc"))]
    {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        let mut host = Vec::with_capacity((width * height * 3) as usize);
        for channel in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    host.push(f32::from(rgb.get_pixel(x, y)[channel]) / 255.0);
                }
            }
        }
        Tensor::from_vec(host, (3, height as usize, width as usize), device)
    }
}

pub(super) fn resize_chw_image(
    image_chw: &Tensor,
    target_h: usize,
    target_w: usize,
) -> CandleResult<Tensor> {
    let dims = image_chw.dims();
    if dims == [3, target_h, target_w] {
        return Ok(image_chw.clone());
    }

    #[cfg(feature = "vision-preproc")]
    {
        ApplyTensorTransforms::apply(
            image_chw,
            TensorTransforms {
                inner_transforms: &[&InterpolateResize { target_w, target_h }],
            },
            image_chw.device(),
        )
    }

    #[cfg(not(feature = "vision-preproc"))]
    {
        image_chw
            .unsqueeze(0)?
            .interpolate2d(target_h, target_w)?
            .squeeze(0)
    }
}

pub(super) fn normalize_chw_image(
    image_chw: &Tensor,
    mean: [f32; 3],
    std: [f32; 3],
) -> CandleResult<Tensor> {
    #[cfg(feature = "vision-preproc")]
    {
        ApplyTensorTransforms::apply(
            image_chw,
            TensorTransforms {
                inner_transforms: &[&Normalize {
                    mean: mean.into_iter().map(f64::from).collect(),
                    std: std.into_iter().map(f64::from).collect(),
                }],
            },
            image_chw.device(),
        )
    }

    #[cfg(not(feature = "vision-preproc"))]
    {
        let device = image_chw.device();
        let dtype = image_chw.dtype();
        let mean = Tensor::from_slice(&mean, (3,), device)?
            .to_dtype(dtype)?
            .reshape((3, 1, 1))?;
        let std = Tensor::from_slice(&std, (3,), device)?
            .to_dtype(dtype)?
            .reshape((3, 1, 1))?;
        image_chw.broadcast_sub(&mean)?.broadcast_div(&std)
    }
}

pub(super) fn normalize_imagenet_chw(image_chw: &Tensor) -> CandleResult<Tensor> {
    normalize_chw_image(image_chw, IMAGENET_MEAN, IMAGENET_STD)
}

pub(super) fn chw_to_hwc_vec_f32(image_chw: &Tensor) -> CandleResult<Vec<f32>> {
    image_chw
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .permute((1, 2, 0))?
        .flatten_all()?
        .to_vec1::<f32>()
}
