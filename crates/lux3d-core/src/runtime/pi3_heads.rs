use candle_core::{D, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::{nn_blocks::linear, point_camera_ops::camera_pose_from_components};

#[derive(Debug)]
pub struct LinearPts3dHead {
    proj: Linear,
    patch_size: usize,
}

impl LinearPts3dHead {
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        output_dim: usize,
        patch_size: usize,
    ) -> CandleResult<Self> {
        Ok(Self {
            proj: linear(
                vb.pp("proj"),
                in_dim,
                output_dim * patch_size * patch_size,
                true,
            )?,
            patch_size,
        })
    }

    pub fn forward(
        &self,
        hidden: &Tensor,
        image_height: usize,
        image_width: usize,
        patch_start_idx: usize,
    ) -> CandleResult<Tensor> {
        let tokens = hidden.i((.., patch_start_idx.., ..))?;
        let (batch, _seq, _dim) = tokens.dims3()?;
        let feat = self.proj.forward(&tokens)?;
        let feat = feat.transpose(1, 2)?.reshape((
            batch,
            feat.dim(D::Minus1)?,
            image_height / self.patch_size,
            image_width / self.patch_size,
        ))?;
        let feat = candle_nn::ops::pixel_shuffle(&feat, self.patch_size)?;
        feat.permute((0, 2, 3, 1))
    }
}

pub fn load_point_head(vb: VarBuilder) -> CandleResult<LinearPts3dHead> {
    LinearPts3dHead::new(vb.pp("point_head"), 1024, 3, 14)
}

pub fn load_conf_head(vb: VarBuilder) -> CandleResult<LinearPts3dHead> {
    LinearPts3dHead::new(vb.pp("conf_head"), 1024, 1, 14)
}

#[derive(Debug)]
struct ResConvBlock {
    res_conv1: Linear,
    res_conv2: Linear,
    res_conv3: Linear,
}

impl ResConvBlock {
    fn new(vb: VarBuilder, dim: usize) -> CandleResult<Self> {
        Ok(Self {
            res_conv1: linear(vb.pp("res_conv1"), dim, dim, true)?,
            res_conv2: linear(vb.pp("res_conv2"), dim, dim, true)?,
            res_conv3: linear(vb.pp("res_conv3"), dim, dim, true)?,
        })
    }

    fn forward(&self, residual: &Tensor) -> CandleResult<Tensor> {
        let x = self.res_conv1.forward(residual)?.relu()?;
        let x = self.res_conv2.forward(&x)?.relu()?;
        let x = self.res_conv3.forward(&x)?.relu()?;
        x + residual
    }
}

#[derive(Debug)]
pub struct CameraHead {
    res_conv: [ResConvBlock; 2],
    more_mlps_0: Linear,
    more_mlps_2: Linear,
    fc_t: Linear,
    fc_rot: Linear,
}

impl CameraHead {
    pub fn new(vb: VarBuilder, dim: usize) -> CandleResult<Self> {
        Ok(Self {
            res_conv: [
                ResConvBlock::new(vb.pp("res_conv").pp("0"), dim)?,
                ResConvBlock::new(vb.pp("res_conv").pp("1"), dim)?,
            ],
            more_mlps_0: linear(vb.pp("more_mlps").pp("0"), dim, dim, true)?,
            more_mlps_2: linear(vb.pp("more_mlps").pp("2"), dim, dim, true)?,
            fc_t: linear(vb.pp("fc_t"), dim, 3, true)?,
            fc_rot: linear(vb.pp("fc_rot"), dim, 9, true)?,
        })
    }

    pub fn forward(&self, hidden: &Tensor, patch_h: usize, patch_w: usize) -> CandleResult<Tensor> {
        let hidden = hidden.i((.., 5.., ..))?;
        let device = hidden.device().clone();
        let (batch, _seq, dim) = hidden.dims3()?;
        let mut feat = hidden;
        feat = self.res_conv[0].forward(&feat)?;
        feat = self.res_conv[1].forward(&feat)?;
        let feat = feat
            .permute((0, 2, 1))?
            .reshape((batch, dim, patch_h, patch_w))?
            .avg_pool2d((patch_h, patch_w))?
            .reshape((batch, dim))?;
        let feat = self.more_mlps_0.forward(&feat)?.relu()?;
        let feat = self.more_mlps_2.forward(&feat)?.relu()?;
        let out_t = self.fc_t.forward(&feat)?;
        let out_r = self.fc_rot.forward(&feat)?;
        camera_pose_from_components(&out_r, &out_t, &device)
    }
}

pub fn load_camera_head(vb: VarBuilder) -> CandleResult<CameraHead> {
    CameraHead::new(vb.pp("camera_head"), 512)
}
