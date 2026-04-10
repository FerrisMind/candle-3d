use candle_core::{D, IndexOp, Result as CandleResult, Tensor};
use nalgebra::{Matrix3, SVD, Vector3};

const ORTHOGONALIZE_EPS: f32 = 1e-6;

pub(crate) fn local_points_from_head_output(head_output: &Tensor) -> CandleResult<Tensor> {
    let xy = head_output.i((.., .., .., ..2))?;
    let z = head_output.i((.., .., .., 2..3))?.exp()?;
    let xy_scaled = xy.broadcast_mul(&z)?;
    Tensor::cat(&[&xy_scaled, &z], D::Minus1)
}

pub(crate) fn camera_pose_from_components(
    out_r: &Tensor,
    out_t: &Tensor,
    device: &candle_core::Device,
) -> CandleResult<Tensor> {
    // Candle does not currently expose the SVD needed for the vendor camera-head
    // orthogonalization, so this remains an explicit CPU offload path.
    let batch = out_t.dim(0)?;
    let rot = svd_orthogonalize_cpu(out_r)?;
    let t = out_t
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut pose = vec![0f32; batch * 16];
    for b in 0..batch {
        let rot_offset = b * 9;
        let pose_offset = b * 16;
        pose[pose_offset..pose_offset + 3].copy_from_slice(&rot[rot_offset..rot_offset + 3]);
        pose[pose_offset + 4..pose_offset + 7]
            .copy_from_slice(&rot[rot_offset + 3..rot_offset + 6]);
        pose[pose_offset + 8..pose_offset + 11]
            .copy_from_slice(&rot[rot_offset + 6..rot_offset + 9]);
        pose[pose_offset + 3] = t[b * 3];
        pose[pose_offset + 7] = t[b * 3 + 1];
        pose[pose_offset + 11] = t[b * 3 + 2];
        pose[pose_offset + 15] = 1.0;
    }
    Tensor::from_vec(pose, (batch, 4, 4), device)
}

fn svd_orthogonalize_cpu(out_r: &Tensor) -> CandleResult<Vec<f32>> {
    let batch = out_r.dim(0)?;
    let raw = out_r
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut result = vec![0f32; batch * 9];
    for b in 0..batch {
        let offset = b * 9;
        let rows = &raw[offset..offset + 9];
        let mut m = Matrix3::from_row_slice(rows);
        for row in 0..3 {
            let v = Vector3::new(m[(row, 0)], m[(row, 1)], m[(row, 2)]);
            let norm = v.norm();
            if !norm.is_finite() || norm <= ORTHOGONALIZE_EPS {
                candle_core::bail!(
                    "camera rotation row {row} for batch {b} was degenerate (norm={norm})"
                );
            }
            let v = v / norm;
            m[(row, 0)] = v[0];
            m[(row, 1)] = v[1];
            m[(row, 2)] = v[2];
        }
        let mt = m.transpose();
        let svd = SVD::new(mt, true, true);
        let Some(u) = svd.u else {
            candle_core::bail!("camera rotation SVD did not return U for batch {b}");
        };
        let Some(v_t) = svd.v_t else {
            candle_core::bail!("camera rotation SVD did not return V^T for batch {b}");
        };
        let mut v = v_t.transpose();
        let det = (v * u.transpose()).determinant();
        if !det.is_finite() {
            candle_core::bail!("camera rotation determinant was not finite for batch {b}");
        }
        for row in 0..3 {
            v[(row, 2)] *= det;
        }
        let r = v * u.transpose();
        if !r.iter().all(|value| value.is_finite()) {
            candle_core::bail!("camera rotation orthogonalization produced non-finite values");
        }
        result[offset] = r[(0, 0)];
        result[offset + 1] = r[(0, 1)];
        result[offset + 2] = r[(0, 2)];
        result[offset + 3] = r[(1, 0)];
        result[offset + 4] = r[(1, 1)];
        result[offset + 5] = r[(1, 2)];
        result[offset + 6] = r[(2, 0)];
        result[offset + 7] = r[(2, 1)];
        result[offset + 8] = r[(2, 2)];
    }
    Ok(result)
}

pub(crate) fn world_points_from_local_and_pose(
    local_points: &Tensor,
    camera_poses: &Tensor,
) -> CandleResult<Tensor> {
    let ones = Tensor::ones_like(&local_points.i((.., .., .., .., ..1))?)?;
    let homo = Tensor::cat(&[local_points, &ones], D::Minus1)?;
    let (b, n, h, w, _d) = homo.dims5()?;
    let homo = homo.reshape((b, n, h * w, 4))?;
    let poses = camera_poses.reshape((b, n, 4, 4))?.transpose(2, 3)?;
    homo.matmul(&poses)?
        .i((.., .., .., ..3))?
        .reshape((b, n, h, w, 3))
}

pub(crate) fn export_mask_from_local_points_and_confidence(
    local_points: &Tensor,
    confidence_logits: &Tensor,
) -> CandleResult<Tensor> {
    let confidence = candle_nn::ops::sigmoid(&confidence_logits.i((.., .., .., .., 0))?)?;
    let confidence_mask = confidence.ge(0.1)?;
    let non_edge = non_edge_mask_from_local_points(local_points)?;
    let false_mask = Tensor::zeros(
        confidence_mask.shape(),
        candle_core::DType::U8,
        confidence_mask.device(),
    )?;
    confidence_mask.where_cond(&non_edge, &false_mask)
}

pub(crate) fn non_edge_mask_from_local_points(local_points: &Tensor) -> CandleResult<Tensor> {
    let depth = local_points.i((.., .., .., .., 2))?;
    non_edge_depth_mask(&depth, 0.03)
}

fn non_edge_depth_mask(depth: &Tensor, rtol: f64) -> CandleResult<Tensor> {
    let (b, n, h, w) = depth.dims4()?;
    let depth = depth.reshape((b * n, 1, h, w))?;
    let max_depth = max_pool_same_3x3(&depth)?;
    let min_depth = max_pool_same_3x3(&depth.affine(-1.0, 0.0)?)?.affine(-1.0, 0.0)?;
    let diff = max_depth.broadcast_sub(&min_depth)?;
    let ratio = diff.broadcast_div(&depth)?;
    let edge = ratio.ge(rtol)?.reshape((b, n, h, w))?;
    let false_mask = Tensor::zeros(edge.shape(), candle_core::DType::U8, edge.device())?;
    let true_mask = Tensor::ones(edge.shape(), candle_core::DType::U8, edge.device())?;
    edge.where_cond(&false_mask, &true_mask)
}

fn max_pool_same_3x3(xs: &Tensor) -> CandleResult<Tensor> {
    let (_b, c, h, _w) = xs.dims4()?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, (xs.dim(0)?, c, h, 1), xs.device())?;
    let xs = Tensor::cat(&[&neg_inf, xs, &neg_inf], 3)?;
    let neg_inf = Tensor::full(
        f32::NEG_INFINITY,
        (xs.dim(0)?, c, 1, xs.dim(3)?),
        xs.device(),
    )?;
    let xs = Tensor::cat(&[&neg_inf, &xs, &neg_inf], 2)?;
    xs.max_pool2d_with_stride((3, 3), (1, 1))
}

#[cfg(test)]
mod tests {
    use super::camera_pose_from_components;
    use candle_core::{Device, Tensor};

    #[test]
    fn rejects_degenerate_rotation_rows() {
        let device = Device::Cpu;
        let out_r = Tensor::zeros((1, 9), candle_core::DType::F32, &device).expect("out_r");
        let out_t = Tensor::zeros((1, 3), candle_core::DType::F32, &device).expect("out_t");
        let err = camera_pose_from_components(&out_r, &out_t, &device).expect_err("must fail");
        let message = err.to_string();
        assert!(
            message.contains("degenerate"),
            "unexpected error: {message}"
        );
    }
}
