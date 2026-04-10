use candle_core::{DType, IndexOp, Result as CandleResult, Tensor};

pub(super) trait TriplaneDecoder {
    fn forward(&self, xs: &Tensor) -> CandleResult<(Tensor, Tensor)>;
}

#[derive(Debug)]
struct PreparedPlane {
    flat: Tensor,
    channels: usize,
    height: usize,
    width: usize,
}

#[derive(Debug)]
struct PreparedTriplanes {
    xy: PreparedPlane,
    xz: PreparedPlane,
    yz: PreparedPlane,
}

pub(super) fn query_triplane_chunked<D: TriplaneDecoder>(
    scene_code: &Tensor,
    positions: &Tensor,
    decoder: &D,
    chunk_size: usize,
) -> CandleResult<(Tensor, Tensor, Tensor)> {
    let leading_dims = positions.dims();
    let flat = positions.reshape((positions.elem_count() / 3, 3))?;
    let planes = prepare_triplanes(scene_code)?;
    let mut feature_chunks = Vec::new();
    let mut density_chunks = Vec::new();
    let mut color_chunks = Vec::new();
    let step = if chunk_size == 0 {
        flat.dim(0)?
    } else {
        chunk_size
    };
    let mut start = 0usize;
    while start < flat.dim(0)? {
        let len = (flat.dim(0)? - start).min(step);
        let chunk = flat.narrow(0, start, len)?;
        let features = sample_triplane_concat_prepared(&planes, &chunk)?;
        let (density, color_features) = decoder.forward(&features)?;
        feature_chunks.push(features);
        density_chunks.push((density.affine(1.0, -1.0)?).exp()?);
        color_chunks.push(candle_nn::ops::sigmoid(&color_features)?);
        start += len;
    }

    let feature_refs = feature_chunks.iter().collect::<Vec<_>>();
    let density_refs = density_chunks.iter().collect::<Vec<_>>();
    let color_refs = color_chunks.iter().collect::<Vec<_>>();

    let base_shape = leading_dims[..leading_dims.len() - 1].to_vec();
    let flat_features = Tensor::cat(&feature_refs, 0)?;
    let flat_density = Tensor::cat(&density_refs, 0)?;
    let flat_color = Tensor::cat(&color_refs, 0)?;

    Ok((
        flat_features.reshape(output_shape(&base_shape, 120))?,
        flat_density.reshape(output_shape(&base_shape, 1))?,
        flat_color.reshape(output_shape(&base_shape, 3))?,
    ))
}

pub(super) fn query_triplane_density_chunked<D: TriplaneDecoder>(
    scene_code: &Tensor,
    positions: &Tensor,
    decoder: &D,
    chunk_size: usize,
) -> CandleResult<Tensor> {
    let leading_dims = positions.dims();
    let flat = positions.reshape((positions.elem_count() / 3, 3))?;
    let planes = prepare_triplanes(scene_code)?;
    let step = if chunk_size == 0 {
        flat.dim(0)?
    } else {
        chunk_size
    };
    let mut density_chunks = Vec::new();
    let mut start = 0usize;
    while start < flat.dim(0)? {
        let len = (flat.dim(0)? - start).min(step);
        let chunk = flat.narrow(0, start, len)?;
        let features = sample_triplane_concat_prepared(&planes, &chunk)?;
        let (density, _color_features) = decoder.forward(&features)?;
        density_chunks.push((density.affine(1.0, -1.0)?).exp()?);
        start += len;
    }
    let density_refs = density_chunks.iter().collect::<Vec<_>>();
    Tensor::cat(&density_refs, 0)?.reshape(output_shape(&leading_dims[..leading_dims.len() - 1], 1))
}

pub(super) fn query_triplane_color_chunked<D: TriplaneDecoder>(
    scene_code: &Tensor,
    positions: &Tensor,
    decoder: &D,
    chunk_size: usize,
) -> CandleResult<Tensor> {
    let leading_dims = positions.dims();
    let flat = positions.reshape((positions.elem_count() / 3, 3))?;
    let planes = prepare_triplanes(scene_code)?;
    let step = if chunk_size == 0 {
        flat.dim(0)?
    } else {
        chunk_size
    };
    let mut color_chunks = Vec::new();
    let mut start = 0usize;
    while start < flat.dim(0)? {
        let len = (flat.dim(0)? - start).min(step);
        let chunk = flat.narrow(0, start, len)?;
        let features = sample_triplane_concat_prepared(&planes, &chunk)?;
        let (_density, color_features) = decoder.forward(&features)?;
        color_chunks.push(candle_nn::ops::sigmoid(&color_features)?);
        start += len;
    }
    let color_refs = color_chunks.iter().collect::<Vec<_>>();
    Tensor::cat(&color_refs, 0)?.reshape(output_shape(&leading_dims[..leading_dims.len() - 1], 3))
}

fn prepare_triplanes(scene_code: &Tensor) -> CandleResult<PreparedTriplanes> {
    Ok(PreparedTriplanes {
        xy: prepare_plane(&scene_code.i(0)?)?,
        xz: prepare_plane(&scene_code.i(1)?)?,
        yz: prepare_plane(&scene_code.i(2)?)?,
    })
}

fn prepare_plane(plane: &Tensor) -> CandleResult<PreparedPlane> {
    let (channels, height, width) = plane.dims3()?;
    Ok(PreparedPlane {
        flat: plane
            .permute((1, 2, 0))?
            .reshape((height * width, channels))?
            .contiguous()?,
        channels,
        height,
        width,
    })
}

fn sample_triplane_concat_prepared(
    planes: &PreparedTriplanes,
    positions: &Tensor,
) -> CandleResult<Tensor> {
    let scaled = positions.affine(1.0 / 0.87f64, 0.0)?;
    let xy = Tensor::cat(
        &[
            &scaled.i((.., 0))?.unsqueeze(1)?,
            &scaled.i((.., 1))?.unsqueeze(1)?,
        ],
        1,
    )?;
    let xz = Tensor::cat(
        &[
            &scaled.i((.., 0))?.unsqueeze(1)?,
            &scaled.i((.., 2))?.unsqueeze(1)?,
        ],
        1,
    )?;
    let yz = Tensor::cat(
        &[
            &scaled.i((.., 1))?.unsqueeze(1)?,
            &scaled.i((.., 2))?.unsqueeze(1)?,
        ],
        1,
    )?;
    let feat_xy = sample_plane_prepared(&planes.xy, &xy)?;
    let feat_xz = sample_plane_prepared(&planes.xz, &xz)?;
    let feat_yz = sample_plane_prepared(&planes.yz, &yz)?;
    Tensor::cat(&[&feat_xy, &feat_xz, &feat_yz], 1)
}

fn sample_plane_prepared(plane: &PreparedPlane, coords: &Tensor) -> CandleResult<Tensor> {
    let channels = plane.channels;
    let height = plane.height;
    let width = plane.width;
    let x = coords
        .i((.., 0))?
        .affine(width as f64 / 2.0, (width as f64 - 1.0) / 2.0)?;
    let y = coords
        .i((.., 1))?
        .affine(height as f64 / 2.0, (height as f64 - 1.0) / 2.0)?;

    let x0_raw = x.floor()?;
    let y0_raw = y.floor()?;
    let x1_raw = x0_raw.affine(1.0, 1.0)?;
    let y1_raw = y0_raw.affine(1.0, 1.0)?;

    let x0 = x0_raw.clamp(0.0, (width - 1) as f64)?;
    let y0 = y0_raw.clamp(0.0, (height - 1) as f64)?;
    let x1 = x1_raw.clamp(0.0, (width - 1) as f64)?;
    let y1 = y1_raw.clamp(0.0, (height - 1) as f64)?;

    let wx = x.broadcast_sub(&x0_raw)?;
    let wy = y.broadcast_sub(&y0_raw)?;
    let one = Tensor::ones(wx.shape(), DType::F32, wx.device())?;
    let wx0 = one.broadcast_sub(&wx)?;
    let wy0 = one.broadcast_sub(&wy)?;

    let idx00 = y0
        .affine(width as f64, 0.0)?
        .broadcast_add(&x0)?
        .to_dtype(DType::U32)?;
    let idx01 = y0
        .affine(width as f64, 0.0)?
        .broadcast_add(&x1)?
        .to_dtype(DType::U32)?;
    let idx10 = y1
        .affine(width as f64, 0.0)?
        .broadcast_add(&x0)?
        .to_dtype(DType::U32)?;
    let idx11 = y1
        .affine(width as f64, 0.0)?
        .broadcast_add(&x1)?
        .to_dtype(DType::U32)?;

    let p = coords.dim(0)?;

    let v00 = plane.flat.embedding(&idx00.contiguous()?)?;
    let v01 = plane.flat.embedding(&idx01.contiguous()?)?;
    let v10 = plane.flat.embedding(&idx10.contiguous()?)?;
    let v11 = plane.flat.embedding(&idx11.contiguous()?)?;

    let w00 = wx0.broadcast_mul(&wy0)?;
    let w01 = wx.broadcast_mul(&wy0)?;
    let w10 = wx0.broadcast_mul(&wy)?;
    let w11 = wx.broadcast_mul(&wy)?;

    let valid_x0 = x0_raw
        .ge(0.0)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&x0_raw.le((width - 1) as f64)?.to_dtype(DType::F32)?)?;
    let valid_x1 = x1_raw
        .ge(0.0)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&x1_raw.le((width - 1) as f64)?.to_dtype(DType::F32)?)?;
    let valid_y0 = y0_raw
        .ge(0.0)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&y0_raw.le((height - 1) as f64)?.to_dtype(DType::F32)?)?;
    let valid_y1 = y1_raw
        .ge(0.0)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&y1_raw.le((height - 1) as f64)?.to_dtype(DType::F32)?)?;

    let m00 = valid_x0.broadcast_mul(&valid_y0)?;
    let m01 = valid_x1.broadcast_mul(&valid_y0)?;
    let m10 = valid_x0.broadcast_mul(&valid_y1)?;
    let m11 = valid_x1.broadcast_mul(&valid_y1)?;

    let w00 = w00.unsqueeze(1)?.broadcast_as((p, channels))?;
    let w01 = w01.unsqueeze(1)?.broadcast_as((p, channels))?;
    let w10 = w10.unsqueeze(1)?.broadcast_as((p, channels))?;
    let w11 = w11.unsqueeze(1)?.broadcast_as((p, channels))?;
    let m00 = m00.unsqueeze(1)?.broadcast_as((p, channels))?;
    let m01 = m01.unsqueeze(1)?.broadcast_as((p, channels))?;
    let m10 = m10.unsqueeze(1)?.broadcast_as((p, channels))?;
    let m11 = m11.unsqueeze(1)?.broadcast_as((p, channels))?;

    v00.broadcast_mul(&w00)?
        .broadcast_mul(&m00)?
        .broadcast_add(&v01.broadcast_mul(&w01)?.broadcast_mul(&m01)?)?
        .broadcast_add(&v10.broadcast_mul(&w10)?.broadcast_mul(&m10)?)?
        .broadcast_add(&v11.broadcast_mul(&w11)?.broadcast_mul(&m11)?)
}

fn output_shape(prefix: &[usize], last_dim: usize) -> Vec<usize> {
    let mut shape = prefix.to_vec();
    shape.push(last_dim);
    shape
}
