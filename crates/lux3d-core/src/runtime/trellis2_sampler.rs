//! TRELLIS.2 flow-matching sampler: Euler with classifier-free guidance,
//! guidance interval and CFG rescale
//! (`FlowEulerGuidanceIntervalSampler` =
//! `GuidanceIntervalSamplerMixin` + `ClassifierFreeGuidanceSamplerMixin` +
//! `FlowEulerSampler`, `trellis2/pipelines/samplers/*.py`).
//!
//! Exact reference math (v-prediction, sigma_min = 1e-5):
//! - schedule: `t_seq = linspace(1, 0, steps + 1)` (float64) then
//!   `t_seq = rescale_t * t / (1 + (rescale_t - 1) * t)`; steps are the
//!   pairs `(t_i, t_{i+1})`.
//! - model time: `t_model = 1000 * t` cast to F32 (the sampler wraps the
//!   scalar in a F32 tensor).
//! - velocity -> x0/eps (`_v_to_xstart_eps`):
//!   `eps = (1 - t) * v + x_t`,
//!   `x_0 = (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * v`.
//! - Euler step: `x_prev = x_t - (t - t_prev) * v`.
//! - CFG (`guidance_strength s`): `s == 1` -> single conditional pass;
//!   `s == 0` -> single negative pass; otherwise
//!   `pred = s * v_pos + (1 - s) * v_neg`.
//! - guidance interval: CFG applies only when
//!   `interval[0] <= t <= interval[1]`, otherwise a single conditional pass.
//! - CFG rescale (`guidance_rescale g > 0`):
//!   `x0_pos = pred_to_xstart(x_t, t, v_pos)`,
//!   `x0_cfg = pred_to_xstart(x_t, t, pred)`,
//!   `std = torch.std over all non-batch dims (Bessel n-1, keepdim)`,
//!   `x0_res = x0_cfg * (std_pos / std_cfg)`,
//!   `x0 = g * x0_res + (1 - g) * x0_cfg`,
//!   `pred = xstart_to_pred(x_t, t, x0)`.
//!
//! All tensor math is F32; the schedule/scalars are F64 like the reference
//! (numpy float64), with each scalar multiplied into the tensors in F32.

use candle_core::{D, Result as CandleResult, Tensor};

/// Sampler parameters (`pipeline.json` `sparse_structure_sampler.params`).
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2SamplerParams {
    /// Euler steps (12).
    pub steps: usize,
    /// Flow sigma_min (1e-5).
    pub sigma_min: f64,
    /// Classifier-free guidance strength (7.5; 1.0 disables CFG).
    pub guidance_strength: f64,
    /// CFG rescale (0.7; 0 disables).
    pub guidance_rescale: f64,
    /// Inclusive t interval where CFG is applied ([0.6, 1.0]).
    pub guidance_interval: [f64; 2],
    /// t-schedule rescale factor R (5.0).
    pub rescale_t: f64,
}

impl Default for Trellis2SamplerParams {
    fn default() -> Self {
        // SS defaults from pipeline.json.
        Self {
            steps: 12,
            sigma_min: 1e-5,
            guidance_strength: 7.5,
            guidance_rescale: 0.7,
            guidance_interval: [0.6, 1.0],
            rescale_t: 5.0,
        }
    }
}

/// Per-step trace collected during sampling (mirrors the oracle arrays
/// `pred_x_t` / `pred_x_0` lists).
#[derive(Debug, Clone)]
pub struct Trellis2SampleTrace {
    /// The `(t, t_prev)` pairs that were stepped.
    pub t_pairs: Vec<(f64, f64)>,
    /// `x_{t_prev}` after each Euler step (`x_t_steps[-1] == samples`).
    pub x_t_steps: Vec<Tensor>,
    /// `pred_x_0` from `_v_to_xstart_eps` for each step.
    pub pred_x0_steps: Vec<Tensor>,
}

/// `np.linspace(start, stop, steps + 1)` in F64 (endpoint forced to
/// `stop` exactly like numpy).
fn linspace_f64(start: f64, stop: f64, points: usize) -> Vec<f64> {
    if points < 2 {
        return vec![stop; points.max(1)];
    }
    let div = points - 1;
    let delta = (stop - start) / div as f64;
    let mut values: Vec<f64> = (0..div).map(|i| start + delta * i as f64).collect();
    values.push(stop);
    values
}

/// The rescaled t schedule: `linspace(1, 0, steps + 1)` then
/// `rescale_t * t / (1 + (rescale_t - 1) * t)` (F64). Length `steps + 1`,
/// monotonically decreasing, last entry exactly 0.
pub fn ss_t_schedule(steps: usize, rescale_t: f64) -> Vec<f64> {
    let linear = linspace_f64(1.0, 0.0, steps + 1);
    linear
        .into_iter()
        .map(|t| rescale_t * t / (1.0 + (rescale_t - 1.0) * t))
        .collect()
}

/// `_v_to_xstart_eps`: `eps = (1 - t) * v + x_t`,
/// `x_0 = (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * v`.
pub fn v_to_xstart_eps(
    x_t: &Tensor,
    t: f64,
    v: &Tensor,
    sigma_min: f64,
) -> CandleResult<(Tensor, Tensor)> {
    let eps = v.affine(1.0 - t, 0.0)?.add(x_t)?;
    let a = (1.0 - sigma_min) as f32;
    let b = (sigma_min + (1.0 - sigma_min) * t) as f32;
    let x0 = x_t.affine(a as f64, 0.0)?.sub(&v.affine(b as f64, 0.0)?)?;
    Ok((x0, eps))
}

/// `_pred_to_xstart`: `x_0 = (1 - sigma_min) * x_t -
/// (sigma_min + (1 - sigma_min) * t) * pred`.
pub fn pred_to_xstart(x_t: &Tensor, t: f64, pred: &Tensor, sigma_min: f64) -> CandleResult<Tensor> {
    let a = (1.0 - sigma_min) as f32;
    let b = (sigma_min + (1.0 - sigma_min) * t) as f32;
    x_t.affine(a as f64, 0.0)?.sub(&pred.affine(b as f64, 0.0)?)
}

/// `_xstart_to_pred`: `pred = ((1 - sigma_min) * x_t - x_0) /
/// (sigma_min + (1 - sigma_min) * t)`.
pub fn xstart_to_pred(x_t: &Tensor, t: f64, x0: &Tensor, sigma_min: f64) -> CandleResult<Tensor> {
    let a = (1.0 - sigma_min) as f32;
    let b = (sigma_min + (1.0 - sigma_min) * t) as f32;
    x_t.affine(a as f64, 0.0)?
        .sub(x0)?
        .affine(1.0 / b as f64, 0.0)
}

/// `torch.std(x, dim=1.., keepdim=True)`: per-sample standard deviation over
/// every non-batch dim with Bessel correction (n - 1), keepdim so it
/// broadcasts.
fn std_over_non_batch(xs: &Tensor) -> CandleResult<Tensor> {
    let batch = xs.dim(0)?;
    let samples = xs.elem_count() / batch;
    let flat = xs.reshape((batch, samples))?;
    let mean = flat.mean_keepdim(D::Minus1)?;
    let centered = flat.broadcast_sub(&mean)?;
    let var = centered
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .affine(1.0 / (samples as f64 - 1.0), 0.0)?;
    // [batch, 1] -> [batch, 1, 1, ..] matching the input rank.
    let mut shape: Vec<usize> = vec![batch, 1];
    shape.resize(xs.rank(), 1);
    var.sqrt()?.reshape(shape)
}

/// One guided model evaluation: guidance interval + CFG mixing + rescale.
/// `model` receives `(x_t, t_model = 1000 * t, cond)`.
pub fn guided_velocity<F>(
    model: &F,
    x_t: &Tensor,
    t: f64,
    pos_cond: &Tensor,
    neg_cond: &Tensor,
    params: &Trellis2SamplerParams,
) -> CandleResult<Tensor>
where
    F: Fn(&Tensor, f32, &Tensor) -> CandleResult<Tensor>,
{
    let t_model = (1000.0 * t) as f32;
    let applies_cfg = params.guidance_interval[0] <= t && t <= params.guidance_interval[1];
    let strength = if applies_cfg {
        params.guidance_strength
    } else {
        1.0
    };
    if strength == 1.0 {
        return model(x_t, t_model, pos_cond);
    }
    if strength == 0.0 {
        return model(x_t, t_model, neg_cond);
    }
    let v_pos = model(x_t, t_model, pos_cond)?;
    let v_neg = model(x_t, t_model, neg_cond)?;
    let pred = v_pos
        .affine(strength, 0.0)?
        .add(&v_neg.affine(1.0 - strength, 0.0)?)?;
    if params.guidance_rescale > 0.0 {
        let x0_pos = pred_to_xstart(x_t, t, &v_pos, params.sigma_min)?;
        let x0_cfg = pred_to_xstart(x_t, t, &pred, params.sigma_min)?;
        let std_pos = std_over_non_batch(&x0_pos)?;
        let std_cfg = std_over_non_batch(&x0_cfg)?;
        let ratio = std_pos.broadcast_div(&std_cfg)?;
        let x0_res = x0_cfg.broadcast_mul(&ratio)?;
        let x0 = x0_res
            .affine(params.guidance_rescale, 0.0)?
            .add(&x0_cfg.affine(1.0 - params.guidance_rescale, 0.0)?)?;
        return xstart_to_pred(x_t, t, &x0, params.sigma_min);
    }
    Ok(pred)
}

/// Euler flow-matching sample with CFG + guidance interval, generic over the
/// model closure. Returns `(samples, trace)` where `samples` is the final
/// `x_{t=0}` and the trace holds the per-step `x_{t_prev}` and `pred_x_0`
/// tensors (`x_t_steps[-1] == samples`).
pub fn sample_dense<F>(
    model: &F,
    noise: &Tensor,
    pos_cond: &Tensor,
    neg_cond: &Tensor,
    params: &Trellis2SamplerParams,
) -> CandleResult<(Tensor, Trellis2SampleTrace)>
where
    F: Fn(&Tensor, f32, &Tensor) -> CandleResult<Tensor>,
{
    let schedule = ss_t_schedule(params.steps, params.rescale_t);
    let mut trace = Trellis2SampleTrace {
        t_pairs: Vec::with_capacity(params.steps),
        x_t_steps: Vec::with_capacity(params.steps),
        pred_x0_steps: Vec::with_capacity(params.steps),
    };
    let mut sample = noise.clone();
    for index in 0..params.steps {
        let t = schedule[index];
        let t_prev = schedule[index + 1];
        let v = guided_velocity(model, &sample, t, pos_cond, neg_cond, params)?;
        let (x0, _eps) = v_to_xstart_eps(&sample, t, &v, params.sigma_min)?;
        let next = sample.sub(&v.affine(t - t_prev, 0.0)?)?;
        trace.t_pairs.push((t, t_prev));
        trace.pred_x0_steps.push(x0);
        trace.x_t_steps.push(next.clone());
        sample = next;
    }
    Ok((sample, trace))
}

/// Sparse SLAT sampling entry point: Euler/CFG over `[N, C]` feature rows on
/// FIXED coordinates (the voxel structure is decided by the SS stage and does
/// not change during SLAT sampling). The Euler/CFG math is exactly
/// [`sample_dense`] — the closure receives the token feats, the scaled model
/// time and the conditioning, plus the frozen coordinate list.
pub fn sample_sparse<M>(
    model: &M,
    coords: &[[i32; 3]],
    noise_feats: &Tensor,
    pos_cond: &Tensor,
    neg_cond: &Tensor,
    params: &Trellis2SamplerParams,
) -> CandleResult<(Tensor, Trellis2SampleTrace)>
where
    M: Fn(&Tensor, f32, &Tensor, &[[i32; 3]]) -> CandleResult<Tensor>,
{
    let coords = coords.to_vec();
    let model = |x: &Tensor, t_model: f32, cond: &Tensor| model(x, t_model, cond, &coords);
    sample_dense(&model, noise_feats, pos_cond, neg_cond, params)
}
