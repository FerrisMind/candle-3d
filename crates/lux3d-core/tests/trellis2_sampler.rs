//! Unit tests for the TRELLIS.2 flow sampler: exact schedule values, Euler
/// updates and CFG/interval/rescale composition against hand-computed
/// reference sequences (`trellis2/pipelines/samplers/*.py`).
use std::cell::Cell;

use candle_core::{Device, Tensor};
use lux3d_core::runtime::{
    Trellis2SamplerParams, guided_velocity, pred_to_xstart, sample_dense, ss_t_schedule,
    v_to_xstart_eps, xstart_to_pred,
};

fn tensor_1x2(values: [f64; 2]) -> Tensor {
    Tensor::from_vec(
        values.iter().map(|v| *v as f32).collect(),
        (1, 2),
        &Device::Cpu,
    )
    .expect("tensor")
}

fn read_1x2(tensor: &Tensor) -> [f64; 2] {
    let values = tensor
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("read");
    [values[0] as f64, values[1] as f64]
}

/// The F64 schedule `linspace(1, 0, steps+1)` then
/// `R * t / (1 + (R - 1) * t)` must reproduce the oracle values exactly
/// (oracle steps=4, R=5: (1, 0.9375), (0.9375, 0.8333333333333334),
/// (0.8333333333333334, 0.625), (0.625, 0.0)).
#[test]
fn schedule_matches_oracle_pairs() {
    let schedule = ss_t_schedule(4, 5.0);
    let expected = [1.0_f64, 0.9375, 0.833_333_333_333_333_4, 0.625, 0.0];
    assert_eq!(schedule.len(), 5);
    for (a, b) in schedule.iter().zip(expected.iter()) {
        assert_eq!(a, b, "schedule value must match np.float64 exactly");
    }
    // steps=12 (SS default): 13 points, ends at exactly 0, monotone.
    let schedule = ss_t_schedule(12, 5.0);
    assert_eq!(schedule.len(), 13);
    assert_eq!(schedule[0], 1.0);
    assert_eq!(schedule[12], 0.0);
    for pair in schedule.windows(2) {
        assert!(pair[0] > pair[1], "schedule must be strictly decreasing");
    }
    // spot: linspace(1,0,13)[3] = 1 - 3/12 = 0.75 -> 5*0.75/(1+4*0.75) = 3.75/4
    assert!((schedule[3] - 0.9375).abs() < 1e-15);
}

/// `v_to_xstart_eps` / `pred_to_xstart` / `xstart_to_pred` round trips on
/// hand values (sigma_min = 1e-5, t = 0.5).
#[test]
fn velocity_conversions_match_reference_formulas() {
    let sigma_min = 1e-5;
    let t = 0.5_f64;
    let x_t = tensor_1x2([3.0, -1.0]);
    let v = tensor_1x2([1.0, 2.0]);
    let (x0, eps) = v_to_xstart_eps(&x_t, t, &v, sigma_min).expect("convert");
    // eps = (1 - t) * v + x_t
    let want_eps = [0.5 * 1.0 + 3.0, 0.5 * 2.0 + (-1.0)];
    // x0 = (1 - sm) * x_t - (sm + (1 - sm) * t) * v
    let a = 1.0 - sigma_min;
    let b = sigma_min + (1.0 - sigma_min) * t;
    let want_x0 = [a * 3.0 - b * 1.0, a * (-1.0) - b * 2.0];
    for (got, want) in read_1x2(&eps).iter().zip(want_eps.iter()) {
        assert!((got - want).abs() < 1e-6);
    }
    for (got, want) in read_1x2(&x0).iter().zip(want_x0.iter()) {
        assert!((got - want).abs() < 1e-6);
    }
    // pred_to_xstart == v_to_xstart_eps x0; xstart_to_pred inverts it.
    let x0_again = pred_to_xstart(&x_t, t, &v, sigma_min).expect("xstart");
    for (got, want) in read_1x2(&x0_again).iter().zip(want_x0.iter()) {
        assert!((got - want).abs() < 1e-6);
    }
    let v_back = xstart_to_pred(&x_t, t, &x0_again, sigma_min).expect("pred");
    for (got, want) in read_1x2(&v_back).iter().zip([1.0, 2.0].iter()) {
        assert!((got - want).abs() < 1e-4, "{got} vs {want}");
    }
}

/// Euler sampling with the identity velocity field `v = x_t` (cond
/// independent): CFG mixing and rescale are both no-ops, so each step is
/// exactly `x_{t_prev} = x_t * (1 - (t - t_prev))`.
#[test]
fn euler_steps_match_hand_computed_sequence() {
    let noise = tensor_1x2([3.0, -1.0]);
    let cond = tensor_1x2([0.0, 0.0]);
    let params = Trellis2SamplerParams {
        steps: 4,
        sigma_min: 1e-5,
        guidance_strength: 7.5,
        guidance_rescale: 0.7,
        guidance_interval: [0.6, 1.0],
        rescale_t: 5.0,
    };
    let model = |x: &Tensor, _t: f32, _c: &Tensor| Ok(x.clone());
    let (samples, trace) = sample_dense(&model, &noise, &cond, &cond, &params).expect("sample");

    let schedule = ss_t_schedule(4, 5.0);
    let mut expected = [3.0_f64, -1.0];
    for step in 0..4 {
        let dt = schedule[step] - schedule[step + 1];
        // pred_x0 = (1 - sm) * x_t - (sm + (1 - sm) * t) * v with v = x_t,
        // evaluated at the PRE-update x_t (before the Euler step).
        let t = schedule[step];
        let sm = 1e-5;
        let want_x0 = expected
            .iter()
            .map(|x| (1.0 - sm) * x - (sm + (1.0 - sm) * t) * x)
            .collect::<Vec<_>>();
        expected = [expected[0] * (1.0 - dt), expected[1] * (1.0 - dt)];
        let got = read_1x2(&trace.x_t_steps[step]);
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "step {step}: {a} vs {b}");
        }
        let got_x0 = read_1x2(&trace.pred_x0_steps[step]);
        for (a, b) in got_x0.iter().zip(want_x0.iter()) {
            assert!((a - b).abs() < 1e-5, "x0 step {step}: {a} vs {b}");
        }
    }
    // Final sample equals the last trace entry (t_prev = 0).
    let final_values = read_1x2(&samples);
    for (a, b) in final_values.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

/// CFG mixing: `pred = s * v_pos + (1 - s) * v_neg` with separate forward
/// passes; strength 1 short-circuits to cond only, 0 to neg only.
#[test]
fn cfg_mixing_and_call_pattern_match_reference() {
    let x_t = tensor_1x2([1.0, 1.0]);
    let pos = tensor_1x2([1.0, 1.0]);
    let neg = tensor_1x2([2.0, 2.0]);
    let pos_calls = Cell::new(0usize);
    let neg_calls = Cell::new(0usize);
    // Discriminate the conditioning tensors by content (Tensor has no
    // PartialEq); pos cond starts with 1.0, neg with 2.0.
    let is_neg = |c: &Tensor| -> bool {
        c.flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map(|values| values[0] > 1.5)
            .expect("cond")
    };
    let model = |x: &Tensor, _t: f32, c: &Tensor| -> candle_core::Result<Tensor> {
        let value = if !is_neg(c) {
            pos_calls.set(pos_calls.get() + 1);
            1.0_f32
        } else {
            neg_calls.set(neg_calls.get() + 1);
            -1.0_f32
        };
        Ok(x.affine(value as f64, 0.0)?)
    };
    let params = Trellis2SamplerParams {
        steps: 1,
        sigma_min: 0.0,
        guidance_strength: 7.5,
        guidance_rescale: 0.0,
        guidance_interval: [0.0, 1.0],
        rescale_t: 1.0,
    };
    let v = guided_velocity(&model, &x_t, 0.75, &pos, &neg, &params).expect("velocity");
    // 7.5 * 1 + (1 - 7.5) * (-1) = 7.5 + 6.5 = 14
    let values = read_1x2(&v);
    assert!((values[0] - 14.0).abs() < 1e-5 && (values[1] - 14.0).abs() < 1e-5);
    assert_eq!(pos_calls.get(), 1);
    assert_eq!(neg_calls.get(), 1);

    // strength == 1: single conditional pass even inside the interval.
    pos_calls.set(0);
    neg_calls.set(0);
    let params = Trellis2SamplerParams {
        guidance_strength: 1.0,
        ..params.clone()
    };
    let v = guided_velocity(&model, &x_t, 0.75, &pos, &neg, &params).expect("velocity");
    assert!((read_1x2(&v)[0] - 1.0).abs() < 1e-6);
    assert_eq!(pos_calls.get(), 1);
    assert_eq!(neg_calls.get(), 0);

    // strength == 0: single negative pass.
    pos_calls.set(0);
    neg_calls.set(0);
    let params = Trellis2SamplerParams {
        guidance_strength: 0.0,
        ..params.clone()
    };
    let v = guided_velocity(&model, &x_t, 0.75, &pos, &neg, &params).expect("velocity");
    assert!((read_1x2(&v)[0] + 1.0).abs() < 1e-6);
    assert_eq!(pos_calls.get(), 0);
    assert_eq!(neg_calls.get(), 1);
}

/// Guidance interval: CFG applies only when
/// `interval[0] <= t <= interval[1]`; outside, a single conditional pass.
#[test]
fn guidance_interval_forces_cond_only_outside() {
    let x_t = tensor_1x2([1.0, 1.0]);
    let pos = tensor_1x2([1.0, 1.0]);
    let neg = tensor_1x2([2.0, 2.0]);
    let pos_calls = Cell::new(0usize);
    let neg_calls = Cell::new(0usize);
    let is_neg = |c: &Tensor| -> bool {
        c.flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map(|values| values[0] > 1.5)
            .expect("cond")
    };
    let model = |x: &Tensor, _t: f32, c: &Tensor| -> candle_core::Result<Tensor> {
        if !is_neg(c) {
            pos_calls.set(pos_calls.get() + 1);
            Ok(x.affine(1.0, 0.0)?)
        } else {
            neg_calls.set(neg_calls.get() + 1);
            Ok(x.affine(-1.0, 0.0)?)
        }
    };
    let params = Trellis2SamplerParams {
        steps: 2,
        sigma_min: 0.0,
        guidance_strength: 7.5,
        guidance_rescale: 0.0,
        guidance_interval: [0.6, 1.0],
        rescale_t: 1.0, // schedule [1, 0.5, 0]: second step t=0.5 < 0.6
    };
    let model_fn = |x: &Tensor, t: f32, c: &Tensor| model(x, t, c);
    let (_samples, trace) = sample_dense(&model_fn, &x_t, &pos, &neg, &params).expect("sample");
    // Step 1 (t=1): CFG -> pos + neg; step 2 (t=0.5): cond only.
    assert_eq!(pos_calls.get(), 2);
    assert_eq!(neg_calls.get(), 1);
    // Step 1 velocity: 7.5 * 1 + (1 - 7.5) * (-1) = 14 -> x_prev = 1 - 0.5*14 = -6.
    let step1 = read_1x2(&trace.x_t_steps[0]);
    assert!((step1[0] + 6.0).abs() < 1e-5);
    // Step 2 velocity: cond only, and the mock model returns v = +x_t = -6,
    // so x_prev = -6 - 0.5 * (-6) = -3.
    let step2 = read_1x2(&trace.x_t_steps[1]);
    assert!((step2[0] + 3.0).abs() < 1e-5, "step2 {} vs -3", step2[0]);
}

/// CFG rescale with hand-computed values (sigma_min = 0, s = 2, g = 0.5,
/// t = 1): x0_pos = [-1, 1], x0_cfg = [-3, 0], std ratio sqrt(2/4.5),
/// x0 = [-2.5, 0]; xstart_to_pred = (x_t - x0) / 1 = [3.5, 2].
#[test]
fn cfg_rescale_matches_hand_computed_values() {
    let x_t = tensor_1x2([1.0, 2.0]);
    let pos = tensor_1x2([1.0, 1.0]);
    let neg = tensor_1x2([2.0, 2.0]);
    let is_neg = |c: &Tensor| -> bool {
        c.flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map(|values| values[0] > 1.5)
            .expect("cond")
    };
    let model = |_x: &Tensor, _t: f32, c: &Tensor| -> candle_core::Result<Tensor> {
        if !is_neg(c) {
            Ok(tensor_1x2([2.0, 1.0]))
        } else {
            Ok(tensor_1x2([0.0, 0.0]))
        }
    };
    let params = Trellis2SamplerParams {
        steps: 1,
        sigma_min: 0.0,
        guidance_strength: 2.0,
        guidance_rescale: 0.5,
        guidance_interval: [0.0, 1.0],
        rescale_t: 1.0,
    };
    let v = guided_velocity(&model, &x_t, 1.0, &pos, &neg, &params).expect("velocity");
    let values = read_1x2(&v);
    let want: [f64; 2] = [3.5, 2.0];
    for (a, b) in values.iter().zip(want.iter()) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}
