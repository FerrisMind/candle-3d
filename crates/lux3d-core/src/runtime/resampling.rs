#[derive(Debug, Clone)]
pub(super) struct InterpSpec {
    pub(super) start: usize,
    pub(super) weights: Vec<f32>,
}

pub(super) fn compute_aa_cubic_weights(input_size: usize, output_size: usize) -> Vec<InterpSpec> {
    let scale = input_size as f32 / output_size as f32;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut specs = Vec::with_capacity(output_size);

    for i in 0..output_size {
        let center = scale * (i as f32 + 0.5);
        let xmin = ((center - support + 0.5) as isize).max(0) as usize;
        let xmax = ((center + support + 0.5) as isize).min(input_size as isize) as usize;
        let mut weights = Vec::with_capacity(xmax.saturating_sub(xmin));
        let mut total = 0.0f32;
        for j in xmin..xmax {
            let w = aa_cubic_filter((j as f32 - center + 0.5) * invscale);
            weights.push(w);
            total += w;
        }
        if total != 0.0 {
            for weight in &mut weights {
                *weight /= total;
            }
        }
        specs.push(InterpSpec {
            start: xmin,
            weights,
        });
    }

    specs
}

fn aa_cubic_filter(x: f32) -> f32 {
    let a = -0.5f32;
    let x = x.abs();
    if x < 1.0 {
        cubic_convolution1(x, a)
    } else if x < 2.0 {
        cubic_convolution2(x, a)
    } else {
        0.0
    }
}

fn cubic_convolution1(x: f32, a: f32) -> f32 {
    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
}

fn cubic_convolution2(x: f32, a: f32) -> f32 {
    ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
}

pub(super) fn compute_aa_linear_weights(input_size: usize, output_size: usize) -> Vec<InterpSpec> {
    let scale = input_size as f32 / output_size as f32;
    let support = if scale > 1.0 { scale } else { 1.0 };
    let invscale = if scale > 1.0 { 1.0 / scale } else { 1.0 };
    let mut specs = Vec::with_capacity(output_size);

    for i in 0..output_size {
        let center = scale * (i as f32 + 0.5);
        let xmin = ((center - support + 0.5).floor() as isize).max(0) as usize;
        let xmax = ((center + support + 0.5).ceil() as isize).min(input_size as isize) as usize;
        let mut weights = Vec::with_capacity(xmax.saturating_sub(xmin));
        let mut total = 0.0f32;
        for j in xmin..xmax {
            let w = aa_linear_filter((j as f32 - center + 0.5) * invscale);
            weights.push(w);
            total += w;
        }
        if total != 0.0 {
            for weight in &mut weights {
                *weight /= total;
            }
        }
        specs.push(InterpSpec {
            start: xmin,
            weights,
        });
    }

    specs
}

fn aa_linear_filter(x: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 { 1.0 - x } else { 0.0 }
}

pub(super) fn cubic_weight(x: f32) -> f32 {
    let a = -0.75f32;
    let x = x.abs();
    if x <= 1.0 {
        (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
    } else {
        0.0
    }
}

pub(super) fn clamp_isize(value: isize, min: isize, max: isize) -> isize {
    value.max(min).min(max)
}
