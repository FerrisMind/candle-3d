//! TRELLIS.2 texture branch (T7) tests:
//!
//! 1. Static config coverage: the 640-name tex DiT configs and the 284-name
//!    guided tex decoder config (plus the canonical package header check).
//! 2. Tiny tex DiT: `noise || shape_slat_norm` concat layout + forward.
//! 3. Guided C2S exactness (tiny synthetic weights): the guided decode walks
//!    exactly the shape decoder's coordinate tree; all-true guides select
//!    every child.
//! 4. PBR postprocess layout/clamps and the trilinear vertex sampling
//!    reference cases.
//! 5. Gated real-weight replays: the tex VAE golden (per-block parity) and
//!    the tex SLAT flow golden (4-step sampler).
//!
//! Tests skip (with a message) when the golden artifacts or the canonical
//! weights package are absent. The real-weight replays load the tex DiT
//! (~2.6 GiB bf16) or the two VAE decoders and run on CPU.

use std::collections::HashSet;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use lux3d_core::runtime::{
    Trellis2PbrLayout, Trellis2SamplerParams, Trellis2ShapeVaeConfig, Trellis2ShapeVaeDecoder,
    Trellis2SlatConfig, Trellis2TexDecoder, Trellis2TexDit, concat_tex_input, pbr_attrs_from_raw,
    query_vertex_attrs, query_vertex_colors, sample_sparse, split_pbr_attrs,
};
use lux3d_core::test_support::workspace_root;

fn slat_golden_dir() -> PathBuf {
    workspace_root()
        .join(".swarm")
        .join("results")
        .join("trellis2-slat-golden")
}

fn vae_golden_dir() -> PathBuf {
    workspace_root()
        .join(".swarm")
        .join("results")
        .join("trellis2-vae-golden")
}

fn canonical_model_file() -> PathBuf {
    if let Some(path) = std::env::var_os("LUX3D_TRELLIS2_CANONICAL_MODEL") {
        return PathBuf::from(path);
    }
    lux3d_core::test_support::canonical_package_dir(lux3d_core::ModelFamily::Trellis2)
        .join("model.safetensors")
}

fn pattern_tensor(shape: &[usize], seed: usize) -> Tensor {
    let len: usize = shape.iter().product();
    Tensor::from_vec(
        (0..len)
            .map(|i| (((i * 7 + seed * 3) % 23) as f32 - 11.0) * 0.05)
            .collect(),
        shape.to_vec(),
        &Device::Cpu,
    )
    .expect("tensor")
}

fn const_tensor(shape: &[usize], value: f32) -> Tensor {
    let len: usize = shape.iter().product();
    Tensor::from_vec(vec![value; len], shape.to_vec(), &Device::Cpu).expect("tensor")
}

fn iota_tensor(shape: &[usize]) -> Tensor {
    let len: usize = shape.iter().product();
    Tensor::from_vec(
        (0..len).map(|i| i as f32).collect(),
        shape.to_vec(),
        &Device::Cpu,
    )
    .expect("tensor")
}

fn tensor_map<S: Into<String>>(
    entries: impl IntoIterator<Item = (S, Tensor)>,
) -> candle_nn::VarBuilder<'static> {
    let map: HashMap<String, Tensor> = entries
        .into_iter()
        .map(|(name, tensor)| (name.into(), tensor))
        .collect();
    candle_nn::VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

use std::collections::HashMap;

fn flat(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("tensor to flat f32")
}

fn compare(actual: &[f32], expected: &[f32], label: &str) -> (f64, f64, f64) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let mut max_abs = 0.0_f64;
    let mut expected_max = 0.0_f64;
    let mut sum_sq_diff = 0.0_f64;
    let mut sum_sq_expected = 0.0_f64;
    for (&a, &b) in actual.iter().zip(expected.iter()) {
        let diff = (a - b).abs() as f64;
        max_abs = max_abs.max(diff);
        expected_max = expected_max.max(b.abs() as f64);
        sum_sq_diff += diff * diff;
        sum_sq_expected += (b as f64) * (b as f64);
    }
    let rel_vs_max = max_abs / expected_max.max(1e-12);
    let rms_rel = (sum_sq_diff / sum_sq_expected.max(1e-12)).sqrt();
    println!(
        "{label}: max_abs={max_abs:.3e}, rel_vs_max_abs={rel_vs_max:.3e}, rms_rel={rms_rel:.3e}"
    );
    (max_abs, rel_vs_max, rms_rel)
}

fn read_safetensors_stage_keys(path: &std::path::Path, prefix: &str) -> Option<Vec<String>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).ok()?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len > 256 * 1024 * 1024 {
        return None;
    }
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes).ok()?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes).ok()?;
    Some(
        header
            .as_object()?
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect(),
    )
}

/// The texture configs map to exactly the canonical stage sizes: both tex
/// DiTs carry 640 unique tensor names (same architecture as the shape DiT,
/// 64 input channels) and the guided tex decoder carries 284 (the 292-name
/// shape decoder minus the four to_subdiv weight/bias pairs).
#[test]
fn tex_configs_expected_tensor_names() {
    for (config, resolution) in [
        (Trellis2SlatConfig::tex_512(), 32usize),
        (Trellis2SlatConfig::tex_1024(), 64usize),
    ] {
        assert_eq!(config.resolution, resolution);
        assert_eq!(config.in_channels, 64, "noise 32 + shape_slat_norm 32");
        assert_eq!(config.out_channels, 32);
        assert_eq!(config.model_channels, 1536);
        assert_eq!(config.cond_channels, 1024);
        assert_eq!(config.num_blocks, 30);
        let names = config.expected_tensor_names();
        assert_eq!(names.len(), 640, "tex_dit stage tensor count");
        let unique: HashSet<&String> = names.iter().collect();
        assert_eq!(unique.len(), 640, "expected names must be unique");
        assert!(names.contains(&"input_layer.weight".to_string()));
        assert!(names.contains(&"out_layer.bias".to_string()));
    }

    let shape_config = Trellis2ShapeVaeConfig::flexi_dual_shape();
    assert_eq!(shape_config.pred_subdiv, true);
    assert_eq!(shape_config.expected_tensor_names().len(), 292);

    let tex_config = Trellis2ShapeVaeConfig::tex_decoder();
    assert_eq!(tex_config.out_channels, Trellis2PbrLayout::CHANNELS);
    assert_eq!(tex_config.pred_subdiv, false);
    assert_eq!(tex_config.c2s_stage_count(), 4);
    assert_eq!(tex_config.model_channels, shape_config.model_channels);
    assert_eq!(tex_config.num_blocks, shape_config.num_blocks);
    assert_eq!(tex_config.latent_channels, 32);
    let names = tex_config.expected_tensor_names();
    assert_eq!(names.len(), 284, "tex_dec stage tensor count");
    let unique: HashSet<&String> = names.iter().collect();
    assert_eq!(unique.len(), 284, "expected names must be unique");
    assert!(
        !names.iter().any(|name| name.contains("to_subdiv")),
        "guided decoder predicts no subdivision"
    );
    assert!(names.contains(&"blocks.0.4.conv1.weight".to_string()));
    assert!(names.contains(&"blocks.3.4.norm1.weight".to_string()));
    assert!(names.contains(&"output_layer.weight".to_string()));
}

/// Canonical package header: `tex_dit_512.`, `tex_dit_1024.` and `tex_dec.`
/// hold exactly the tensors the configs expect.
#[test]
fn canonical_package_covers_tex_stages() {
    let model_file = canonical_model_file();
    if !model_file.is_file() {
        eprintln!(
            "SKIP: canonical TRELLIS.2 weights not found at {}",
            model_file.display()
        );
        return;
    }
    for (prefix, count) in [
        ("tex_dit_512.", 640usize),
        ("tex_dit_1024.", 640usize),
        ("tex_dec.", 284usize),
    ] {
        let stage_keys = read_safetensors_stage_keys(&model_file, prefix)
            .unwrap_or_else(|| panic!("{prefix} header"));
        assert_eq!(stage_keys.len(), count, "{prefix} stage tensor count");
        let expected: HashSet<String> = if prefix.starts_with("tex_dit") {
            let config = if prefix == "tex_dit_512." {
                Trellis2SlatConfig::tex_512()
            } else {
                Trellis2SlatConfig::tex_1024()
            };
            config.expected_tensor_names().into_iter().collect()
        } else {
            Trellis2ShapeVaeConfig::tex_decoder()
                .expected_tensor_names()
                .into_iter()
                .collect()
        };
        let stage: HashSet<String> = stage_keys
            .iter()
            .map(|k| k.trim_start_matches(prefix).to_string())
            .collect();
        assert_eq!(
            expected, stage,
            "{prefix}: expected names must cover exactly the stage tensors"
        );
    }
}

/// The texture input contract: `concat_tex_input` places the sampled noise
/// channels FIRST and the normalized shape SLAT second, summing to 64, and
/// rejects mismatched rows or channels. A tiny tex DiT forward keeps
/// `[N, 32]` and the loader rejects non-texture configs.
#[test]
fn tiny_tex_dit_concat_layout_and_forward() {
    let noise = pattern_tensor(&[3, 32], 0);
    let cond = pattern_tensor(&[3, 32], 1);
    let concat = concat_tex_input(&noise, &cond).expect("concat");
    assert_eq!(concat.dims(), &[3, 64]);
    let noise_v = flat(&noise);
    let cond_v = flat(&cond);
    let concat_v = flat(&concat);
    for row in 0..3 {
        for c in 0..32 {
            assert_eq!(concat_v[row * 64 + c], noise_v[row * 32 + c]);
            assert_eq!(concat_v[row * 64 + 32 + c], cond_v[row * 32 + c]);
        }
    }
    assert!(concat_tex_input(&pattern_tensor(&[4, 32], 2), &cond).is_err());
    assert!(concat_tex_input(&pattern_tensor(&[3, 30], 3), &cond).is_err());

    let channels = 16;
    let heads = 2;
    let head_dim = channels / heads;
    let cond_channels = 8;
    let mlp_hidden = 24;
    let in_channels = 64;
    let resolution = 8;
    let config = Trellis2SlatConfig {
        resolution,
        in_channels,
        out_channels: 32,
        model_channels: channels,
        cond_channels,
        num_blocks: 1,
        num_heads: heads,
        mlp_ratio: mlp_hidden as f64 / channels as f64,
        share_mod: true,
        qk_rms_norm: true,
        qk_rms_norm_cross: true,
    };
    // The wrapper must reject shape-style configs before touching weights.
    let shape_config = Trellis2SlatConfig {
        in_channels: 32,
        out_channels: 32,
        ..config
    };
    assert!(
        Trellis2TexDit::load(tensor_map(Vec::<(String, Tensor)>::new()), &shape_config).is_err()
    );

    let entries: Vec<(&str, Tensor)> = vec![
        (
            "t_embedder.mlp.0.weight",
            pattern_tensor(&[channels, 256], 0),
        ),
        ("t_embedder.mlp.0.bias", const_tensor(&[channels], 0.1)),
        (
            "t_embedder.mlp.2.weight",
            pattern_tensor(&[channels, channels], 1),
        ),
        ("t_embedder.mlp.2.bias", const_tensor(&[channels], -0.05)),
        (
            "adaLN_modulation.1.weight",
            pattern_tensor(&[6 * channels, channels], 2),
        ),
        (
            "adaLN_modulation.1.bias",
            const_tensor(&[6 * channels], 0.2),
        ),
        (
            "input_layer.weight",
            pattern_tensor(&[channels, in_channels], 3),
        ),
        ("input_layer.bias", const_tensor(&[channels], 0.0)),
        (
            "out_layer.weight",
            pattern_tensor(&[config.out_channels, channels], 4),
        ),
        ("out_layer.bias", const_tensor(&[config.out_channels], 0.0)),
        ("blocks.0.norm2.weight", const_tensor(&[channels], 1.0)),
        ("blocks.0.norm2.bias", const_tensor(&[channels], 0.0)),
        ("blocks.0.modulation", pattern_tensor(&[6 * channels], 5)),
        (
            "blocks.0.self_attn.to_qkv.weight",
            pattern_tensor(&[3 * channels, channels], 6),
        ),
        (
            "blocks.0.self_attn.to_qkv.bias",
            const_tensor(&[3 * channels], 0.0),
        ),
        (
            "blocks.0.self_attn.to_out.weight",
            pattern_tensor(&[channels, channels], 7),
        ),
        (
            "blocks.0.self_attn.to_out.bias",
            const_tensor(&[channels], 0.0),
        ),
        (
            "blocks.0.self_attn.q_rms_norm.gamma",
            const_tensor(&[heads, head_dim], 1.0),
        ),
        (
            "blocks.0.self_attn.k_rms_norm.gamma",
            const_tensor(&[heads, head_dim], 1.0),
        ),
        (
            "blocks.0.cross_attn.to_q.weight",
            pattern_tensor(&[channels, channels], 8),
        ),
        (
            "blocks.0.cross_attn.to_q.bias",
            const_tensor(&[channels], 0.0),
        ),
        (
            "blocks.0.cross_attn.to_kv.weight",
            pattern_tensor(&[2 * channels, cond_channels], 9),
        ),
        (
            "blocks.0.cross_attn.to_kv.bias",
            const_tensor(&[2 * channels], 0.0),
        ),
        (
            "blocks.0.cross_attn.to_out.weight",
            pattern_tensor(&[channels, channels], 10),
        ),
        (
            "blocks.0.cross_attn.to_out.bias",
            const_tensor(&[channels], 0.0),
        ),
        (
            "blocks.0.cross_attn.q_rms_norm.gamma",
            const_tensor(&[heads, head_dim], 1.0),
        ),
        (
            "blocks.0.cross_attn.k_rms_norm.gamma",
            const_tensor(&[heads, head_dim], 1.0),
        ),
        (
            "blocks.0.mlp.mlp.0.weight",
            pattern_tensor(&[mlp_hidden, channels], 11),
        ),
        ("blocks.0.mlp.mlp.0.bias", const_tensor(&[mlp_hidden], 0.0)),
        (
            "blocks.0.mlp.mlp.2.weight",
            pattern_tensor(&[channels, mlp_hidden], 12),
        ),
        ("blocks.0.mlp.mlp.2.bias", const_tensor(&[channels], 0.0)),
    ];
    let vb = tensor_map(entries);
    let dit = Trellis2TexDit::load(vb, &config).expect("tex dit load");
    assert_eq!(dit.config().in_channels, 64);

    let coords: Vec<[i32; 3]> = vec![[0, 0, 0], [1, 2, 3], [7, 0, 5]];
    let shape_norm = pattern_tensor(&[3, 32], 13);
    let noise = pattern_tensor(&[3, 32], 14);
    let cond = pattern_tensor(&[1, 5, cond_channels], 15);
    let out = dit
        .forward(&noise, &shape_norm, &coords, 500.0, &cond)
        .expect("forward");
    assert_eq!(out.dims(), &[3, 32]);
    assert!(flat(&out).iter().all(|v| v.is_finite()));

    // Row-count mismatch between noise and shape cond is rejected.
    assert!(
        dit.forward(
            &pattern_tensor(&[4, 32], 16),
            &shape_norm,
            &coords,
            1.0,
            &cond
        )
        .is_err()
    );
}

/// Tiny texture-decoder config: 5 stages, channels [16, 8, 8, 8, 8],
/// latent 4, 6 output channels, guided C2S (no to_subdiv weights).
fn tiny_tex_config(num_blocks: [usize; 5]) -> Trellis2ShapeVaeConfig {
    Trellis2ShapeVaeConfig {
        model_channels: vec![16, 8, 8, 8, 8],
        num_blocks: num_blocks.to_vec(),
        latent_channels: 4,
        out_channels: 6,
        mlp_ratio: 4.0,
        pred_subdiv: false,
    }
}

fn tiny_shape_config(num_blocks: [usize; 5]) -> Trellis2ShapeVaeConfig {
    Trellis2ShapeVaeConfig {
        model_channels: vec![16, 8, 8, 8, 8],
        num_blocks: num_blocks.to_vec(),
        latent_channels: 4,
        out_channels: 7,
        mlp_ratio: 4.0,
        pred_subdiv: true,
    }
}

/// The stage-0 (16-channel) C2S block with the alternating subdivision bias
/// (+1, -1, ...): even subidx children survive. `with_to_subdiv` toggles
/// between the shape variant (pred path) and the texture variant (guided).
fn tiny_c2s_entries(
    stage: usize,
    block: usize,
    channels: usize,
    with_to_subdiv: bool,
) -> Vec<(String, Tensor)> {
    let prefix = format!("blocks.{stage}.{block}");
    let mut entries = vec![
        (
            format!("{prefix}.conv1.weight"),
            pattern_tensor(&[64, 3, 3, 3, channels], 100 + stage),
        ),
        (format!("{prefix}.conv1.bias"), const_tensor(&[64], 0.0)),
        // C2S conv2 runs on the spatial (repeated) channels: [8, 3, 3, 3, 8].
        (
            format!("{prefix}.conv2.weight"),
            pattern_tensor(&[8, 3, 3, 3, 8], 110 + stage),
        ),
        (format!("{prefix}.conv2.bias"), const_tensor(&[8], 0.0)),
        (
            format!("{prefix}.norm1.weight"),
            const_tensor(&[channels], 1.0),
        ),
        (
            format!("{prefix}.norm1.bias"),
            const_tensor(&[channels], 0.0),
        ),
    ];
    if with_to_subdiv {
        entries.push((
            format!("{prefix}.to_subdiv.weight"),
            const_tensor(&[8, channels], 0.0),
        ));
        // Alternating sign: logits > 0 exactly on the even subidx lanes.
        entries.push((
            format!("{prefix}.to_subdiv.bias"),
            Tensor::from_vec(
                (0..8)
                    .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                    .collect(),
                (8,),
                &Device::Cpu,
            )
            .expect("subdiv bias"),
        ));
    }
    entries
}

/// Guided C2S exactness (synthetic weights): the shape decoder predicts the
/// alternating subdivision logits; the guided texture decoder consuming them
/// externally walks the IDENTICAL coordinate tree (per-stage coords equal,
/// final coords equal), and all-true guides select every child (8x per
/// stage).
#[test]
fn guided_c2s_walks_the_shape_tree() {
    let blocks = [0usize, 0, 0, 0, 0];
    let shape_config = tiny_shape_config(blocks);
    let tex_config = tiny_tex_config(blocks);

    let mut shape_entries = vec![
        (
            "from_latent.weight".to_string(),
            const_tensor(&[16, 4], 0.0),
        ),
        ("from_latent.bias".to_string(), iota_tensor(&[16])),
        (
            "output_layer.weight".to_string(),
            const_tensor(&[7, 8], 0.0),
        ),
        ("output_layer.bias".to_string(), const_tensor(&[7], 0.0)),
    ];
    shape_entries.extend(tiny_c2s_entries(0, 0, 16, true));
    for stage in 1..4usize {
        shape_entries.extend(tiny_c2s_entries(stage, 0, 8, true));
    }
    let shape_decoder =
        Trellis2ShapeVaeDecoder::load(tensor_map(shape_entries), &shape_config).expect("shape");

    let mut tex_entries = vec![
        (
            "from_latent.weight".to_string(),
            const_tensor(&[16, 4], 0.0),
        ),
        ("from_latent.bias".to_string(), iota_tensor(&[16])),
        (
            "output_layer.weight".to_string(),
            const_tensor(&[6, 8], 0.0),
        ),
        ("output_layer.bias".to_string(), const_tensor(&[6], 0.0)),
    ];
    tex_entries.extend(tiny_c2s_entries(0, 0, 16, false));
    for stage in 1..4usize {
        tex_entries.extend(tiny_c2s_entries(stage, 0, 8, false));
    }
    let tex_vb = tensor_map(tex_entries);
    let tex_decoder = Trellis2TexDecoder::load(tex_vb, &tex_config).expect("tex");
    // Wrong configs are rejected by the wrapper.
    assert!(
        Trellis2TexDecoder::load(tensor_map(Vec::<(String, Tensor)>::new()), &shape_config)
            .is_err()
    );

    let coords = vec![[0i32, 0, 0]];
    let latent = const_tensor(&[1, 4], 0.0);
    let shape_decoded = shape_decoder
        .decode(&coords, &latent, true)
        .expect("shape decode");

    // Alternating logits: 4 of 8 children per stage -> 4/16/64/256 rows.
    let counts: Vec<usize> = shape_decoded.trace.iter().map(|b| b.coords.len()).collect();
    assert_eq!(counts, vec![4, 16, 64, 256]);
    assert_eq!(shape_decoded.subdiv_logits.len(), 4);

    let tex_decoded = tex_decoder
        .decode(&coords, &latent, &shape_decoded.subdiv_logits)
        .expect("tex decode");
    assert_eq!(tex_decoded.coords, shape_decoded.coords);
    assert_eq!(tex_decoded.feats.dims(), &[256, 6]);
    // Per-stage trace coords identical (the guided decode owns no logits).
    let tex_traced = tex_decoder
        .decode_with_trace(&coords, &latent, &shape_decoded.subdiv_logits)
        .expect("tex trace decode");
    assert_eq!(tex_traced.trace.len(), shape_decoded.trace.len());
    for (tex_block, shape_block) in tex_traced.trace.iter().zip(&shape_decoded.trace) {
        assert_eq!(tex_block.coords, shape_block.coords);
    }

    // All-true guides select every child: 8 / 64 / 512 / 4096 rows. The
    // guide tensors carry one [N_parents_stage, 8] row block per stage.
    let all_true: Vec<Tensor> = [1usize, 8, 64, 512]
        .iter()
        .map(|&rows| const_tensor(&[rows, 8], 1.0))
        .collect();
    let full = tex_decoder
        .decode(&coords, &latent, &all_true)
        .expect("full");
    let stage_sizes: Vec<usize> = [8, 64, 512, 4096].to_vec();
    assert_eq!(full.coords.len(), 4096);
    assert_eq!(full.feats.dims(), &[4096, 6]);
    let _ = stage_sizes;
    // Even-subidx guides keep the shape decode's exact rows on stage 0.
    let first_stage = &tex_traced.trace[0];
    assert_eq!(
        first_stage.coords,
        vec![[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]
    );
}

/// PBR postprocess: `raw * 0.5 + 0.5` clamps to [0, 1], the 6-channel
/// layout splits into base_color/metallic/roughness/alpha, wrong channel
/// counts are rejected.
#[test]
fn pbr_postprocess_layout_and_clamps() {
    // Row: [-3, -0.4, 0.0, 0.4, 1.0, 3.0] -> [0(clamped), 0.3, 0.5, 0.7, 1.0, 1(clamped)].
    let feats = Tensor::from_vec(
        vec![-3.0_f32, -0.4, 0.0, 0.4, 1.0, 3.0],
        (1, 6),
        &Device::Cpu,
    )
    .expect("feats");
    let attrs = pbr_attrs_from_raw(&feats).expect("attrs");
    assert_eq!(
        attrs,
        vec![0.0_f32, 0.3, 0.5, 0.7, 1.0, 1.0],
        "affine + clamp"
    );
    assert!(attrs.iter().all(|&v| (0.0..=1.0).contains(&v)));

    // Two rows: base_color rows split apart from the scalars.
    let feats2 = Tensor::from_vec(
        vec![
            0.0_f32, 0.2, 0.4, 0.6, 0.8, 1.0, // row 0
            -1.0, 0.0, 1.0, -0.2, 0.4, 0.9, // row 1
        ],
        (2, 6),
        &Device::Cpu,
    )
    .expect("feats2");
    let attrs2 = pbr_attrs_from_raw(&feats2).expect("attrs2");
    let (base, metallic, roughness, alpha) = split_pbr_attrs(&attrs2).expect("split");
    assert_eq!(base, vec![[0.5_f32, 0.6, 0.7], [0.0, 0.5, 1.0]]);
    assert_eq!(metallic, vec![0.8_f32, 0.4]);
    assert_eq!(roughness, vec![0.9_f32, 0.7]);
    assert_eq!(alpha, vec![1.0_f32, 0.95]);

    let wrong = Tensor::from_vec(vec![0.0_f32; 5], (1, 5), &Device::Cpu).expect("wrong");
    assert!(pbr_attrs_from_raw(&wrong).is_err());
    assert!(split_pbr_attrs(&vec![0.0_f32; 7]).is_err());
}

/// Sparse trilinear vertex sampling with the flex_gemm semantics:
/// `q = (vertex + 0.5) * resolution`, base `floor(q - 0.5)`, neighbor weight
/// `prod(1 - |q - cell - 0.5|)`, missing neighbors drop out and the weights
/// normalize by the surviving mass.
#[test]
fn trilinear_vertex_attrs_reference_cases() {
    let resolution = 4u32;
    // Two adjacent voxels on the x axis with distinct 6-channel attrs.
    let coords = vec![[0i32, 0, 0], [1, 0, 0]];
    let a0 = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
    let a1 = [0.7_f32, 0.6, 0.5, 0.4, 0.3, 0.2];
    let mut attrs = Vec::new();
    attrs.extend_from_slice(&a0);
    attrs.extend_from_slice(&a1);

    // Vertex at the center of voxel (0,0,0): q = (v + 0.5) * 4 = [0.5, 0.5,
    // 0.5] -> v = [-0.375; 3]. Weight 1 on (0,0,0), 0 elsewhere: exact a0.
    let center = [[-0.375_f32; 3]];
    let sampled = query_vertex_attrs(&center, &coords, &attrs, 6, resolution).expect("center");
    assert_eq!(sampled.len(), 6);
    for (got, want) in sampled.iter().zip(&a0) {
        assert!((got - want).abs() < 1e-6, "center {got} vs {want}");
    }

    // Vertex exactly halfway between the two voxels: q = [1.0, 0.5, 0.5]
    // (v = [-0.25, -0.375, -0.375]). Both voxels carry weight 0.5: the
    // mean.
    let midpoint = [[-0.25_f32, -0.375, -0.375]];
    let sampled = query_vertex_attrs(&midpoint, &coords, &attrs, 6, resolution).expect("mid");
    for axis in 0..6 {
        let want = 0.5 * a0[axis] + 0.5 * a1[axis];
        assert!(
            (sampled[axis] - want).abs() < 1e-6,
            "mid axis {axis}: {} vs {want}",
            sampled[axis]
        );
    }

    // Missing neighbor: only voxel (1,0,0) present. The same midpoint
    // query drops the (0,0,0) weight and renormalizes by 0.5 -> exact a1.
    let coords_right = vec![[1i32, 0, 0]];
    let attrs_right: Vec<f32> = a1.to_vec();
    let sampled =
        query_vertex_attrs(&midpoint, &coords_right, &attrs_right, 6, resolution).expect("missing");
    for (got, want) in sampled.iter().zip(&a1) {
        assert!((got - want).abs() < 1e-6, "missing {got} vs {want}");
    }

    // All neighbors missing (out-of-grid vertex): q = (0.6 + 0.5) * 4 = 4.4
    // on the x axis puts every candidate cell outside [0, 4) or absent ->
    // zero attributes.
    let outside = [[0.6_f32, -0.375, -0.375]];
    let sampled = query_vertex_attrs(&outside, &coords, &attrs, 6, resolution).expect("outside");
    assert!(sampled.iter().all(|&v| v == 0.0));

    // The base-color helper picks channels 0..3.
    let colors = query_vertex_colors(&center, &coords, &attrs, resolution).expect("colors");
    assert_eq!(colors, vec![a0[0], a0[1], a0[2]]);

    // Buffer mismatches are rejected.
    assert!(query_vertex_attrs(&center, &coords, &attrs[..4], 6, resolution).is_err());
    assert!(query_vertex_attrs(&center, &coords, &attrs, 0, resolution).is_err());
    assert!(query_vertex_attrs(&center, &coords[..1], &attrs, 6, resolution).is_err());
}

/// Gated real-weight golden: the guided tex VAE replay. The shape decode
/// (collect_trace) supplies the raw subdivision guides; the tex decode must
/// reproduce every golden per-block coordinate set exactly and every per-
/// block feature block within the F32 tolerance, and its final coords must
/// equal the shape decode's final coords with the golden [7666, 6] shape.
#[test]
fn tex_vae_golden_replay() {
    let golden = vae_golden_dir();
    let model_file = canonical_model_file();
    if !golden.join("tex_blocks_0_0_coords.npy").is_file() {
        eprintln!("SKIP: tex VAE golden missing at {}", golden.display());
        return;
    }
    if !model_file.is_file() {
        eprintln!(
            "SKIP: canonical TRELLIS.2 weights not found at {}",
            model_file.display()
        );
        return;
    }

    use ndarray::Array2;
    let slat_coords: Array2<i32> =
        ndarray_npy::read_npy(golden.join("slat_coords.npy")).expect("slat_coords");
    let slat_feats: Array2<f32> =
        ndarray_npy::read_npy(golden.join("slat_feats.npy")).expect("slat_feats");
    assert_eq!(slat_coords.dim(), (64, 3));
    assert_eq!(slat_feats.dim(), (64, 32));
    let coords: Vec<[i32; 3]> = (0..64)
        .map(|i| {
            [
                slat_coords[[i, 0]],
                slat_coords[[i, 1]],
                slat_coords[[i, 2]],
            ]
        })
        .collect();

    let device = Device::Cpu;
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)
    }
    .expect("mmaped canonical weights");

    // Shape decode with trace: the C2S blocks emit the guide logits.
    let shape_decoder = Trellis2ShapeVaeDecoder::load(
        vb.pp("shape_dec"),
        &Trellis2ShapeVaeConfig::flexi_dual_shape(),
    )
    .expect("shape_dec load");
    let latent = Tensor::from_vec(
        slat_feats.iter().copied().collect::<Vec<f32>>(),
        (64, 32),
        &device,
    )
    .expect("latent tensor");
    let shape_decoded = shape_decoder
        .decode(&coords, &latent, true)
        .expect("shape decode");
    drop(shape_decoder);

    let tex_decoder =
        Trellis2TexDecoder::load(vb.pp("tex_dec"), &Trellis2ShapeVaeConfig::tex_decoder())
            .expect("tex_dec load");
    let started = std::time::Instant::now();
    let tex_decoded = tex_decoder
        .decode_with_trace(&coords, &latent, &shape_decoded.subdiv_logits)
        .expect("tex decode");
    println!("tex decode in {:.1}s", started.elapsed().as_secs_f64());
    drop(tex_decoder);

    // Final coords equal the shape decode's final coords (same guides).
    assert_eq!(
        tex_decoded.coords, shape_decoded.coords,
        "guided tex decode coords must equal the shape decode coords"
    );

    // Final [N, 6] output shape from the golden manifest.
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(golden.join("manifest.json")).expect("manifest"))
            .expect("manifest json");
    let output_shape: Vec<usize> = manifest["tex_output_shape"]
        .as_array()
        .expect("tex_output_shape")
        .iter()
        .map(|v| v.as_u64().expect("u64") as usize)
        .collect();
    assert_eq!(output_shape, vec![tex_decoded.coords.len(), 6]);
    assert_eq!(
        tex_decoded.feats.dims(),
        &[tex_decoded.coords.len(), 6],
        "tex output channel count"
    );

    // Per-block parity: exact coords, F32-tolerance feats (same caps as the
    // shape golden replay; the measured maxima sit ~2.5e-6, two orders below
    // the caps).
    let tolerances: [(usize, f64, f64); 4] = [
        (0, 2.5e-4, 1.6e-4),
        (1, 2.5e-4, 1.6e-4),
        (2, 2.5e-4, 1.6e-4),
        (3, 2.5e-4, 1.6e-4),
    ];
    assert_eq!(tex_decoded.trace.len(), 36, "4 + 16 + 8 + 4 + 4 C2S blocks");
    for block in &tex_decoded.trace {
        let golden_coords: Array2<i32> = ndarray_npy::read_npy(golden.join(format!(
            "tex_blocks_{}_{}_coords.npy",
            block.stage, block.block
        )))
        .unwrap_or_else(|e| {
            panic!(
                "tex_blocks_{}_{}_coords.npy: {}",
                block.stage, block.block, e
            )
        });
        assert_eq!(
            golden_coords.dim().0,
            block.coords.len(),
            "stage {} block {} coord count",
            block.stage,
            block.block
        );
        for (row, coord) in block.coords.iter().enumerate() {
            assert_eq!(golden_coords[[row, 0]], 0, "batch column");
            assert_eq!(
                [
                    golden_coords[[row, 1]],
                    golden_coords[[row, 2]],
                    golden_coords[[row, 3]]
                ],
                *coord,
                "stage {} block {} row {row}",
                block.stage,
                block.block
            );
        }
        let golden_feats: Array2<f32> = ndarray_npy::read_npy(golden.join(format!(
            "tex_blocks_{}_{}_feats.npy",
            block.stage, block.block
        )))
        .unwrap_or_else(|e| {
            panic!(
                "tex_blocks_{}_{}_feats.npy: {}",
                block.stage, block.block, e
            )
        });
        assert_eq!(golden_feats.dim().1, block.feats.dim(1).expect("channels"));
        let (_, rel, rms) = compare(
            &flat(&block.feats),
            golden_feats
                .iter()
                .copied()
                .collect::<Vec<f32>>()
                .as_slice(),
            &format!("tex_blocks.{}.{}.feats", block.stage, block.block),
        );
        let cap = tolerances[block.stage.min(3)];
        assert!(
            rel <= cap.1,
            "tex_blocks.{}.{} rel {rel:.3e}",
            block.stage,
            block.block
        );
        assert!(
            rms <= cap.2,
            "tex_blocks.{}.{} rms {rms:.3e}",
            block.stage,
            block.block
        );
    }
}

/// Gated real-weight golden: the tex SLAT flow 4-step replay. The golden
/// `tex_noise || shape_slat_norm` concat must match bit-exact; the 4-step
/// Euler trace (interval [0.6, 0.9], cond-only guidance) and the final
/// normalized/de-normalized texture SLAT are compared with the same
/// tolerance scheme as the shape flow golden (bf16 CUDA reference vs F32).
#[test]
fn tex_dit_golden_replay() {
    let golden = slat_golden_dir();
    let npz_path = golden.join("slat_golden.npz");
    let model_file = canonical_model_file();
    if !npz_path.is_file() {
        eprintln!("SKIP: golden npz missing at {}", npz_path.display());
        return;
    }
    if !model_file.is_file() {
        eprintln!(
            "SKIP: canonical TRELLIS.2 weights not found at {}",
            model_file.display()
        );
        return;
    }

    let file = std::fs::File::open(&npz_path).expect("open npz");
    let mut npz = ndarray_npy::NpzReader::new(file).expect("npz reader");
    let cond_array = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix3>("cond")
        .expect("cond");
    let coords_array = npz
        .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix2>("coords_32")
        .expect("coords_32");
    let noise_array = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("tex_noise")
        .expect("tex_noise");
    let concat_array = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("tex_input_concat")
        .expect("tex_input_concat");
    let shape_norm_array = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("shape_slat_norm")
        .expect("shape_slat_norm");
    let x_t_array = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix3>("tex_x_t_steps")
        .expect("tex_x_t_steps");
    let x0_array = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix3>("tex_pred_x0_steps")
        .expect("tex_pred_x0_steps");
    let norm_array = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("tex_slat_norm")
        .expect("tex_slat_norm");
    let denorm_array = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("tex_slat_denorm")
        .expect("tex_slat_denorm");
    assert_eq!(cond_array.shape(), &[1, 1029, 1024]);
    assert_eq!(coords_array.shape(), &[3591, 4]);
    assert_eq!(noise_array.shape(), &[3591, 32]);
    assert_eq!(concat_array.shape(), &[3591, 64]);
    assert_eq!(x_t_array.shape(), &[4, 3591, 32]);
    let tokens = 3591usize;

    let coords: Vec<[i32; 3]> = (0..tokens)
        .map(|i| {
            let b = coords_array[[i, 0]];
            assert_eq!(b, 0, "single batch element");
            [
                coords_array[[i, 1]],
                coords_array[[i, 2]],
                coords_array[[i, 3]],
            ]
        })
        .collect();

    // Stage coverage: the tex_dit_512 stage carries exactly the 640 tensors
    // the config expects.
    let stage_keys =
        read_safetensors_stage_keys(&model_file, "tex_dit_512.").expect("canonical package header");
    assert_eq!(stage_keys.len(), 640, "tex_dit_512 stage tensor count");
    let config = Trellis2SlatConfig::tex_512();
    let expected: HashSet<String> = config.expected_tensor_names().into_iter().collect();
    let stage: HashSet<String> = stage_keys
        .iter()
        .map(|k| k.trim_start_matches("tex_dit_512.").to_string())
        .collect();
    assert_eq!(expected, stage, "expected names must cover the stage");

    // Concat parity (bit-exact): noise first, golden shape_slat_norm second.
    let noise = Tensor::from_vec(
        noise_array.iter().copied().collect(),
        (tokens, 32),
        &Device::Cpu,
    )
    .expect("tex noise");
    let shape_norm = Tensor::from_vec(
        shape_norm_array.iter().copied().collect(),
        (tokens, 32),
        &Device::Cpu,
    )
    .expect("shape_slat_norm");
    let concat = concat_tex_input(&noise, &shape_norm).expect("concat");
    let concat_v = flat(&concat);
    let golden_concat: Vec<f32> = concat_array.iter().copied().collect();
    // The golden concat was captured inside the reference graph; the stored
    // operands re-derive it to the last bits (measured max_abs 2.4e-7), so
    // the parity check uses a last-bit-scale cap instead of bit equality.
    let (max_abs, rel, rms) = compare(&concat_v, &golden_concat, "tex_input_concat");
    assert!(max_abs <= 1e-6, "tex_input_concat max_abs {max_abs:.3e}");
    assert!(rel <= 1e-6, "tex_input_concat rel {rel:.3e}");
    assert!(rms <= 1e-6, "tex_input_concat rms {rms:.3e}");

    let device = Device::Cpu;
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[model_file.clone()], DType::F32, &device)
    }
    .expect("mmaped canonical weights");
    let load_started = std::time::Instant::now();
    let dit = Trellis2TexDit::load(vb.pp("tex_dit_512"), &config).expect("tex_dit_512 load");
    println!(
        "tex_dit_512 loaded in {:.1}s",
        load_started.elapsed().as_secs_f64()
    );

    let cond = Tensor::from_vec(
        cond_array.iter().copied().collect(),
        (1, 1029, 1024),
        &device,
    )
    .expect("cond tensor");

    // Golden settings: steps 4, guidance_strength 1.0 (cond-only),
    // guidance_rescale 0.0, interval [0.6, 0.9], rescale_t 3.0,
    // sigma_min 1e-5.
    let params = Trellis2SamplerParams {
        steps: 4,
        sigma_min: 1e-5,
        guidance_strength: 1.0,
        guidance_rescale: 0.0,
        guidance_interval: [0.6, 0.9],
        rescale_t: 3.0,
    };
    let forward_started = std::time::Instant::now();
    let model = |x: &Tensor, t_model: f32, c: &Tensor, cs: &[[i32; 3]]| {
        dit.forward(x, &shape_norm, cs, t_model, c)
    };
    let (samples, trace) =
        sample_sparse(&model, &coords, &noise, &cond, &cond, &params).expect("sample");
    println!(
        "4-step tex sampler forward in {:.1}s",
        forward_started.elapsed().as_secs_f64()
    );
    assert_eq!(samples.dims(), &[tokens, 32]);
    assert_eq!(trace.t_pairs.len(), 4);
    println!("tex t_pairs: {:?}", trace.t_pairs);

    // Same tolerance scheme as the shape flow golden: the reference velocity
    // carries ~3% relative bf16 noise; the Euler update averages it over dt
    // and the 30-block trajectory amplifies accumulated drift.
    let x_t_cap = [2e-2_f64, 2e-2, 2e-2, 1.5e-1];
    let x0_cap = [8e-2_f64, 8e-2, 8e-2, 3e-1];
    for step in 0..4 {
        let actual = trace.x_t_steps[step]
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("x_t");
        let golden_x_t: Vec<f32> = x_t_array
            .index_axis(ndarray::Axis(0), step)
            .iter()
            .copied()
            .collect();
        let (_max, rel, rms) = compare(&actual, &golden_x_t, &format!("tex x_t step {step}"));
        assert!(rel <= x_t_cap[step], "tex x_t step {step} rel {rel:.3e}");
        assert!(
            rms <= x_t_cap[step],
            "tex x_t step {step} rms_rel {rms:.3e}"
        );

        let actual_x0 = trace.pred_x0_steps[step]
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("x0");
        let golden_x0: Vec<f32> = x0_array
            .index_axis(ndarray::Axis(0), step)
            .iter()
            .copied()
            .collect();
        let (_max, rel, rms) = compare(&actual_x0, &golden_x0, &format!("tex pred_x0 step {step}"));
        assert!(rel <= x0_cap[step], "tex x0 step {step} rel {rel:.3e}");
        assert!(rms <= x0_cap[step], "tex x0 step {step} rms_rel {rms:.3e}");
    }

    // Final normalized SLAT vs the saved output.
    let actual_norm = samples
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("slat_norm");
    let golden_norm: Vec<f32> = norm_array.iter().copied().collect();
    let (_norm_max, norm_rel, norm_rms) = compare(&actual_norm, &golden_norm, "tex_slat_norm");
    assert!(norm_rel <= 1.5e-1, "tex_slat_norm rel {norm_rel:.3e}");
    assert!(norm_rms <= 6e-2, "tex_slat_norm rms_rel {norm_rms:.3e}");

    // Denorm with the manifest texture vectors.
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(golden.join("manifest.json")).expect("manifest"))
            .expect("manifest json");
    let mean: Vec<f32> = manifest["tex_normalization"]["mean"]
        .as_array()
        .expect("mean")
        .iter()
        .map(|v| v.as_f64().expect("f64") as f32)
        .collect();
    let std: Vec<f32> = manifest["tex_normalization"]["std"]
        .as_array()
        .expect("std")
        .iter()
        .map(|v| v.as_f64().expect("f64") as f32)
        .collect();
    assert_eq!(mean.len(), 32);
    let mut denorm = vec![0.0_f32; actual_norm.len()];
    for row in 0..tokens {
        for c in 0..32 {
            denorm[row * 32 + c] = actual_norm[row * 32 + c] * std[c] + mean[c];
        }
    }
    let golden_denorm: Vec<f32> = denorm_array.iter().copied().collect();
    let (_denorm_max, denorm_rel, denorm_rms) = compare(&denorm, &golden_denorm, "tex_slat_denorm");
    assert!(denorm_rel <= 2e-1, "tex_slat_denorm rel {denorm_rel:.3e}");
    assert!(
        denorm_rms <= 1e-1,
        "tex_slat_denorm rms_rel {denorm_rms:.3e}"
    );
}

/// Gated real-weight end-to-end texture run (CUDA): the full 512 pipeline
/// (SS -> shape SLAT -> tex SLAT -> shape VAE -> guided tex VAE) with the
/// texture products attached to the mesh and the PBR export pair on disk.
/// Gated behind `LUX3D_TRELLIS2_RUN_TEX_E2E=1` and the `cuda` feature;
/// `LUX3D_TRELLIS2_TEX_E2E_STEPS` overrides all three Euler step counts
/// (use 2 for a smoke pass).
#[cfg(feature = "cuda")]
#[test]
fn tex_pipeline_cuda_e2e_real_weights() {
    use lux3d_core::contracts::Trellis2PipelineType;
    use lux3d_core::runtime::{Trellis2Pipeline, Trellis2RunOptions};

    if std::env::var("LUX3D_TRELLIS2_RUN_TEX_E2E").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set LUX3D_TRELLIS2_RUN_TEX_E2E=1 to run the CUDA texture e2e");
        return;
    }
    let raw = match std::env::var_os("LUX3D_TRELLIS2_RAW_DIR") {
        Some(path) => std::path::PathBuf::from(path),
        None => workspace_root().join("models").join("TRELLIS.2-4B"),
    };
    if !raw.is_dir() {
        eprintln!(
            "SKIP: TRELLIS.2 raw checkpoint dir not found at {}",
            raw.display()
        );
        return;
    }
    let device = candle_core::Device::new_cuda(0).expect("CUDA device");

    // Synthetic RGBA input: opaque rounded square on transparent background.
    let temp = std::env::temp_dir().join(format!("lux3d-trellis2-tex-{}", std::process::id()));
    std::fs::create_dir_all(&temp).expect("temp dir");
    let image_path = temp.join("input.png");
    {
        let mut image = image::RgbaImage::new(96, 96);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let inside = (16..80).contains(&x) && (16..80).contains(&y);
            *pixel = if inside {
                image::Rgba([180, 120, 60, 255])
            } else {
                image::Rgba([0, 0, 0, 0])
            };
        }
        image.save(&image_path).expect("save input png");
    }

    let steps: usize = std::env::var("LUX3D_TRELLIS2_TEX_E2E_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let pipeline = Trellis2Pipeline::load_from_raw_dir(&raw, &device).expect("raw-dir pipeline");
    let options = Trellis2RunOptions {
        pipeline_type: Trellis2PipelineType::R512,
        ss_steps: steps,
        shape_steps: steps,
        tex_steps: steps,
        ..Trellis2RunOptions::default()
    };

    let started = std::time::Instant::now();
    let stages = pipeline
        .run_from_path_with_stages(&image_path, &device, &options)
        .expect("texture e2e run");
    let elapsed = started.elapsed().as_secs_f64();

    // Texture stage products present: latent on the shape latent coords,
    // PBR attrs on the guided decode's FINAL (subdivided) coords, one guide
    // block per C2S stage.
    let tex_latent = stages.tex_latent.as_ref().expect("tex latent");
    let tex_attrs = stages.tex_attrs.as_ref().expect("tex attrs");
    assert_eq!(stages.subdiv_logits.len(), 4);
    assert_eq!(
        tex_latent.dim(0).expect("rows"),
        stages.shape_coords.len(),
        "tex latent rows == shape latent coords"
    );
    let decoded_rows = stages.decoded_feats.dim(0).expect("decoded rows");
    assert_eq!(
        tex_attrs.len(),
        decoded_rows * Trellis2PbrLayout::CHANNELS,
        "tex attr rows == shape decode voxel count (guided C2S)"
    );
    assert!(
        decoded_rows >= stages.shape_coords.len(),
        "subdivided decode must cover every latent voxel"
    );

    // Mesh carries the PBR payload + vertex colors.
    let mesh = stages.mesh.expect("mesh");
    assert_eq!(mesh.voxel_attr_channels, Trellis2PbrLayout::CHANNELS);
    assert_eq!(
        mesh.voxel_count(),
        decoded_rows,
        "mesh voxels == decode rows"
    );
    assert!(
        mesh.voxel_attrs.iter().all(|&v| (0.0..=1.0).contains(&v)),
        "PBR attrs clamped to [0, 1]"
    );
    let colors = mesh.vertex_colors.as_ref().expect("vertex colors");
    assert_eq!(colors.len(), mesh.vertex_count() * 3);
    assert!(colors.iter().all(|&v| (0.0..=1.0).contains(&v)));
    mesh.validate().expect("mesh valid");
    println!(
        "texture e2e: {steps} steps, {} voxels, {} vertices, {elapsed:.1}s",
        mesh.voxel_count(),
        mesh.vertex_count(),
    );

    // PBR export pair round-trips.
    let stage = lux3d_core::export::Trellis2ExportStage;
    let glb_path = temp.join("mesh.glb");
    let sidecar = stage
        .write_mesh_with_pbr(&mesh, &glb_path)
        .expect("mesh + pbr export");
    assert!(glb_path.is_file());
    assert!(sidecar.is_file());
    let json_path = temp.join("mesh.pbr.json");
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&json_path).expect("json")).expect("json parse");
    assert_eq!(json["resolution"], mesh.resolution);
    assert_eq!(json["voxel_count"], mesh.voxel_count());
    let bin_bytes = std::fs::read(&sidecar).expect("sidecar bytes");
    assert_eq!(
        &bin_bytes[..8],
        lux3d_core::export::pbrvox::MAGIC.as_slice()
    );
    let count = u64::from_le_bytes(bin_bytes[16..24].try_into().expect("count"));
    assert_eq!(count, mesh.voxel_count() as u64);
    println!("texture e2e export at {}", temp.display());
    let _ = std::fs::remove_dir_all(&temp);
}
