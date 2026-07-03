# Lux3D / candle-3d — Feature Inventory & User Stories

## Legend
| Status | Meaning |
|--------|---------|
| ✅ Done | Implemented & tested |
| 🔍 Review | Code exists, needs testing |
| 🐛 Bug | Issue found |
| ❌ Missing | Not implemented |
| ⏳ Pending | Planned |

---

## 1. CLI — Entry & Dispatch

### US-1.1: CLI parses subcommands
As a user, I can run `lux3d inspect|weights normalize|run <family>` and get correct dispatch.

**Functions:**
- `main.rs:main()` — Entry point ✅
- `cli.rs:Cli` — Clap parser struct ✅
- `cli.rs:Command::Inspect` / `Weights` / `Run` — Subcommand enum ✅
- `cli.rs:Family` — Pi3 / Pi3x / Triposr enum ✅
- `cli.rs:InjectCondition` — Pose / Depth / Ray / Intrinsic enum ✅

**Tests:** `tests/run_args.rs` ✅

---

### US-1.2: Inspect contracts
As a user, I can run `lux3d inspect --repo-root <dir> <family>` and get a JSON spec.

**Functions:**
- `cli.rs:inspect_model()` — Reads ModelSpec from repo root ✅

**Tests:** None found 🔍

---

### US-1.3: Normalize weights
As a maintainer, I can run `lux3d weights normalize` to canonicalize raw weights.

**Functions:**
- `cli.rs:normalize_weights()` — Calls Python baseline script ✅
- `cli.rs:python_executable()` — Resolves Python binary ✅
- `cli.rs:canonical_package_dir()` — Finds canonical weight package ✅

**Tests:** None found 🔍

---

### US-1.4: Run inference (CLI)
As a user, I can run `lux3d run <family> --source <input> --output <file>` to produce 3D output.

**Functions:**
- `cli.rs:run_model()` — Main inference dispatcher ✅
- `cli.rs:RunArgs` — All run parameters ✅
- `cli.rs:Command::run()` — Builds RunArgs from parsed CLI ✅

**Tests:** `tests/run_args.rs` covers model_path & cache_dir ✅

---

## 2. Pi3 Pipeline

### US-2.1: Pi3 loads model weights
As a user, the Pi3 pipeline loads canonical weights from disk or HuggingFace cache.

**Functions:**
- `runtime/mod.rs:Pi3Pipeline::load()` ✅
- `runtime/mod.rs:Pi3ModelBundle::load()` — Loads encoder, decoder, heads ✅
- `runtime/mod.rs:Pi3ModelBundle` — struct with 8 sub-modules ✅
- `runtime/mod.rs:bundle_for()` — DeviceLocalCache wrapper ✅
- `runtime/mod.rs:DeviceLocalCache` — Per-device singleton cache ✅

**Tests:** `runtime/mod.rs:pi3_encoder_reuses_cached_interpolated_positional_embeddings` ✅

---

### US-2.2: Pi3 preprocesses input
As a user, the Pi3 pipeline resizes input frames to a patch-multiple size.

**Functions:**
- `preprocess/pi3.rs:Pi3PreprocessStage::prepare_inputs_from_path()` ✅
- `preprocess/pi3.rs:Pi3PreprocessStage::target_size_for()` — Computes patch-aligned target size ✅
- `preprocess/pi3.rs:Pi3PreprocessStage::rescale_intrinsics()` ✅
- `runtime/mod.rs:prepare_pi3_inputs_with_stage()` ✅
- `runtime/mod.rs:load_sampled_pi3_frames()` — Directory or MP4 input ✅
- `runtime/mod.rs:collect_rgb_entries()` — Filters PNG/JPG/JPEG ✅
- `runtime/mod.rs:extract_video_frames()` — ffmpeg frame extraction ✅

**Tests:** `preprocess/pi3.rs:rescales_intrinsics_using_target_size` ✅

---

### US-2.3: Pi3 encodes frames with DINOv2
As the model, patch tokens are extracted from normalized frames.

**Functions:**
- `runtime/pi3_encoder.rs:Pi3DinoEncoder::forward_patch_tokens()` — Full forward pass ✅
- `runtime/pi3_encoder.rs:Pi3DinoEncoder::prepare_tokens()` — Patch embed + cls + register + pos ✅
- `runtime/pi3_encoder.rs:Pi3DinoEncoder::interpolate_pos_encoding()` — Cached bicubic resize ✅
- `runtime/pi3_encoder.rs:PatchEmbed::forward()` — Conv2d patch embedding ✅
- `runtime/pi3_encoder.rs:Block` — Pre-normalized ViT block with LayerScale ✅
- `runtime/pi3_encoder.rs:Attention` — QKV attention ✅
- `runtime/pi3_encoder.rs:resize_patch_pos_embed()` — AA-cubic interpolation ✅

**Tests:** `cached_pos_encoding_entries()` assertion ✅

---

### US-2.4: Pi3 decodes hidden states
As the model, encoder outputs are decoded into point/confidence/camera representations.

**Functions:**
- `runtime/pi3_decoder.rs:Pi3Decoder::decode()` — 36-block RoPE decoder ✅
- `runtime/pi3_decoder.rs:Pi3Decoder::decoder_positions()` — Grid + special token positions ✅
- `runtime/pi3_decoder.rs:RopeBlock` — Transformer block with RoPE attention ✅
- `runtime/pi3_decoder.rs:RopeAttention` — Q/K norm + RoPE + flash attention ✅
- `runtime/pi3_decoder.rs:Pi3BranchDecoder` — 5-block branch decoder ✅
- `runtime/pi3_decoder.rs:BranchBlock` — Simplified branch block (no LayerScale) ✅

**Tests:** None found 🔍

---

### US-2.5: Pi3 produces point cloud heads
As the model, local points, confidence, and camera poses are predicted.

**Functions:**
- `runtime/pi3_heads.rs:LinearPts3dHead::forward()` — Pixel-shuffle upsampling ✅
- `runtime/pi3_heads.rs:CameraHead::forward()` — Pooled camera prediction ✅
- `runtime/pi3_heads.rs:ResConvBlock` — Residual conv block ✅

**Tests:** None found 🔍

---

### US-2.6: Pi3 assembles geometry
As the model, local points are transformed to world space and filtered.

**Functions:**
- `runtime/mod.rs:Pi3Pipeline::encode_patch_tokens()` ✅
- `runtime/mod.rs:Pi3Pipeline::decode_hidden()` ✅
- `runtime/mod.rs:Pi3Pipeline::decode_positions_only()` ✅
- `runtime/mod.rs:Pi3Pipeline::point_decoder_hidden()` ✅
- `runtime/mod.rs:Pi3Pipeline::conf_decoder_hidden()` ✅
- `runtime/mod.rs:Pi3Pipeline::camera_decoder_hidden()` ✅
- `runtime/mod.rs:Pi3Pipeline::point_head_output()` ✅
- `runtime/mod.rs:Pi3Pipeline::conf_head_output()` ✅
- `runtime/mod.rs:Pi3Pipeline::camera_poses_from_hidden()` ✅
- `runtime/mod.rs:Pi3Pipeline::world_points()` ✅
- `runtime/mod.rs:Pi3Pipeline::export_mask()` ✅
- `runtime/mod.rs:cast_tensor_to_f32()` ✅

**Tests:** None found 🔍

---

### US-2.7: Pi3 exports to PLY
As a user, the output point cloud is written to a PLY file.

**Functions:**
- `export/pi3.rs:Pi3ExportStage::write_ply()` — ASCII PLY writer ✅
- `geometry/pi3.rs:Pi3GeometryStage::assemble_cpu()` — Mask-based filtering ✅
- `geometry/pi3.rs:Pi3GeometryStage::assemble()` — Contract assembly ✅

**Tests:** CLI test `run_pi3_smoke_emits_ply_output` (in cli.rs) ✅

---

## 3. Pi3X Pipeline

### US-3.1: Pi3X loads extended weights
As the model, Pi3X loads encoder + depth encoder + ray embed + metric modules.

**Functions:**
- `runtime/pi3x.rs:Pi3xModelBundle::load()` — Loads 15 sub-modules ✅
- `runtime/pi3x.rs:Pi3xPipeline::load()` ✅
- `runtime/pi3x.rs:Pi3xPatchEmbed` — Conv2d patch embedding for rays ✅
- `runtime/pi3x.rs:Pi3xCoreDecoder` — 36-block decoder with projected cross-attention ✅
- `runtime/pi3x.rs:Pi3xCrossBlock` — Block with cross-attn to pose/depth features ✅
- `runtime/pi3x.rs:Pi3xConvHead` — Convolutional output head ✅
- `runtime/pi3x.rs:Pi3xContextOnlyTransformerDecoder` — Metric decoder ✅

**Tests:** None found 🔍

---

### US-3.2: Pi3X processes frames and conditions
As a user, Pi3X accepts RGB frames + optional condition NPZ files.

**Functions:**
- `runtime/pi3x.rs:Pi3xPipeline::infer_prepared()` — Full inference pipeline ✅
- `runtime/pi3x.rs:Pi3xPipeline::prepare_inputs_from_path()` ✅
- `runtime/pi3x.rs:load_sampled_pi3x_frames()` — Directory or MP4 ✅
- `runtime/pi3x.rs:prepare_pi3x_inputs_with_stage()` ✅
- `runtime/pi3x.rs:Pi3xPreparedInputs` struct ✅
- `runtime/pi3x.rs:load_pi3x_conditions()` — NPZ condition loading ✅
- `runtime/pi3x.rs:Pi3xPipeline::cross_blocks_with_inject()` — Condition injection ✅
- `runtime/pi3x.rs:Pi3xPipeline::infer_vo_from_path()` — Video object pipeline ✅
- `runtime/pi3x.rs:Pi3xPipeline::infer_vo()` — Chunked VO inference ✅
- `runtime/pi3x.rs:Pi3xPipeline::infer_vo_chunk()` ✅
- `runtime/pi3x.rs:build_pi3x_prefix_mask()` — Causal masking ✅
- `runtime/pi3x.rs:repeated_identity_poses()` ✅

**Tests:** CLI tests `run_pi3x_smoke_emits_ply_output`, `run_pi3x_vo_inject_smoke_emits_ply_output` ✅

---

### US-3.3: Pi3X ray and depth embedding
As the model, rays and depth maps are embedded into the decoder.

**Functions:**
- `runtime/pi3x.rs:Pi3xPipeline::embed_rays_and_depth()` ✅
- `runtime/pi3x.rs:Pi3xPipeline::build_ray_embeds()` ✅
- `runtime/pi3x.rs:resize_depth_nearest()` ✅

**Tests:** None found 🔍

---

### US-3.4: Pi3X metric head
As the model, a metric value is predicted per scene.

**Functions:**
- `runtime/pi3x.rs:Pi3xPipeline::run_metric_head()` ✅
- `runtime/pi3x.rs:Pi3xMetricCache` struct ✅

**Tests:** None found 🔍

---

## 4. TripoSR Pipeline

### US-4.1: TripoSR loads model
As the model, TripoSR loads image tokenizer, backbone, post-processor, and NeRF decoder.

**Functions:**
- `runtime/triposr.rs:TripoSrModelBundle::load()` ✅
- `runtime/triposr.rs:TripoSrPipeline::load()` ✅
- `runtime/triposr.rs:TripoImageTokenizer` — DINO ViT-based encoder ✅
- `runtime/triposr.rs:Triplane1DTokenizer` — Learned triplane tokens ✅
- `runtime/triposr.rs:Transformer1D` — 16-block transformer backbone ✅
- `runtime/triposr.rs:BasicTransformerBlock1d` — Dual cross-attention block ✅
- `runtime/triposr.rs:CrossAttention` — Q/K/V cross-attn ✅
- `runtime/triposr.rs:TriplaneUpsampleNetwork` — Transposed conv upsampler ✅
- `runtime/triposr.rs:NerfMlp` — 10-layer NeRF MLP ✅

**Tests:** None found 🔍

---

### US-4.2: TripoSR preprocesses image
As a user, an input image is resized, composited over gray, and normalized.

**Functions:**
- `preprocess/triposr.rs:TripoPreprocessStage::prepare_inputs_from_path()` ✅
- `preprocess/triposr.rs:TripoPreprocessStage::composite_over_gray()` ✅
- `preprocess/triposr.rs:TripoPreprocessStage::normalize_rgb()` ✅
- `runtime/triposr.rs:prepare_triposr_inputs_with_stage()` ✅

**Tests:** None found 🔍

---

### US-4.3: TripoSR generates scene codes
As the model, image tokens are transformed into triplane scene codes.

**Functions:**
- `runtime/triposr.rs:TripoSrPipeline::scene_codes()` ✅
- `runtime/triposr.rs:TripoSrPipeline::image_tokens()` ✅
- `runtime/triposr.rs:TripoSrPipeline::seed_tokens()` ✅
- `runtime/triposr.rs:TripoSrPipeline::backbone_forward()` ✅
- `runtime/triposr.rs:TripoSrPipeline::detokenize_triplanes()` ✅
- `runtime/triposr.rs:TripoSrPipeline::query_triplane()` ✅

**Tests:** None found 🔍

---

### US-4.4: TripoSR runs marching cubes
As a user, the scene codes are converted to a mesh.

**Functions:**
- `geometry/triposr.rs:TripoGeometryStage::assemble_cpu()` — Marching cubes with dedup ✅
- `geometry/triposr.rs:TripoGeometryStage::attach_vertex_colors()` ✅
- `geometry/triposr.rs:TripoGeometryStage::materialize_buffers()` ✅
- `geometry/triposr.rs:TripoGeometryStage::cpu_from_buffers()` ✅
- `neural/triposr.rs:TripoNeuralStage::build_density_grid_cpu()` — Chunked density extraction ✅
- `runtime/triposr_field.rs:query_triplane_chunked()` ✅
- `runtime/triposr_field.rs:query_triplane_density_chunked()` ✅
- `runtime/triposr_field.rs:query_triplane_color_chunked()` ✅
- `runtime/triposr_field.rs:TriplaneDecoder` trait ✅

**Tests:** CLI test `run_triposr_smoke_emits_obj_output` ✅
**Tests:** CLI test `run_triposr_rejects_invalid_marching_cubes_resolution` ✅

---

### US-4.5: TripoSR exports to OBJ
As a user, the mesh is written to an OBJ file.

**Functions:**
- `export/triposr.rs:TripoExportStage::write_obj()` — OBJ writer ✅

**Tests:** Via CLI smoke test ✅

---

## 5. Attention & Math Utilities

### US-5.1: Rotary Position Embedding (RoPE)
As the model, 2D RoPE is applied to attention queries and keys.

**Functions:**
- `runtime/attention_math.rs:Rope2d::embeddings()` ✅
- `runtime/attention_math.rs:Rope2d::apply_with_embeddings()` ✅
- `runtime/attention_math.rs:Rope2d::cos_sin()` ✅
- `runtime/attention_math.rs:Rope2d::apply_embedding()` ✅
- `runtime/attention_math.rs:Rope2d::apply_rope_embedded()` ✅
- `runtime/attention_math.rs:RopeEmbeddings` struct ✅
- `runtime/attention_math.rs:position_getter()` ✅
- `runtime/attention_math.rs:rotate_half()` ✅
- `runtime/attention_math.rs:exact_query_chunked_sdpa()` ✅
- `runtime/attention_math.rs:exact_sdpa_heads()` ✅

**Tests:** None found 🔍

---

### US-5.2: Resampling
As the model, bicubic and linear resampling is used for positional encoding.

**Functions:**
- `runtime/resampling.rs:compute_aa_cubic_weights()` ✅
- `runtime/resampling.rs:compute_aa_linear_weights()` ✅
- `runtime/resampling.rs:cubic_weight()` ✅
- `runtime/resampling.rs:clamp_isize()` ✅

**Tests:** None found 🔍

---

### US-5.3: Camera operations
As the model, 3D camera pose is computed from rotation components and translation.

**Functions:**
- `runtime/point_camera_ops.rs:camera_pose_from_components()` ✅
- `runtime/point_camera_ops.rs:world_points_from_local_and_pose()` ✅
- `runtime/point_camera_ops.rs:export_mask_from_local_points_and_confidence()` ✅
- `runtime/point_camera_ops.rs:non_edge_mask_from_local_points()` ✅

**Tests:** None found 🔍

---

## 6. Neural Stage Controllers

### US-6.1: Pi3 neural stage
The neural stage wires Pi3 pipeline outputs to contract types.

**Functions:**
- `neural/pi3.rs:Pi3NeuralStage::infer()` — Contract tensor shapes ✅
- `neural/pi3.rs:Pi3NeuralStage::infer_tensors()` — Full inference pipeline ✅

**Tests:** None found 🔍

---

### US-6.2: Pi3X neural stage
The neural stage wires Pi3X pipeline outputs.

**Functions:**
- `neural/pi3x.rs:Pi3xNeuralStage::infer()` ✅
- `neural/pi3x.rs:Pi3xNeuralStage::infer_tensors()` ✅
- `neural/pi3x.rs:Pi3xNeuralStage::infer_vo_from_path()` ✅

**Tests:** None found 🔍

---

### US-6.3: TripoSR neural stage
The neural stage wires TripoSR pipeline outputs.

**Functions:**
- `neural/triposr.rs:TripoNeuralStage::infer()` ✅
- `neural/triposr.rs:TripoNeuralStage::infer_tensors()` ✅
- `neural/triposr.rs:TripoNeuralStage::build_density_grid_cpu()` ✅

**Tests:** None found 🔍

---

## 7. Weight Management

### US-7.1: Load canonical weights
As the runtime, canonical safetensors weights are loaded from disk or HF cache.

**Functions:**
- `weights/mod.rs:load_canonical_weights()` ✅
- `weights/mod.rs:ensure_canonical_weights()` — HF auto-download ✅
- `weights/mod.rs:CanonicalWeightSetPaths` — File path bundle ✅
- `weights/mod.rs:CanonicalWeightSetPaths::var_builder()` — Unsafe mmap builder ✅
- `weights/mod.rs:ModelAssetOptions` struct ✅
- `weights/mod.rs:WeightLocator` — HF or local resolution ✅

**Tests:** None found 🔍

---

### US-7.2: Validate weight checksums
As a maintainer, canonical package integrity is verified via SHA-256.

**Functions:**
- `weights/mod.rs:verify_checksums()` ✅
- `weights/mod.rs:CanonicalChecksums` — Serialization struct ✅
- `weights/mod.rs:CanonicalWeightsManifest` — Manifest struct ✅

**Tests:** None found 🔍

---

## 8. Contract System

### US-8.1: Model specification contracts
As a maintainer, model specs define the expected architecture and data flow.

**Functions:**
- `contracts/spec.rs:ModelSpec` — Full model specification ✅
- `contracts/spec.rs:ModelSpec::inspect()` — Reads from repo root ✅
- `contracts/family.rs:ModelFamily` — 3-model enum ✅
- `contracts/baseline.rs:BaselineManifest` — Baseline artifact manifest ✅
- `contracts/baseline.rs:BaselineManifest::validate_against_model_spec()` ✅
- `contracts/stages.rs:PreprocessStage` / `NeuralStage` / `GeometryStage` / `ExportStage` traits ✅
- `contracts/common.rs:SpatialSize` / `TensorContract` / `NormalizationStats` ✅
- `contracts/pi3.rs` / `pi3x.rs` / `triposr.rs` — Family-specific contracts ✅

**Tests:** `tests/contracts.rs` ✅, `tests/model_spec.rs` ✅

---

## 9. Natural Sort Utility

### US-9.1: Natural filename sorting
As the runtime, filenames are sorted with natural number ordering (e.g., frame_2 before frame_10).

**Functions:**
- `runtime/path_utils.rs:sort_paths_natural()` ✅
- `runtime/path_utils.rs:compare_natural()` ✅
- `runtime/path_utils.rs:trim_leading_zeros()` ✅

**Tests:** `runtime/path_utils.rs:sorts_numeric_suffixes_in_human_order` ✅

---

## 10. Neural Network Building Blocks

### US-10.1: Shared NN components
Common neural network layers used across all pipelines.

**Functions:**
- `runtime/nn_blocks.rs:linear()` ✅
- `runtime/nn_blocks.rs:LayerScale` ✅
- `runtime/nn_blocks.rs:Mlp` — 4x hidden MLP with GeLU ✅
- `runtime/nn_blocks.rs:GeGlu` — Gated GeLU ✅
- `runtime/nn_blocks.rs:FeedForward` — GeGLU + Linear ✅

**Tests:** None found 🔍

---

## 11. Vision Preprocessing

### US-11.1: Image loading and resize
As the runtime, images are loaded and resized for model input.

**Functions:**
- `runtime/vision_preproc.rs` — Image loading pipeline ✅

**Tests:** None found 🔍

---

## 12. Utility & Error Handling

### US-12.1: Error types
The library defines typed errors for all failure modes.

**Functions:**
- `error.rs:Lux3dError` — 18 error variants ✅
- `error.rs:Result<T>` — Type alias ✅

**Tests:** None needed ✅

---

## Summary Statistics

| Category | Functions | Test Coverage |
|----------|-----------|---------------|
| CLI Dispatch | 14 | Partial |
| Pi3 Pipeline | 28 | Minimal |
| Pi3X Pipeline | 24 | Minimal |
| TripoSR Pipeline | 22 | Minimal |
| Attention Math | 10 | None |
| Resampling | 4 | None |
| Camera Ops | 4 | None |
| NN Blocks | 6 | None |
| Export | 4 | Via CLI |
| Geometry | 7 | Partial |
| Neural Stages | 6 | None |
| Weights | 8 | None |
| Contracts | 20 | Yes |
| Preprocess | 10 | Minimal |
| Path Utils | 3 | Yes |
| Error Types | 2 | N/A |
| Vision Preproc | 1 | None |
| **Total** | **~183** | **~10%** |

