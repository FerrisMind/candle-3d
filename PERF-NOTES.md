# Performance notes: measured results and remaining levers (RTX 3060 12 GiB, Windows WDDM)

Status: 2026-09-06, candle `wgpu/vulkan` rev `bb197082`, candle-3d `317d4e7`.

## Verified state

Inference wall (`[stage] infer`, single iteration per process):

| model | cuda | vulkan | wgpu | vulkan/cuda | wgpu/cuda |
|---|---:|---:|---:|---:|---:|
| pi3 (5 frames, 518x518) | 4.82s | 7.66s | 82.9s | 1.59x | 17.2x |
| pi3x (6 frames, 518x518) | 7.00s | 11.43s | 88.7s | 1.63x | 12.7x |
| triposr (single image) | 1.23s | 3.19s | 6.09s | 2.59x | 5.0x |

Memory is flat under repeated inference on all three backends (wgpu 10-iteration
bench: rc=0, VRAM peak 9.65 GiB, zero errors; cuda pool step eliminated by
SDPA chunking; vulkan bounded by grace-gated inflight drain + 2 GiB pool).

All mesh comparisons vs the cuda reference PASS on every backend.

## Measured dead ends (do not retry on this hardware)

- single-submission graph (huge batch caps): 27.3s — VRAM thrash on 12 GiB.
- CANDLE_VK_MAX_BATCH_DISPATCHES=512 + compute 4 GiB: 14.9s; 1024/8 GiB: 18.0s.
- wgpu batch 64: 6x WORSE (uniform ring reuse waits); grace band on wgpu: worse
  (retirement only advances inside device.poll).
- scalar flash_attn on vulkan: 29.9s; coopmat FA2 (`flash_attn_f32_f16_f32_cm1`,
  enabled for the measurement): 12.83s vs 11.43s chunked — thousands of small
  per-tile dispatches pay the WDDM fence-signal tax and f32 K/V locality is
  worse than two large coopmat GEMMs. Chunked two-pass stays the default.

## Warm-loop criterion results (2026-09-06, rev e0e24758)

10-sample criterion runs (steady state, weights resident — the candle-bench
protocol):

| bench | cuda | vulkan | ratio |
|---|---:|---:|---:|
| pi3/iter | 2.80s | 3.34s | 1.19x |
| pi3x/iter | 4.21s | 5.13s | 1.22x |

The warm gap is much smaller than the single-shot gap (cold allocator/first
iteration dominates the single-shot numbers).

## MUL_MAT_ADD — implemented (guarded)

mul_mm.comp BIAS_ADD epilogue + `_bias` variants from vulkan-shaders-gen +
`VulkanStorage::matmul_bias` + `candle_nn::ops::mul_mat_add` (CustomOp3,
vulkan-routed from lux3d `linear_fwd`). Fused ONLY for verified shapes
(m%64==0 && n%64==0 && k%32==0, m>8, f32 -> matmul_f32_f32_aligned_cm1_bias);
the unaligned cm1 staged-store epilogue writes bias without the matmul
contribution on some tiles, so those shapes stay unfused (llama.cpp upstream
does not fuse bias into coopmat either). Unit test: fused vs unfused on the
same device.

## Remaining levers (integration plans)

### 1. MUL_MAT_ADD — GEMM with bias epilogue (vulkan)

Expected gain: <=5-10% of vulkan wall. Batches close on compute_bytes (big GEMM
ops), so removing ~500 bias-add dispatches out of 9477 reduces submissions only
indirectly; the direct saving is one activation-size read+write per linear.

Plan:
1. mul_mm.comp: add `#ifdef BIAS_ADD` binding 5 (`readonly buffer BIAS
   {float data_bias[];}`); at each store site (coopMatStore paths ~484/496,
   scalar ~526/529) add `data_bias[col]` where col is the output column index
   already computed at that site.
2. build.rs: register `matmul_f32_f32_fp32_bias` / `..._cm1_bias` variants
   (copy the non-bias entries + BIAS_ADD define) — coopmat glslc support is
   already probed at line 80.
3. vulkan_backend.rs: extend the matmul descriptor layout with the 5th
   binding only for bias pipelines; new `pub fn matmul_bias(...)` that mirrors
   `run_matmul_f32` and passes the bias storage.
4. lux3d-core: `CustomOp3` FusedLinear (cpu/cuda fwd = matmul+add fallback;
   vulkan_fwd = matmul_bias) applied only on vulkan devices at Linear call
   sites (pi3_encoder.rs, pi3_decoder.rs, pi3x.rs).
Risk: descriptor-layout plumbing in vulkan_backend.rs (pipeline registry) is
the only unfamiliar part; everything else is mechanical.

### 2. WGSL GEMM (wgpu)

Before writing a third kernel: build a wgpu GPU-timestamp profiler (mirror the
vulkan one) to attribute the remaining ~80s of pi3x wall between GPU and CPU.
The backend already ships matmul_warptile (64x64 tile, BK=32) and coop64
(0.53ms/1024^3 measured); wgpu-llm's gemm.wgsl (16x16 tile) is simpler than
either and unlikely to win. If attribution shows GEMM time dominates, port the
born matmul tile scheme with subgroup ops instead.

### 3. Attention on vulkan

Chunked two-pass with coopmat GEMMs is the measured optimum for F32 vision
attention on WDDM. Revisit only after MUL_MAT_ADD reduces the dispatch count.

## Harnesses

- bench: `LUX3D_BENCH_DEVICE=... LUX3D_BENCH_MODEL=... cargo bench -p
  lux3d-core --bench bench_main --features vulkan,wgpu -- <backend>_<model>`
  (one process per pair; `.swarm/tools/bench_one.sh` adds VRAM/RAM sampling).
- profiles: `CANDLE_VULKAN_CPU_PROFILE=1` (per-phase CPU + flush reasons),
  `CANDLE_VULKAN_GPU_PROFILE=1` + `LUX3D_GPU_PROFILE` (kernel report).
- correctness: `python .swarm/tools/compare_mesh.py compare <ref> <out>`
  against `.swarm/out/perfprof/*_cuda_ref.*`.
