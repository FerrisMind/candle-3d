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

## Warm-loop wgpu (2026-09-06, rev bba156a7, post-FA2)

| bench | cuda | wgpu | ratio |
|---|---:|---:|---:|
| pi3/iter | 2.80s | 76.9s | 27.5x |
| pi3x/iter | 4.21s | 76.3s | 18.1x |

Zero OOM/invalid across the 10-sample runs. The remaining gap is
GPU-kernel-bound (WGSL GEMM + elementwise quality vs cuBLAS); a
competitive WGSL GEMM remains the one open lever, sized as its own
project (see plan below).

## MUL_MAT_ADD batch>1 failure — ROOT-CAUSED (2026-09-06, rev 64bef917)

Not a barrier or stage-aliasing issue: presenting the weight with a
single real batch made the kernel's `batch_idx_a = i03*ne02 + i02` walk
past the 1-batch weight buffer for batches 1..n-1 — the (3,64,128,256)
probe lost exactly two full batch planes (bias-only output). Fix:
`candle_nn::ops::mul_mat_add` now feeds the weight as a stride-0
broadcast view (`unsqueeze(0).broadcast_as(batch,k,n)`), matching
candle's own broadcast_matmul; `batch_stride_a` becomes 0 and every
batch reads weight batch 0. The shape-matrix unit test is exact on ALL
shapes (aligned, unaligned cm1, batch>1); the alignment/batch gates are
removed.

Fusion remains opt-in (CANDLE_LUX3D_FUSED_LINEAR=1): on the 12 GiB WDDM
card fusing the dense pi3x linears is net-negative (warm criterion
5.13s -> 8.85s median + OOM risk — removing the add dispatches shifts
batch/allocator dynamics over the VRAM edge).

Measurement caveat learned the hard way: warm vulkan numbers drift
strongly with ambient WDDM load (identical binary: 5.13s and 9.7s
within one day; cuda stable at 4.21s). vulkan/cuda ratios are only
meaningful within a single back-to-back measurement window.

The forced staged coopmat epilogue writes bias without the matmul
contribution on some tiles when batch > 1 (RTX 3060, current driver;
batch=1 identical shapes are exact; the unaligned scalar variant is
exact within accumulation-order noise). Shape-matrix unit test added
(candle-nn `mul_mat_add_tests`); the tensor-level gate requires
batch == 1 on top of m/n/k alignment. Production multi-view linears
(batch = num_views, m = 8244) stay unfused until this is root-caused;
candidates: subgroup-scope barrier semantics after coopMatStore to
shared, or stage slot aliasing across cms_per_row/cms_per_col fragments.

## Back-to-back warm criterion, one window (2026-09-06, rev e69fe16d)

| bench | cuda | vulkan | wgpu |
|---|---:|---:|---:|
| pi3/iter | 2.82s | 3.59s (1.27x) | 54.9s (19.5x) |
| pi3x/iter | 4.34s | 9.36s (2.16x)* | 60.4s (13.9x) |

*The vulkan_pi3x window hit the slow side of the ambient-WDDM drift (the
same build measured 1.22x earlier; cuda is stable). wgpu improved from
76.3s/76.9s to 60.4s/54.9s after enabling coop64 for unaligned shapes.

## WGSL GEMM attribution (wgpu GPU-timestamp profiler, rev 3241f1a5)

pi3x GPU time by kernel (before the coop fix): matmul-warptile 515x
20783ms (95% of GPU, 40.4ms/ea ~ 0.43 TFLOPS); the coop64 gate required
m/n/k % 16 while pi3x linears are m=3903/1301. The kernel already
zero-fills ragged shared-tile loads and clips stores, so the host
%16 checks were removed: matmul-coop64 521x 2127ms (4.1ms/ea, 10x).
GPU total ~5.9s of the ~70s wall — the remaining wgpu gap is CPU-side
dispatch/pass overhead (~9.5k passes x ~6ms WDDM), the same structural
tax as vulkan; pass-folding for provably independent dispatches is the
next wgpu lever.

## Pass folding (rev 029c5acc) — implemented, NEUTRAL on dense models

encode_pending_dispatches folds consecutive deferred dispatches whose
storage-buffer sets are pairwise disjoint into one compute pass (uniform
ring buffers excluded from the dependency set). pi3x wgpu: 70.5s vs
69.9s, mesh compare PASS — the dense-model dispatch chain is almost
strictly dependent (consecutive ops share the activation tensor), so
the fold rate is near zero. Conclusion: the ~64s of wall over ~5.9s GPU
is inter-pass driver gap on a strictly dependent chain — removable only
by dispatch-count reduction (op fusion in the model graph), not by
pass-level folding. vulkan_pi3x drift (2.16x evening vs 1.22x morning
at identical builds): cuda stays 4.21s across windows while vulkan
swings 5.1-12.2s — ambient WDDM/driver state, not code; A/B on the
same build confirmed it.

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
