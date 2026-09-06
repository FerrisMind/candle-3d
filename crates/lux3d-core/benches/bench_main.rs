mod benchmarks;

// Custom main instead of criterion_main! so the GPU GEMM opt-ins are enabled
// before the device is created — mirrors lux3d-cli.
fn main() {
    lux3d_core::enable_reduced_precision_gemm_opt_ins();
    benchmarks::models::benches();
}
