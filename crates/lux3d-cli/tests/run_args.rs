use clap::Parser;
use lux3d_cli::{Cli, Command, run_model};
use lux3d_core::ModelFamily;

#[test]
fn run_command_accepts_model_path_and_cache_dir_without_repo_root() {
    let model_path = std::env::temp_dir().join("lux3d-models").join("pi3");
    let cache_dir = std::env::temp_dir().join("luxrt-cache");

    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "pi3".to_string(),
        "--source".to_string(),
        "input.mp4".to_string(),
        "--model-path".to_string(),
        model_path.display().to_string(),
        "--cache-dir".to_string(),
        cache_dir.display().to_string(),
        "--output".to_string(),
        "out.ply".to_string(),
    ]);

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };

    assert_eq!(Some(model_path), args.model_path);
    assert_eq!(Some(cache_dir), args.cache_dir);
}

#[test]
fn run_triposg_parses_family_and_option_overrides() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "triposg".to_string(),
        "--source".to_string(),
        "image.png".to_string(),
        "--steps".to_string(),
        "12".to_string(),
        "--guidance-scale".to_string(),
        "3.5".to_string(),
        "--seed".to_string(),
        "7".to_string(),
        "--num-tokens".to_string(),
        "1024".to_string(),
        "--dense-octree-depth".to_string(),
        "5".to_string(),
        "--hierarchical-octree-depth".to_string(),
        "6".to_string(),
        "--flash-decoder".to_string(),
        "false".to_string(),
        "--flash-octree-depth".to_string(),
        "7".to_string(),
        "--chunk-size".to_string(),
        "4096".to_string(),
        "--output".to_string(),
        "out.obj".to_string(),
    ]);

    assert_eq!("triposg", cli.command.family_name());
    assert_eq!(Some(ModelFamily::TripoSg), cli.command.family());

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(Some(12), args.steps);
    assert_eq!(Some(3.5), args.guidance_scale);
    assert_eq!(Some(7), args.seed);
    assert_eq!(Some(1024), args.num_tokens);
    assert_eq!(Some(5), args.dense_octree_depth);
    assert_eq!(Some(6), args.hierarchical_octree_depth);
    assert_eq!(Some(false), args.flash_decoder);
    assert_eq!(Some(7), args.flash_octree_depth);
    assert_eq!(Some(4096), args.chunk_size);
}

#[test]
fn run_triposg_flags_default_to_none() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "triposg".to_string(),
        "--source".to_string(),
        "image.png".to_string(),
        "--output".to_string(),
        "out.obj".to_string(),
    ]);

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };
    assert_eq!(None, args.steps);
    assert_eq!(None, args.guidance_scale);
    assert_eq!(None, args.seed);
    assert_eq!(None, args.num_tokens);
    assert_eq!(None, args.dense_octree_depth);
    assert_eq!(None, args.hierarchical_octree_depth);
    assert_eq!(None, args.flash_decoder);
    assert_eq!(None, args.flash_octree_depth);
}

/// Dispatch-level regression (T5 review minor 3): `run_model` must reject a
/// TripoSG-only flag for another family BEFORE loading any weights, so a
/// plain `--device cpu` run fails fast with the rejection message and never
/// touches the source file or the canonical package.
#[test]
fn run_model_rejects_triposg_only_flag_at_dispatch_before_weights() {
    let cli = Cli::parse_from(vec![
        "lux3d".to_string(),
        "run".to_string(),
        "pi3".to_string(),
        "--device".to_string(),
        "cpu".to_string(),
        "--source".to_string(),
        "does-not-exist.mp4".to_string(),
        "--steps".to_string(),
        "5".to_string(),
        "--output".to_string(),
        "out.ply".to_string(),
    ]);

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };

    let error = run_model(args).expect_err("`--steps` must be rejected for pi3 at dispatch");
    let message = error.to_string();
    assert!(
        message.contains("`--steps` is not supported for `pi3`"),
        "unexpected rejection message: {message}"
    );
}
