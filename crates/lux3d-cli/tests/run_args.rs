use clap::Parser;
use lux3d_cli::{Cli, Command};

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
