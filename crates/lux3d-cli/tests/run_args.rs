use std::path::PathBuf;

use clap::Parser;
use lux3d_cli::{Cli, Command};

#[test]
fn run_command_accepts_model_path_and_cache_dir_without_repo_root() {
    let cli = Cli::parse_from([
        "lux3d",
        "run",
        "pi3",
        "--source",
        "input.mp4",
        "--model-path",
        "C:\\models\\pi3",
        "--cache-dir",
        "D:\\luxrt-cache",
        "--output",
        "out.ply",
    ]);

    let Command::Run(args) = cli.command else {
        panic!("expected run command");
    };

    assert_eq!(Some(PathBuf::from(r"C:\models\pi3")), args.model_path);
    assert_eq!(Some(PathBuf::from(r"D:\luxrt-cache")), args.cache_dir);
}
