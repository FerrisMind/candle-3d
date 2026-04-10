use clap::Parser;
use lux3d_cli::{Cli, Command, WeightsCommand, inspect_model, normalize_weights, run_model};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect(args) => {
            let rendered = inspect_model(args.repo_root, args.family.model_family())?;
            println!("{rendered}");
        }
        Command::Weights(args) => match args.command {
            WeightsCommand::Normalize(family) => {
                let rendered = normalize_weights(
                    family.repo_root,
                    family.family.model_family(),
                    family.raw_model_dir,
                    family.output_dir,
                )?;
                println!("{rendered}");
            }
        },
        Command::Run(args) => println!("{}", run_model(args)?.display()),
    }

    Ok(())
}
