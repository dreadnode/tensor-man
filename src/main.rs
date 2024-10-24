use clap::Parser;
use cli::{Args, Command};

mod cli;
mod core;

fn main() {
    let args = Args::parse();

    match args.command {
        Command::Inspect {
            file_path,
            detail,
            filter,
            to_json,
        } => cli::inspect(file_path, detail, filter, to_json).unwrap(),
    }
}
