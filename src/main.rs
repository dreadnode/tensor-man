use clap::Parser;
use cli::{Args, Command};

mod cli;
mod core;

fn main() {
    let args = Args::parse();

    let ret = match args.command {
        Command::Inspect {
            file_path,
            detail,
            format,
            filter,
            to_json,
        } => cli::inspect(file_path, detail, format, filter, to_json),
    };

    if let Err(e) = ret {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}
