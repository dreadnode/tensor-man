use clap::Parser;
use cli::{Arguments, Command};

mod cli;
mod core;

fn main() {
    let args = Arguments::parse();

    let ret = match args.command {
        Command::Inspect(args) => cli::inspect(args),
        Command::CreateKey(args) => cli::create_key(args),
        Command::Sign(args) => cli::sign(args),
        Command::Verify(args) => cli::verify(args),
        Command::Graph(args) => cli::graph(args),
    };

    if let Err(e) = ret {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}
