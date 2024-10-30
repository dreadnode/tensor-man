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
        Command::Version => {
            println!("{} v{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
            Ok(())
        }
    };

    if let Err(e) = ret {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
