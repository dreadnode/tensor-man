use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

mod inspect;

pub(crate) use inspect::*;

use crate::core::FileType;

#[derive(Debug, Parser)]
#[clap(name = "tman", version)]
pub(crate) struct Arguments {
    #[clap(subcommand)]
    pub command: Command,
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum DetailLevel {
    /// Print metadata and high level information only.
    Brief,
    /// Print the metadata and detailed tensor information.
    Full,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Inspect a safetensors file.
    Inspect(InspectArgs),
}

#[derive(Debug, Args)]
pub(crate) struct InspectArgs {
    // Safetensors file to inspect.
    file_path: PathBuf,
    /// Override the file format detection by file extension.
    #[clap(long)]
    format: Option<FileType>,
    /// Detail level.
    #[clap(long, short = 'D', default_value = "brief")]
    detail: DetailLevel,
    /// If the detail level is set to full, filter the tensors by this substring.
    #[clap(long, short = 'F')]
    filter: Option<String>,
    /// Save as JSON to the specified file.
    #[clap(long, short = 'J')]
    to_json: Option<PathBuf>,
}
