use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

mod graph;
mod inspect;
mod signing;

pub(crate) use graph::*;
pub(crate) use inspect::*;
pub(crate) use signing::*;

use crate::core::FileType;

#[derive(Debug, Parser)]
#[clap(name = "tensor-man", version, about)]
pub(crate) struct Arguments {
    #[clap(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Inspect a file in one of the supported formats.
    Inspect(InspectArgs),
    /// Create a new key pair for signging and save it to a file.
    CreateKey(CreateKeyArgs),
    /// Sign the model with the provided key and generate a signature file.
    Sign(SignArgs),
    /// Verify model signature.
    Verify(VerifyArgs),
    /// Generate a DOT representation of the graph of the model.
    Graph(GraphArgs),
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum DetailLevel {
    /// Print metadata and high level information only.
    Brief,
    /// Print the metadata and detailed tensor information.
    Full,
}

#[derive(Debug, Args)]
pub(crate) struct InspectArgs {
    // File to inspect.
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

#[derive(Debug, Args)]
pub(crate) struct CreateKeyArgs {
    /// Output path for private key file.
    #[clap(long, default_value = "./private.key")]
    private_key: PathBuf,
    /// Output path for public key file.
    #[clap(long, default_value = "./public.key")]
    public_key: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct SignArgs {
    // File to sign.
    file_path: PathBuf,
    /// Override the file format detection by file extension.
    #[clap(long)]
    format: Option<FileType>,
    // Private key file.
    #[clap(long, short = 'K')]
    key_path: PathBuf,
    /// Output signature file. If not set the original file name will be used as base name.
    #[clap(long, short = 'O')]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct VerifyArgs {
    // File to verify.
    file_path: PathBuf,
    /// Override the file format detection by file extension.
    #[clap(long)]
    format: Option<FileType>,
    /// Public key file.
    #[clap(long, short = 'K')]
    key_path: PathBuf,
    /// Signature file. If not set the file name will be used as base name.
    #[clap(long, short = 'S')]
    signature: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct GraphArgs {
    // File to inspect.
    file_path: PathBuf,
    /// Output DOT file.
    #[clap(long, short = 'O', default_value = "graph.dot")]
    output: PathBuf,
    /// Override the file format detection by file extension.
    #[clap(long)]
    format: Option<FileType>,
}
