use crate::core::FileType;

use super::GraphArgs;

pub(crate) fn graph(args: GraphArgs) -> anyhow::Result<()> {
    println!(
        "Generating DOT graph for {} to {} ...",
        args.file_path.display(),
        args.output.display()
    );

    let forced_format = args.format.unwrap_or(FileType::Unknown);
    let file_ext = args
        .file_path
        .extension()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("")
        .to_ascii_lowercase();

    if !forced_format.is_onnx() && file_ext != "onnx" {
        anyhow::bail!("this format does not embed graph information");
    }

    crate::core::onnx::create_graph(args.file_path, args.output)
}
