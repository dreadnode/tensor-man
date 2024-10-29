use crate::core::handlers::Scope;

use super::GraphArgs;

pub(crate) fn graph(args: GraphArgs) -> anyhow::Result<()> {
    println!(
        "Generating DOT graph for {} to {} ...",
        args.file_path.display(),
        args.output.display()
    );

    crate::core::handlers::handler_for(args.format, &args.file_path, Scope::Inspection)?
        .create_graph(&args.file_path, &args.output)
}
