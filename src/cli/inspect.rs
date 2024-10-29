use crate::core::handlers::Scope;

use super::InspectArgs;

pub(crate) fn inspect(args: InspectArgs) -> anyhow::Result<()> {
    let handler =
        crate::core::handlers::handler_for(args.format, &args.file_path, Scope::Inspection)?;

    if !args.quiet {
        println!(
            "Inspecting {:?} (format={}, detail={:?}{}):\n",
            args.file_path,
            handler.file_type(),
            args.detail,
            args.filter
                .as_ref()
                .map(|f| format!(" filter_by={:?}", f))
                .unwrap_or("".to_string())
        );
    }

    let inspection = handler.inspect(&args.file_path, args.detail, args.filter)?;

    if !args.quiet {
        println!("file type:     {}", inspection.file_type);
        println!("version:       {}", inspection.version);
        println!(
            "file size:     {} ({})",
            humansize::format_size(inspection.file_size, humansize::DECIMAL),
            inspection.file_size
        );
        if inspection.header_size > 0 {
            println!(
                "header size:   {} ({})",
                humansize::format_size(inspection.header_size, humansize::DECIMAL),
                inspection.header_size
            );
        }
        println!("total tensors: {}", inspection.num_tensors);
        println!(
            "data size:     {} ({})",
            humansize::format_size(inspection.data_size, humansize::DECIMAL),
            inspection.data_size
        );
        println!(
            "average size:  {}",
            humansize::format_size(inspection.average_tensor_size(), humansize::DECIMAL)
        );

        println!("data types:    {}", inspection.unique_dtypes.join(", "));
        println!(
            "shapes:        {}",
            inspection
                .unique_shapes
                .iter()
                .map(|s| format!("{:?}", s))
                .collect::<Vec<_>>()
                .join(", ")
        );

        if !inspection.metadata.is_empty() {
            println!("\nmetadata:\n");
            for (meta_key, meta_value) in &inspection.metadata {
                println!("  {}: {}", meta_key, meta_value);
            }
        }

        if let Some(tensors) = &inspection.tensors {
            println!("\ntensors:\n");

            for tensor_info in tensors {
                println!(
                    "  {}",
                    tensor_info
                        .id
                        .as_ref()
                        .unwrap_or(&"<no tensor id>".to_string())
                );
                println!("    dtype: {:?}", tensor_info.dtype);
                println!("    shape: {:?}", tensor_info.shape);
                println!(
                    "    size: {} ({})",
                    humansize::format_size(tensor_info.size, humansize::DECIMAL),
                    tensor_info.size
                );

                if !tensor_info.metadata.is_empty() {
                    println!("    metadata:");
                    for (meta_key, meta_value) in &tensor_info.metadata {
                        println!("      {}: {}", meta_key, meta_value);
                    }
                }

                println!();
            }
        }
    }

    if let Some(json_file_path) = &args.to_json {
        let json_str = serde_json::to_string_pretty(&inspection)?;
        std::fs::write(json_file_path, json_str)?;

        if !args.quiet {
            println!("\nsaved to {:?}", json_file_path);
        }
    }

    Ok(())
}
