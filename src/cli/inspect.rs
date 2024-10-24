use std::path::PathBuf;

use crate::core::FileType;

use super::DetailLevel;

pub(crate) fn inspect(
    file_path: PathBuf,
    detail: DetailLevel,
    format: Option<FileType>,
    filter: Option<String>,
    to_json: Option<PathBuf>,
) -> anyhow::Result<()> {
    println!(
        "Inspecting {:?} (detail={:?}{}):\n",
        file_path,
        detail,
        filter
            .as_ref()
            .map(|f| format!(" filter_by={:?}", f))
            .unwrap_or("".to_string())
    );

    let forced_format = format.unwrap_or(FileType::Unknown);
    let file_ext = file_path
        .extension()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("")
        .to_ascii_lowercase();

    // TODO: check if file_path is a safetensors index (or model path) and iterate if so

    let inspection = if forced_format.is_safetensors() || file_ext == "safetensors" {
        crate::core::safetensors::inspect(file_path, detail, filter)?
    } else if forced_format.is_onnx() || file_ext == "onnx" {
        crate::core::onnx::inspect(file_path, detail, filter)?
    } else {
        anyhow::bail!("unsupported file extension: {:?}", file_ext)
    };

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

    if let Some(json_file_path) = &to_json {
        let json_str = serde_json::to_string_pretty(&inspection)?;
        std::fs::write(json_file_path, json_str)?;

        println!("\nsaved to {:?}", json_file_path);
    }

    Ok(())
}
