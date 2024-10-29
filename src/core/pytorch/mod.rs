use std::path::{Path, PathBuf};

use crate::cli::DetailLevel;

use super::{docker, Inspection};

pub(crate) fn is_pytorch(file_path: &Path) -> bool {
    let file_ext = file_path
        .extension()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("")
        .to_ascii_lowercase();

    let file_name = file_path
        .file_name()
        .unwrap_or_default()
        .to_str()
        .unwrap_or_default()
        .to_ascii_lowercase();

    file_ext == "pt"
    || file_ext == "pth"
        || file_name.ends_with("pytorch_model.bin")
        // cases like diffusion_pytorch_model.fp16.bin
        || (file_name.contains("pytorch_model") && file_name.ends_with(".bin"))
}

pub(crate) fn paths_to_sign(file_path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    // TODO: can a pytorch model reference external files?
    Ok(vec![file_path.to_path_buf()])
}

pub(crate) fn inspect(
    file_path: PathBuf,
    detail: DetailLevel,
    filter: Option<String>,
) -> anyhow::Result<Inspection> {
    docker::Inspector::new(
        include_str!("inspect.Dockerfile"),
        include_str!("inspect.py"),
        include_str!("inspect.requirements"),
    )
    .run(file_path, vec![], detail, filter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_pytorch() {
        // Standard .pt extension
        assert!(is_pytorch(Path::new("model.pt")));
        assert!(is_pytorch(Path::new("path/to/model.pt")));
        assert!(is_pytorch(Path::new("MODEL.PT"))); // Case insensitive

        // Standard .pth extension
        assert!(is_pytorch(Path::new("model.pth")));
        assert!(is_pytorch(Path::new("path/to/model.pth")));
        assert!(is_pytorch(Path::new("MODEL.PTH"))); // Case insensitive

        // Standard pytorch_model.bin filename
        assert!(is_pytorch(Path::new("pytorch_model.bin")));
        assert!(is_pytorch(Path::new("path/to/pytorch_model.bin")));
        assert!(is_pytorch(Path::new("PYTORCH_MODEL.BIN"))); // Case insensitive

        // Variants of pytorch_model.*.bin
        assert!(is_pytorch(Path::new("diffusion_pytorch_model.bin")));
        assert!(is_pytorch(Path::new("diffusion_pytorch_model.fp16.bin")));
        assert!(is_pytorch(Path::new(
            "text_encoder_pytorch_model.safetensors.bin"
        )));

        // Non-matching cases
        assert!(!is_pytorch(Path::new("model.onnx")));
        assert!(!is_pytorch(Path::new("model.safetensors")));
        assert!(!is_pytorch(Path::new("model.bin"))); // Just .bin isn't enough
        assert!(!is_pytorch(Path::new("pytorch.txt")));
        assert!(!is_pytorch(Path::new("")));
    }
}
