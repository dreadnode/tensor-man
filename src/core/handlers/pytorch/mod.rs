use std::path::{Path, PathBuf};

use crate::{
    cli::DetailLevel,
    core::{docker, FileType, Inspection},
};

use super::{Handler, Scope};

pub(crate) struct PyTorchHandler;

impl PyTorchHandler {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl Handler for PyTorchHandler {
    fn file_type(&self) -> FileType {
        FileType::PyTorch
    }

    fn is_handler_for(&self, file_path: &Path, _scope: &Scope) -> bool {
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

    fn paths_to_sign(&self, file_path: &Path) -> anyhow::Result<Vec<PathBuf>> {
        // TODO: can a pytorch model reference external files?
        Ok(vec![file_path.to_path_buf()])
    }

    fn inspect(
        &self,
        file_path: &Path,
        detail: DetailLevel,
        filter: Option<String>,
    ) -> anyhow::Result<Inspection> {
        if !docker::docker_exists() {
            return Err(anyhow::anyhow!(
                "docker is required to inspect pytorch models, make sure the docker binary is in $PATH and that /var/run/docker.sock is shared from the host if you are running tensor-man itself inside a container."
            ));
        }

        docker::Inspector::new(
            include_str!("inspect.Dockerfile"),
            include_str!("inspect.py"),
            include_str!("inspect.requirements"),
        )
        .run(file_path, vec![], detail, filter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_pytorch() {
        // Standard .pt extension
        let handler = PyTorchHandler {};

        assert!(handler.is_handler_for(Path::new("model.pt"), &Scope::Inspection));
        assert!(handler.is_handler_for(Path::new("path/to/model.pt"), &Scope::Inspection));
        assert!(handler.is_handler_for(Path::new("MODEL.PT"), &Scope::Inspection)); // Case insensitive

        // Standard .pth extension
        assert!(handler.is_handler_for(Path::new("model.pth"), &Scope::Inspection));
        assert!(handler.is_handler_for(Path::new("path/to/model.pth"), &Scope::Inspection));
        assert!(handler.is_handler_for(Path::new("MODEL.PTH"), &Scope::Inspection)); // Case insensitive

        // Standard pytorch_model.bin filename
        assert!(handler.is_handler_for(Path::new("pytorch_model.bin"), &Scope::Inspection));
        assert!(handler.is_handler_for(Path::new("path/to/pytorch_model.bin"), &Scope::Inspection));
        assert!(handler.is_handler_for(Path::new("PYTORCH_MODEL.BIN"), &Scope::Inspection)); // Case insensitive

        // Variants of pytorch_model.*.bin
        assert!(
            handler.is_handler_for(Path::new("diffusion_pytorch_model.bin"), &Scope::Inspection)
        );
        assert!(handler.is_handler_for(
            Path::new("diffusion_pytorch_model.fp16.bin"),
            &Scope::Inspection
        ));
        assert!(handler.is_handler_for(
            Path::new("text_encoder_pytorch_model.safetensors.bin"),
            &Scope::Inspection
        ));

        // Non-matching cases
        assert!(!handler.is_handler_for(Path::new("model.onnx"), &Scope::Inspection));
        assert!(!handler.is_handler_for(Path::new("model.safetensors"), &Scope::Inspection));
        assert!(!handler.is_handler_for(Path::new("model.bin"), &Scope::Inspection)); // Just .bin isn't enough
        assert!(!handler.is_handler_for(Path::new("pytorch.txt"), &Scope::Inspection));
        assert!(!handler.is_handler_for(Path::new(""), &Scope::Inspection));
    }
}
