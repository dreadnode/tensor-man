use std::path::{Path, PathBuf};

use crate::cli::DetailLevel;

use super::{FileType, Inspection};

pub(crate) mod gguf;
pub(crate) mod onnx;
pub(crate) mod pytorch;
pub(crate) mod safetensors;

pub(crate) enum Scope {
    Inspection,
    Signing,
}

pub(crate) trait Handler {
    fn file_type(&self) -> FileType;

    fn is_handler_for(&self, file_path: &Path, scope: &Scope) -> bool;
    fn paths_to_sign(&self, file_path: &Path) -> anyhow::Result<Vec<PathBuf>>;
    fn inspect(
        &self,
        file_path: &Path,
        detail: DetailLevel,
        filter: Option<String>,
    ) -> anyhow::Result<Inspection>;

    fn create_graph(&self, _file_path: &Path, _output_path: &Path) -> anyhow::Result<()> {
        Err(anyhow::anyhow!(
            "graph generation not supported for this format"
        ))
    }
}

pub(crate) fn handler_for(
    format: Option<FileType>,
    file_path: &Path,
    scope: Scope,
) -> anyhow::Result<Box<dyn Handler>> {
    let safetensors_handler = safetensors::SafeTensorsHandler::new();
    let onnx_handler = onnx::OnnxHandler::new();
    let gguf_handler = gguf::GGUFHandler::new();
    let pytorch_handler = pytorch::PyTorchHandler::new();

    match &format {
        None => {
            if safetensors_handler.is_handler_for(file_path, &scope) {
                Ok(Box::new(safetensors_handler))
            } else if onnx_handler.is_handler_for(file_path, &scope) {
                Ok(Box::new(onnx_handler))
            } else if gguf_handler.is_handler_for(file_path, &scope) {
                Ok(Box::new(gguf_handler))
            } else if pytorch_handler.is_handler_for(file_path, &scope) {
                Ok(Box::new(pytorch_handler))
            } else {
                anyhow::bail!("unsupported file format")
            }
        }
        Some(forced_format) => {
            if forced_format.is_safetensors() {
                Ok(Box::new(safetensors_handler))
            } else if forced_format.is_onnx() {
                Ok(Box::new(onnx_handler))
            } else if forced_format.is_gguf() {
                Ok(Box::new(gguf_handler))
            } else if forced_format.is_pytorch() {
                Ok(Box::new(pytorch_handler))
            } else {
                anyhow::bail!("unsupported file format")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_handler_for_with_forced_format() {
        // Test each forced format
        let path = Path::new("test.bin");
        let handler = handler_for(Some(FileType::SafeTensors), path, Scope::Inspection).unwrap();
        assert!(matches!(handler.file_type(), FileType::SafeTensors));

        let handler = handler_for(Some(FileType::ONNX), path, Scope::Inspection).unwrap();
        assert!(matches!(handler.file_type(), FileType::ONNX));

        let handler = handler_for(Some(FileType::GGUF), path, Scope::Inspection).unwrap();
        assert!(matches!(handler.file_type(), FileType::GGUF));

        let handler = handler_for(Some(FileType::PyTorch), path, Scope::Inspection).unwrap();
        assert!(matches!(handler.file_type(), FileType::PyTorch));
    }

    #[test]
    fn test_handler_for_with_file_extension() {
        // Test auto-detection by file extension
        let handler = handler_for(None, Path::new("model.safetensors"), Scope::Inspection).unwrap();
        assert!(matches!(handler.file_type(), FileType::SafeTensors));

        let handler = handler_for(None, Path::new("model.onnx"), Scope::Inspection).unwrap();
        assert!(matches!(handler.file_type(), FileType::ONNX));

        let handler = handler_for(None, Path::new("model.gguf"), Scope::Inspection).unwrap();
        assert!(matches!(handler.file_type(), FileType::GGUF));

        let handler = handler_for(None, Path::new("model.pt"), Scope::Inspection).unwrap();
        assert!(matches!(handler.file_type(), FileType::PyTorch));
    }

    #[test]
    fn test_handler_for_unknown_format() {
        // Test handling of unknown format
        let result = handler_for(None, Path::new("model.unknown"), Scope::Inspection);
        assert!(result.is_err());
    }

    #[test]
    fn test_handler_for_different_scopes() {
        // Test that handlers work with different scopes
        let handler = handler_for(
            Some(FileType::SafeTensors),
            Path::new("test.bin"),
            Scope::Signing,
        )
        .unwrap();
        assert!(matches!(handler.file_type(), FileType::SafeTensors));

        let handler = handler_for(
            Some(FileType::SafeTensors),
            Path::new("test.bin"),
            Scope::Inspection,
        )
        .unwrap();
        assert!(matches!(handler.file_type(), FileType::SafeTensors));
    }

    #[test]
    fn test_format_override() {
        // Test that forced format overrides file extension
        let handler = handler_for(
            Some(FileType::ONNX),
            Path::new("model.safetensors"),
            Scope::Inspection,
        )
        .unwrap();
        assert!(matches!(handler.file_type(), FileType::ONNX));

        let handler = handler_for(
            Some(FileType::GGUF),
            Path::new("model.pt"),
            Scope::Inspection,
        )
        .unwrap();
        assert!(matches!(handler.file_type(), FileType::GGUF));
    }
}
