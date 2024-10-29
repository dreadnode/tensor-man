use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::{Path, PathBuf},
};

use rayon::prelude::*;

use safetensors::{tensor::TensorInfo, SafeTensors};
use serde::Deserialize;

use crate::{
    cli::DetailLevel,
    core::{FileType, Inspection, Metadata, TensorDescriptor},
};

use super::{Handler, Scope};

#[derive(Debug, Deserialize)]
struct TensorIndex {
    weight_map: HashMap<String, String>,
}

pub(crate) struct SafeTensorsHandler;

impl SafeTensorsHandler {
    pub(crate) fn new() -> Self {
        Self
    }
}

fn is_safetensors_index(file_path: &Path) -> bool {
    file_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .ends_with(".safetensors.index.json")
}

fn build_tensor_descriptor(tensor_id: &str, tensor_info: &TensorInfo) -> TensorDescriptor {
    TensorDescriptor {
        id: Some(tensor_id.to_string()),
        shape: tensor_info.shape.clone(),
        dtype: format!("{:?}", &tensor_info.dtype),
        size: tensor_info.data_offsets.1 - tensor_info.data_offsets.0,
        metadata: Metadata::new(),
    }
}

impl Handler for SafeTensorsHandler {
    fn file_type(&self) -> FileType {
        FileType::SafeTensors
    }

    fn is_handler_for(&self, file_path: &Path, scope: &Scope) -> bool {
        let is_safetensors = file_path
            .extension()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("")
            .to_ascii_lowercase()
            == "safetensors";

        match scope {
            // can only inspect safetensors files
            Scope::Inspection => is_safetensors,
            // can sign safetensors files directly or an index referencing multiple files
            Scope::Signing => is_safetensors || is_safetensors_index(file_path),
        }
    }

    fn paths_to_sign(&self, file_path: &Path) -> anyhow::Result<Vec<PathBuf>> {
        if is_safetensors_index(file_path) {
            // load unique paths from index
            let base_path = file_path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("no parent path"))?;

            let index = std::fs::read_to_string(file_path)?;
            let index: TensorIndex = serde_json::from_str(&index)?;

            let unique: HashSet<PathBuf> = index
                .weight_map
                .values()
                .map(PathBuf::from)
                .map(|p| {
                    if p.is_relative() {
                        base_path.join(p)
                    } else {
                        p
                    }
                })
                .collect();

            let mut paths = vec![file_path.to_path_buf()];
            paths.extend(unique);
            Ok(paths)
        } else {
            // safetensors are self contained
            Ok(vec![file_path.to_path_buf()])
        }
    }

    fn inspect(
        &self,
        file_path: &Path,
        detail: DetailLevel,
        filter: Option<String>,
    ) -> anyhow::Result<Inspection> {
        let mut inspection = Inspection::default();

        let file = std::fs::File::open(file_path)?;
        let buffer = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .unwrap_or_else(|_| panic!("failed to map file {}", file_path.display()))
        };

        inspection.file_path = file_path.canonicalize()?;
        inspection.file_size = file.metadata()?.len();

        // read header
        let (header_size, header) = SafeTensors::read_metadata(&buffer)?;

        inspection.file_type = FileType::SafeTensors;
        inspection.header_size = header_size;
        inspection.version = "0.x".to_string();

        let tensors = header.tensors();

        // transform tensors to a vector
        let mut tensors: Vec<_> = tensors.into_iter().collect();

        inspection.num_tensors = tensors.len();
        inspection.data_size = tensors
            .par_iter()
            .map(|t| t.1.data_offsets.1 - t.1.data_offsets.0)
            .sum::<usize>();

        inspection.unique_shapes = tensors
            .par_iter()
            .map(|t| t.1.shape.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        // sort shapes by volume
        inspection.unique_shapes.sort_by(|a, b| {
            let size_a: usize = a.iter().product();
            let size_b: usize = b.iter().product();
            size_a.cmp(&size_b)
        });

        inspection.unique_dtypes = tensors
            .par_iter()
            .map(|t| format!("{:?}", t.1.dtype))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        if let Some(block_metadata) = header.metadata() {
            inspection.metadata = BTreeMap::from_iter(
                block_metadata
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string())),
            );
        }

        if matches!(detail, DetailLevel::Full) {
            // sort by offset
            tensors.sort_by_key(|(_, info)| info.data_offsets.0);

            inspection.tensors = Some(
                tensors
                    .par_iter()
                    .filter(|(tensor_id, _)| {
                        filter.as_ref().map_or(true, |f| tensor_id.contains(f))
                    })
                    .map(|(tensor_id, tensor_info)| build_tensor_descriptor(tensor_id, tensor_info))
                    .collect(),
            );
        }

        Ok(inspection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_handler_for() {
        let handler = SafeTensorsHandler::new();

        // Test inspection scope
        assert!(handler.is_handler_for(Path::new("model.safetensors"), &Scope::Inspection));
        assert!(handler.is_handler_for(Path::new("path/to/model.safetensors"), &Scope::Inspection));
        assert!(handler.is_handler_for(Path::new("MODEL.SAFETENSORS"), &Scope::Inspection)); // Case insensitive

        // Test signing scope
        assert!(handler.is_handler_for(Path::new("model.safetensors"), &Scope::Signing));
        assert!(handler.is_handler_for(Path::new("path/to/model.safetensors"), &Scope::Signing));
        assert!(handler.is_handler_for(Path::new("MODEL.SAFETENSORS"), &Scope::Signing)); // Case insensitive

        // Test non-matching cases for both scopes
        for scope in [Scope::Inspection, Scope::Signing] {
            assert!(!handler.is_handler_for(Path::new("model.onnx"), &scope));
            assert!(!handler.is_handler_for(Path::new("model.pt"), &scope));
            assert!(!handler.is_handler_for(Path::new("model.bin"), &scope));
            assert!(!handler.is_handler_for(Path::new("safetensors.txt"), &scope));
            assert!(!handler.is_handler_for(Path::new(""), &scope));
        }
    }

    #[test]
    fn test_is_handler_for_index() {
        let handler = SafeTensorsHandler::new();

        // Index files should only be handled in signing scope
        assert!(handler.is_handler_for(Path::new("model.safetensors.index.json"), &Scope::Signing));
        assert!(handler.is_handler_for(
            Path::new("path/to/model.safetensors.index.json"),
            &Scope::Signing
        ));

        // Index files should not be handled in inspection scope
        assert!(!handler.is_handler_for(
            Path::new("model.safetensors.index.json"),
            &Scope::Inspection
        ));
        assert!(!handler.is_handler_for(
            Path::new("path/to/model.safetensors.index.json"),
            &Scope::Inspection
        ));
    }
}
