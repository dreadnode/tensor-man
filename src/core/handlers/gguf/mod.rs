use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use gguf::{GGMLType, GGUFTensorInfo};
use rayon::prelude::*;

use super::{Handler, Scope};
use crate::{
    cli::DetailLevel,
    core::{FileType, Inspection, Metadata, TensorDescriptor},
};

#[inline]
fn data_type_bits(dtype: GGMLType) -> usize {
    match dtype {
        GGMLType::F32 => 32,
        GGMLType::F16 => 16,
        GGMLType::Q4_0 => 4,
        GGMLType::Q4_1 => 4,
        GGMLType::Q5_0 => 5,
        GGMLType::Q5_1 => 5,
        GGMLType::Q8_0 => 8,
        GGMLType::Q8_1 => 8,
        GGMLType::Q2K => 2,
        GGMLType::Q3K => 3,
        GGMLType::Q4K => 4,
        GGMLType::Q5K => 5,
        GGMLType::Q6K => 6,
        GGMLType::Q8K => 8,
        GGMLType::I8 => 8,
        GGMLType::I16 => 16,
        GGMLType::I32 => 32,
        GGMLType::Count => 32, // Assuming Count is 32-bit, adjust if needed
    }
}

fn build_tensor_descriptor(t_info: &GGUFTensorInfo) -> TensorDescriptor {
    TensorDescriptor {
        id: Some(t_info.name.to_string()),
        shape: t_info.dimensions.iter().map(|d| *d as usize).collect(),
        dtype: format!("{:?}", t_info.tensor_type),
        size: if t_info.dimensions.is_empty() {
            0
        } else {
            (data_type_bits(t_info.tensor_type)
                * t_info
                    .dimensions
                    .iter()
                    .map(|d| *d as usize)
                    .product::<usize>())
                / 8
        },
        metadata: Metadata::new(),
    }
}

fn format_parsing_error(error: &str) -> String {
    // the GGUF library dumps the entire buffer in the error message, we don't want that.
    if error.len() > 100 {
        format!("failed to parse GGUF file: {}", &error[..100])
    } else {
        format!("failed to parse GGUF file: {}", error)
    }
}

pub(crate) struct GGUFHandler {}

impl GGUFHandler {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Handler for GGUFHandler {
    fn file_type(&self) -> FileType {
        FileType::GGUF
    }

    fn is_handler_for(&self, file_path: &Path, _scope: &Scope) -> bool {
        file_path
            .extension()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("")
            .to_ascii_lowercase()
            == "gguf"
    }

    fn paths_to_sign(&self, file_path: &Path) -> anyhow::Result<Vec<PathBuf>> {
        // GGUF are self contained
        Ok(vec![file_path.to_path_buf()])
    }

    fn inspect(
        &self,
        file_path: &Path,
        detail: crate::cli::DetailLevel,
        filter: Option<String>,
    ) -> anyhow::Result<crate::core::Inspection> {
        let mut inspection = Inspection::default();

        let file = std::fs::File::open(file_path)?;
        let buffer = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .unwrap_or_else(|_| panic!("failed to map file {}", file_path.display()))
        };

        inspection.file_path = file_path.canonicalize()?;
        inspection.file_size = file.metadata()?.len();

        let gguf = gguf::GGUFFile::read(&buffer)
            .map_err(|e| anyhow::anyhow!(format_parsing_error(&e.to_string())))?
            .unwrap_or_else(|| panic!("failed to read GGUF file {}", file_path.display()));

        inspection.file_type = FileType::GGUF;
        inspection.version = format!("{}", gguf.header.version);
        inspection.num_tensors = gguf.header.tensor_count as usize;
        inspection.unique_shapes = gguf
            .tensors
            .par_iter()
            .map(|t| t.dimensions.iter().map(|d| *d as usize).collect::<Vec<_>>())
            .filter(|shape| !shape.is_empty())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        // sort shapes by volume
        inspection.unique_shapes.sort_by(|a, b| {
            let size_a: usize = a.iter().product();
            let size_b: usize = b.iter().product();
            size_a.cmp(&size_b)
        });

        inspection.unique_dtypes = gguf
            .tensors
            .par_iter()
            .map(|t| format!("{:?}", t.tensor_type))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        inspection.data_size = gguf
            .tensors
            .par_iter()
            .map(|t| {
                if t.dimensions.is_empty() {
                    0
                } else {
                    data_type_bits(t.tensor_type)
                        * t.dimensions.iter().map(|d| *d as usize).product::<usize>()
                }
            })
            .sum::<usize>()
            / 8;

        for meta in &gguf.header.metadata {
            inspection
                .metadata
                .insert(meta.key.clone(), format!("{:?}", meta.value));
        }

        if matches!(detail, DetailLevel::Full) {
            inspection.tensors = Some(
                gguf.tensors
                    .par_iter()
                    .filter(|t_info| filter.as_ref().map_or(true, |f| t_info.name.contains(f)))
                    .map(build_tensor_descriptor)
                    .collect(),
            );
        }

        Ok(inspection)
    }
}
