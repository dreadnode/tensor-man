use std::{
    collections::{BTreeMap, HashSet},
    path::PathBuf,
};

use rayon::prelude::*;

use safetensors::SafeTensors;

use crate::{cli::DetailLevel, core::TensorDescriptor};

use super::{FileType, Inspection};

pub(crate) fn inspect(
    file_path: PathBuf,
    detail: DetailLevel,
    filter: Option<String>,
) -> anyhow::Result<Inspection> {
    let mut inspection = Inspection::default();

    let file = std::fs::File::open(&file_path)?;
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

        let mut tensor_descriptors = Vec::new();

        for (tensor_id, tensor_info) in tensors {
            if let Some(filter) = &filter {
                if !tensor_id.contains(filter) {
                    continue;
                }
            }

            tensor_descriptors.push(TensorDescriptor {
                id: Some(tensor_id.to_string()),
                shape: tensor_info.shape.clone(),
                dtype: format!("{:?}", &tensor_info.dtype),
                size: tensor_info.data_offsets.1 - tensor_info.data_offsets.0,
                metadata: super::Metadata::new(),
            });
        }

        inspection.tensors = Some(tensor_descriptors);
    }

    Ok(inspection)
}
