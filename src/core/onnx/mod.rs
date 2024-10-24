use std::{collections::HashSet, path::PathBuf};

use protobuf::Message;
use protos::{tensor_proto::DataLocation, ModelProto};
use rayon::prelude::*;

use crate::cli::DetailLevel;

use super::{FileType, Inspection, TensorDescriptor};

mod protos;

#[inline]
fn data_type_bytes(dtype: i32) -> usize {
    match dtype {
        1 => 4,   // float
        2 => 1,   // uint8_t
        3 => 1,   // int8_t
        4 => 2,   // uint16_t
        5 => 2,   // int16_t
        6 => 4,   // int32_t
        7 => 8,   // int64_t
        8 => 1,   // string
        9 => 1,   // bool
        10 => 2,  // FLOAT16
        11 => 8,  // DOUBLE
        12 => 4,  // UINT32
        13 => 8,  // UINT64
        14 => 8,  // COMPLEX64
        15 => 16, // COMPLEX128
        16 => 2,  // BFLOAT16
        17 => 1,  // FLOAT8E4M3FN
        18 => 1,  // FLOAT8E4M3FNUZ
        19 => 1,  // FLOAT8E5M2
        20 => 1,  // FLOAT8E5M2FNUZ
        21 => 1,  // UINT4 (4-bit values are packed, returning 1 as a placeholder)
        22 => 1,  // INT4 (4-bit values are packed, returning 1 as a placeholder)
        23 => 1,  // FLOAT4E2M1 (4-bit values are packed, returning 1 as a placeholder)
        _ => panic!("Unsupported data type: {}", dtype),
    }
}

#[inline]
pub(crate) fn data_type_string(dtype: i32) -> &'static str {
    match dtype {
        1 => "FLOAT",
        2 => "UINT8",
        3 => "INT8",
        4 => "UINT16",
        5 => "INT16",
        6 => "INT32",
        7 => "INT64",
        8 => "STRING",
        9 => "BOOL",
        10 => "FLOAT16",
        11 => "DOUBLE",
        12 => "UINT32",
        13 => "UINT64",
        14 => "COMPLEX64",
        15 => "COMPLEX128",
        16 => "BFLOAT16",
        17 => "FLOAT8E4M3FN",
        18 => "FLOAT8E4M3FNUZ",
        19 => "FLOAT8E5M2",
        20 => "FLOAT8E5M2FNUZ",
        21 => "UINT4",
        22 => "INT4",
        23 => "FLOAT4E2M1",
        _ => "UNKNOWN",
    }
}

pub(crate) fn inspect(
    file_path: PathBuf,
    detail: DetailLevel,
    filter: Option<String>,
) -> anyhow::Result<Inspection> {
    let mut inspection = Inspection::default();

    let mut file = std::fs::File::open(&file_path)?;

    inspection.file_path = file_path.canonicalize()?;
    inspection.file_size = file.metadata()?.len();

    let onnx_model: ModelProto = Message::parse_from_reader(&mut file)?;

    inspection.file_type = FileType::ONNX;

    if onnx_model.model_version != 0 {
        inspection.version = format!(
            "{} (IR v{})",
            onnx_model.model_version, onnx_model.ir_version
        );
    } else {
        inspection.version = format!("IR v{}", onnx_model.ir_version);
    }

    // TODO: check the presence of sparse tensors from graph.sparse_initializer

    inspection.num_tensors = onnx_model.graph.initializer.len();
    inspection.data_size = onnx_model
        .graph
        .initializer
        .par_iter()
        .map(|t| {
            if t.dims.is_empty() {
                0
            } else {
                data_type_bytes(t.data_type) * t.dims.iter().map(|d| *d as usize).product::<usize>()
            }
        })
        .sum::<usize>();

    inspection.unique_shapes = onnx_model
        .graph
        .initializer
        .par_iter()
        .map(|t| t.dims.iter().map(|d| *d as usize).collect::<Vec<_>>())
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

    inspection.unique_dtypes = onnx_model
        .graph
        .initializer
        .iter()
        .map(|t| data_type_string(t.data_type).to_string())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    if !onnx_model.producer_name.is_empty() {
        inspection.metadata.insert(
            "producer_name".to_string(),
            onnx_model.producer_name.clone(),
        );
    }

    if !onnx_model.producer_version.is_empty() {
        inspection.metadata.insert(
            "producer_version".to_string(),
            onnx_model.producer_version.clone(),
        );
    }

    if !onnx_model.domain.is_empty() {
        inspection
            .metadata
            .insert("domain".to_string(), onnx_model.domain.clone());
    }

    if !onnx_model.doc_string.is_empty() {
        inspection
            .metadata
            .insert("doc_string".to_string(), onnx_model.doc_string.clone());
    }

    onnx_model.metadata_props.iter().for_each(|prop| {
        inspection
            .metadata
            .insert(prop.key.clone(), prop.value.clone());
    });

    if matches!(detail, DetailLevel::Full) {
        let mut tensor_descriptors = Vec::new();

        for tensor in onnx_model.graph.initializer.iter() {
            if let Some(filter) = &filter {
                if !tensor.name.contains(filter) {
                    continue;
                }
            }

            let mut metadata = super::Metadata::new();
            if !tensor.doc_string.is_empty() {
                metadata.insert("doc_string".to_string(), tensor.doc_string.clone());
            }

            if tensor.data_location.value() == DataLocation::EXTERNAL as i32 {
                metadata.insert("data_location".to_string(), "external".to_string());
                if let Some(external_data) = tensor.external_data.first() {
                    metadata.insert("location".to_string(), external_data.value.clone());
                }
            }

            tensor.metadata_props.iter().for_each(|prop| {
                metadata.insert(prop.key.clone(), prop.value.clone());
            });

            tensor_descriptors.push(TensorDescriptor {
                id: Some(tensor.name.to_string()),
                shape: tensor.dims.iter().map(|d| *d as usize).collect(),
                dtype: data_type_string(tensor.data_type).to_string(),
                size: if tensor.dims.is_empty() {
                    0
                } else {
                    data_type_bytes(tensor.data_type)
                        * tensor.dims.iter().map(|d| *d as usize).product::<usize>()
                },
                metadata,
            });
        }

        inspection.tensors = Some(tensor_descriptors);
    }

    Ok(inspection)
}
