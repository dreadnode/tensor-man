use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

mod protos;

use dot_graph::Graph;
use protobuf::Message;

use protos::{tensor_proto::DataLocation, ModelProto, NodeProto, TensorProto};
use rayon::prelude::*;

use crate::{
    cli::DetailLevel,
    core::{handlers::Handler, FileType, Inspection, Metadata, TensorDescriptor},
};

use super::Scope;

#[inline]
fn data_type_bits(dtype: i32) -> usize {
    match dtype {
        1 => 32,   // float
        2 => 8,    // uint8_t
        3 => 8,    // int8_t
        4 => 16,   // uint16_t
        5 => 16,   // int16_t
        6 => 32,   // int32_t
        7 => 64,   // int64_t
        8 => 8,    // string (assuming 8 bits per character)
        9 => 8,    // bool (typically 8 bits in most systems)
        10 => 16,  // FLOAT16
        11 => 64,  // DOUBLE
        12 => 32,  // UINT32
        13 => 64,  // UINT64
        14 => 64,  // COMPLEX64 (two 32-bit floats)
        15 => 128, // COMPLEX128 (two 64-bit floats)
        16 => 16,  // BFLOAT16
        17 => 8,   // FLOAT8E4M3FN
        18 => 8,   // FLOAT8E4M3FNUZ
        19 => 8,   // FLOAT8E5M2
        20 => 8,   // FLOAT8E5M2FNUZ
        21 => 4,   // UINT4
        22 => 4,   // INT4
        23 => 4,   // FLOAT4E2M1
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

fn build_tensor_descriptor(tensor: &TensorProto) -> TensorDescriptor {
    let mut metadata = Metadata::new();
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

    TensorDescriptor {
        id: Some(tensor.name.to_string()),
        shape: tensor.dims.iter().map(|d| *d as usize).collect(),
        dtype: data_type_string(tensor.data_type).to_string(),
        size: if tensor.dims.is_empty() {
            0
        } else {
            (data_type_bits(tensor.data_type)
                * tensor.dims.iter().map(|d| *d as usize).product::<usize>())
                / 8
        },
        metadata,
    }
}

#[inline]
fn is_letter_or_underscore_or_dot(c: char) -> bool {
    in_range('a', c, 'z') || in_range('A', c, 'Z') || c == '_' || c == '.'
}

#[inline]
fn is_constituent(c: char) -> bool {
    is_letter_or_underscore_or_dot(c) || in_range('0', c, '9')
}

#[inline]
fn in_range(low: char, c: char, high: char) -> bool {
    low as usize <= c as usize && c as usize <= high as usize
}

fn str_to_node_name(s: &str) -> String {
    let mut result = String::new();

    // make sure the name starts with a letter or underscore or dot
    if let Some(first) = s.chars().next() {
        if !is_letter_or_underscore_or_dot(first) {
            result.push('_');
        }
    }

    for c in s.chars() {
        if is_constituent(c) {
            result.push(c);
        } else {
            result.push('_');
        }
    }
    result.trim_matches('_').to_string()
}

fn op_to_dot_node(op: &NodeProto, op_id: usize) -> dot_graph::Node {
    let node_label = if !op.name.is_empty() {
        format!("{}/{} (op#{})", op.name, op.op_type, op_id)
    } else {
        format!("{} (op#{})", op.op_type, op_id)
    };
    let node_name = str_to_node_name(&node_label);

    dot_graph::Node::new(&node_name).label(&node_label)
}

pub(crate) struct OnnxHandler;

impl OnnxHandler {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl Handler for OnnxHandler {
    fn file_type(&self) -> FileType {
        FileType::ONNX
    }

    fn is_handler_for(&self, file_path: &Path, _scope: &Scope) -> bool {
        file_path
            .extension()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("")
            .to_ascii_lowercase()
            == "onnx"
    }

    fn paths_to_sign(&self, file_path: &Path) -> anyhow::Result<Vec<PathBuf>> {
        let base_path = file_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("no parent path"))?;
        let mut file = std::fs::File::open(file_path)?;
        let onnx_model: ModelProto = Message::parse_from_reader(&mut file)?;

        // ONNX files can contain external data
        let external_paths: HashSet<PathBuf> = onnx_model
            .graph
            .initializer
            .par_iter()
            .filter(|t| t.data_location.value() == DataLocation::EXTERNAL as i32)
            .filter_map(|t| {
                t.external_data
                    .first()
                    .map(|data| PathBuf::from(&data.value))
                    .map(|p| {
                        if p.is_relative() {
                            base_path.join(p)
                        } else {
                            p
                        }
                    })
            })
            .collect();

        let mut paths = vec![file_path.to_path_buf()];
        paths.extend(external_paths);

        Ok(paths)
    }

    fn inspect(
        &self,
        file_path: &Path,
        detail: DetailLevel,
        filter: Option<String>,
    ) -> anyhow::Result<Inspection> {
        let mut inspection = Inspection::default();

        let mut file = std::fs::File::open(file_path)?;

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
                    data_type_bits(t.data_type)
                        * t.dims.iter().map(|d| *d as usize).product::<usize>()
                }
            })
            .sum::<usize>()
            / 8;

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
            .par_iter()
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
            inspection.tensors = Some(
                onnx_model
                    .graph
                    .initializer
                    .par_iter()
                    .filter(|t_info| filter.as_ref().map_or(true, |f| t_info.name.contains(f)))
                    .map(build_tensor_descriptor)
                    .collect(),
            );
        }

        Ok(inspection)
    }

    // adapted from https://github.com/onnx/onnx/blob/main/onnx/tools/net_drawer.py
    fn create_graph(&self, file_path: &Path, output_path: &Path) -> anyhow::Result<()> {
        let mut file = std::fs::File::open(file_path)?;
        let onnx_model: ModelProto = Message::parse_from_reader(&mut file)?;
        let mut dot_graph = Graph::new(
            // make sure the name is quoted
            &format!(
                "{:?}",
                file_path.file_stem().unwrap().to_string_lossy().as_ref()
            ),
            dot_graph::Kind::Digraph,
        );
        let mut dot_nodes = HashMap::new();
        let mut dot_node_counts = HashMap::new();

        for (op_id, op) in onnx_model.graph.node.iter().enumerate() {
            let op_node = op_to_dot_node(op, op_id);
            dot_graph.add_node(op_node.clone());
            for input_name in &op.input {
                let input_node = dot_nodes.entry(input_name.clone()).or_insert_with(|| {
                    let count = dot_node_counts.entry(input_name.clone()).or_insert(0);
                    let node = dot_graph::Node::new(&str_to_node_name(&format!(
                        "{}{}",
                        input_name, count
                    )));
                    node.label(input_name);
                    *count += 1;
                    node
                });
                dot_graph.add_node(input_node.clone());
                dot_graph.add_edge(dot_graph::Edge::new(&input_node.name, &op_node.name, ""));
            }
            for output_name in &op.output {
                let count = dot_node_counts.entry(output_name.clone()).or_insert(0);
                let output_node =
                    dot_graph::Node::new(&str_to_node_name(&format!("{}{}", output_name, count)));
                output_node.label(output_name);
                dot_nodes.insert(output_name.clone(), output_node.clone());
                dot_graph.add_node(output_node.clone());
                dot_graph.add_edge(dot_graph::Edge::new(&op_node.name, &output_node.name, ""));
            }
        }

        let dot_string = dot_graph.to_dot_string()?;

        std::fs::write(output_path, dot_string)
            .map_err(|e| anyhow::anyhow!("failed to write dot string to output path: {:?}", e))
    }
}
