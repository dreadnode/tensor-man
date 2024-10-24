use std::{collections::HashMap, path::PathBuf};

use dot_graph::Graph;
use protobuf::Message;

use super::protos::{ModelProto, NodeProto};

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

// adapted from https://github.com/onnx/onnx/blob/main/onnx/tools/net_drawer.py

pub(crate) fn create_graph(file_path: PathBuf, output_path: PathBuf) -> anyhow::Result<()> {
    let mut file = std::fs::File::open(&file_path)?;
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
                let node =
                    dot_graph::Node::new(&str_to_node_name(&format!("{}{}", input_name, count)));
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

    std::fs::write(&output_path, dot_string)
        .map_err(|e| anyhow::anyhow!("failed to write dot string to output path: {:?}", e))
}
