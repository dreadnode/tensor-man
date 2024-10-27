use std::{collections::BTreeMap, fmt, path::PathBuf};

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

pub(crate) mod gguf;
pub(crate) mod onnx;
pub(crate) mod safetensors;

pub(crate) mod signing;

pub(crate) type Metadata = BTreeMap<String, String>;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct TensorDescriptor {
    pub id: Option<String>,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub size: usize,
    pub metadata: Metadata,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Default, Serialize, ValueEnum, Deserialize)]
pub(crate) enum FileType {
    #[default]
    Unknown,
    SafeTensors,
    ONNX,
    GGUF,
}

#[allow(dead_code)]
impl FileType {
    pub fn is_unknown(&self) -> bool {
        matches!(self, FileType::Unknown)
    }

    pub fn is_safetensors(&self) -> bool {
        matches!(self, FileType::SafeTensors)
    }

    pub fn is_onnx(&self) -> bool {
        matches!(self, FileType::ONNX)
    }

    pub fn is_gguf(&self) -> bool {
        matches!(self, FileType::GGUF)
    }
}

impl fmt::Display for FileType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileType::Unknown => write!(f, "unknown"),
            FileType::SafeTensors => write!(f, "SafeTensors"),
            FileType::ONNX => write!(f, "ONNX"),
            FileType::GGUF => write!(f, "GGUF"),
        }
    }
}

pub(crate) type Shape = Vec<usize>;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct Inspection {
    pub file_path: PathBuf,
    pub file_type: FileType,
    pub file_size: u64,
    pub header_size: usize,
    pub version: String,
    pub num_tensors: usize,
    pub data_size: usize,
    pub unique_shapes: Vec<Shape>,
    pub unique_dtypes: Vec<String>,
    pub metadata: Metadata,
    pub tensors: Option<Vec<TensorDescriptor>>,
}

impl Inspection {
    pub fn average_tensor_size(&self) -> usize {
        if self.num_tensors == 0 {
            return 0;
        }
        self.data_size / self.num_tensors
    }
}
