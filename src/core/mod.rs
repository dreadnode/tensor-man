use std::{collections::BTreeMap, fmt, path::PathBuf};

use serde::Serialize;

pub(crate) mod onnx;
pub(crate) mod safetensors;

pub(crate) type Metadata = BTreeMap<String, String>;

#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct TensorDescriptor {
    pub id: Option<String>,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub size: usize,
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Default, Serialize)]
pub(crate) enum FileType {
    #[default]
    Unknown,
    SafeTensors,
    ONNX,
}

impl fmt::Display for FileType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileType::Unknown => write!(f, "Unknown"),
            FileType::SafeTensors => write!(f, "SafeTensors"),
            FileType::ONNX => write!(f, "ONNX"),
        }
    }
}

pub(crate) type Shape = Vec<usize>;

#[derive(Debug, Clone, Default, Serialize)]
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
