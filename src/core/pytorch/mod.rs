use std::path::{Path, PathBuf};

use crate::cli::DetailLevel;

use super::{docker, Inspection};

pub(crate) fn is_pytorch(file_path: &Path) -> bool {
    file_path
        .extension()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("")
        .to_ascii_lowercase()
        == "pt"
        || file_path
            .file_name()
            .unwrap_or_default()
            .to_str()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .ends_with("pytorch_model.bin")
}

pub(crate) fn paths_to_sign(file_path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    // TODO: can a pytorch model reference external files?
    Ok(vec![file_path.to_path_buf()])
}

pub(crate) fn inspect(
    file_path: PathBuf,
    detail: DetailLevel,
    filter: Option<String>,
) -> anyhow::Result<Inspection> {
    docker::Inspector::new(
        include_str!("inspect.Dockerfile"),
        include_str!("inspect.py"),
        include_str!("inspect.requirements"),
    )
    .run(file_path, vec![], detail, filter)
}
