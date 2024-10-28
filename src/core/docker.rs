use core::str;
use std::{
    path::{Path, PathBuf},
    process::Command,
};

use blake2::{Blake2b512, Digest};

use crate::cli::DetailLevel;

use super::Inspection;

fn run_command(command: &str, args: &[&str]) -> anyhow::Result<(String, String)> {
    let output = Command::new(command).args(args).output()?;

    let stdout = str::from_utf8(&output.stdout)?.to_string();
    let stderr = str::from_utf8(&output.stderr)?.to_string();

    if output.status.success() {
        Ok((stdout, stderr))
    } else {
        Err(anyhow::anyhow!(
            "Command `{} {}` failed with exit code {:?}\nStderr: {}\nStdout: {}",
            command,
            args.join(" "),
            output.status.code(),
            stderr,
            stdout
        ))
    }
}

fn docker_exists() -> bool {
    run_command("docker", &["version"]).is_ok()
}

fn image_exists(image: &str) -> bool {
    run_command(
        "sh",
        &["-c", &format!("docker images -q '{image}' | grep -q .")],
    )
    .is_ok()
}

fn build_image(name: &str, path: &str) -> anyhow::Result<()> {
    let dockerfile = PathBuf::from(path);
    if !dockerfile.exists() {
        return Err(anyhow::anyhow!("dockerfile '{}' does not exist", path));
    } else if !dockerfile.is_file() {
        return Err(anyhow::anyhow!("path '{}' is not a dockerfile", path));
    }

    run_command(
        "sh",
        &[
            "-c",
            &format!(
                "docker build -f '{}' -t '{name}' --quiet '{}'",
                dockerfile.display(),
                dockerfile.parent().unwrap_or(Path::new(".")).display(),
            ),
        ],
    )?;

    Ok(())
}

fn run(
    image_id: &str,
    args: Vec<String>,
    volumes: Vec<(String, String)>,
) -> anyhow::Result<(String, String)> {
    let mut all_args = vec![
        "run".to_string(),
        // remove after execution
        "--rm".to_string(),
        // disable network
        "--network=none".to_string(),
    ];

    for (src, dst) in volumes {
        all_args.push(format!("-v{src}:{dst}"));
    }

    all_args.push(image_id.to_string());
    all_args.extend(args);

    run_command(
        "docker",
        all_args
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .as_slice(),
    )
}

pub(crate) struct DockerizedInspection {
    image_id: String,
    dockerfile: String,
    script: String,
    requirements: String,
}

impl DockerizedInspection {
    pub fn new(dockerfile: &str, script: &str, requirements: &str) -> Self {
        // make sure that the image id is deterministic and unique to the contents
        let mut hasher = Blake2b512::new();
        hasher.update(dockerfile);
        hasher.update(script);
        hasher.update(requirements);

        let image_id = format!("tensor-man-inspect-{}", hex::encode(hasher.finalize()));
        Self {
            image_id,
            dockerfile: dockerfile.to_string(),
            script: script.to_string(),
            requirements: requirements.to_string(),
        }
    }

    fn build_if_needed(&self) -> anyhow::Result<()> {
        if !image_exists(&self.image_id) {
            println!("building image '{}'", &self.image_id);

            // extract the image assets in a temporary directory
            let tmp_dir = tempfile::tempdir()?;
            let base_path = tmp_dir.path().join(&self.image_id);
            std::fs::create_dir_all(&base_path)?;

            let dockerfile_path = base_path.join("Dockerfile");
            std::fs::write(&dockerfile_path, &self.dockerfile)?;
            std::fs::write(base_path.join("script_main.py"), &self.script)?;
            std::fs::write(base_path.join("requirements.txt"), &self.requirements)?;

            // build the image
            build_image(&self.image_id, &dockerfile_path.display().to_string())?;
        }
        Ok(())
    }

    pub fn run(
        &self,
        file_path: PathBuf,
        detail: DetailLevel,
        filter: Option<String>,
    ) -> anyhow::Result<Inspection> {
        if !docker_exists() {
            anyhow::bail!("docker is not installed or not running");
        }

        self.build_if_needed()?;

        let file_path = file_path.canonicalize()?;
        let mut args = vec!["/model.bin".to_string()];
        if let Some(filter) = filter {
            args.push(format!("--filter={filter}"));
        }

        if matches!(detail, DetailLevel::Full) {
            args.push("--detailed".to_string());
        }

        let (stdout, stderr) = run(
            &self.image_id,
            args,
            vec![(file_path.display().to_string(), "/model.bin".to_string())],
        )?;

        if !stderr.is_empty() {
            anyhow::bail!("docker container error: {}", stderr);
        }

        let inspection: Inspection = serde_json::from_str(&stdout)?;
        Ok(inspection)
    }
}
