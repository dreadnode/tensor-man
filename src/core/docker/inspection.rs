use std::path::Path;

use blake2::{Blake2b512, Digest};

use crate::{cli::DetailLevel, core::Inspection};

pub(crate) struct Inspector {
    image_id: String,
    dockerfile: String,
    script: String,
    requirements: String,
}

impl Inspector {
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
        if !super::image_exists(&self.image_id) {
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
            super::build_image(&self.image_id, &dockerfile_path.display().to_string())?;
        }
        Ok(())
    }

    pub fn run(
        &self,
        file_path: &Path,
        additional_files: Vec<String>,
        detail: DetailLevel,
        filter: Option<String>,
    ) -> anyhow::Result<Inspection> {
        if !super::docker_exists() {
            anyhow::bail!("docker is not installed or not running");
        }

        self.build_if_needed()?;

        let file_path = file_path.canonicalize()?;
        let file_name = file_path.file_name().unwrap().to_str().unwrap();

        let mut args = vec![format!("/{}", &file_name)];
        if let Some(filter) = filter {
            args.push(format!("--filter={filter}"));
        }

        if matches!(detail, DetailLevel::Full) {
            args.push("--detailed".to_string());
        }

        let mut volumes = vec![(file_path.display().to_string(), format!("/{}", &file_name))];
        for additional_file in additional_files {
            let add_file_path = file_path.parent().unwrap().join(&additional_file);
            if add_file_path.exists() {
                volumes.push((
                    add_file_path.display().to_string(),
                    format!("/{}", &additional_file),
                ));
            }
        }

        let (stdout, stderr) = super::run(&self.image_id, args, volumes)?;

        if !stderr.is_empty() {
            anyhow::bail!("docker container error: {}", stderr);
        }

        let inspection: Inspection = serde_json::from_str(&stdout)?;
        Ok(inspection)
    }
}
