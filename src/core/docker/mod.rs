use core::str;
use std::{
    path::{Path, PathBuf},
    process::Command,
};

mod inspection;

pub(crate) use inspection::*;

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
