use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use anyhow::anyhow;
use glob::glob;

use crate::core::{handlers::Scope, signing::Manifest, FileType};

use super::{CreateKeyArgs, SignArgs, VerifyArgs};

pub(crate) fn create_key(args: CreateKeyArgs) -> anyhow::Result<()> {
    crate::core::signing::create_key(&args.private_key, &args.public_key)
}

fn get_paths_for(format: Option<FileType>, file_path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    // determine handler
    let handler = crate::core::handlers::handler_for(format, file_path, Scope::Signing);
    // get the paths to sign or verify
    if let Ok(handler) = handler {
        handler.paths_to_sign(file_path).map(|paths| {
            paths
                .iter()
                .map(|path| path.canonicalize().unwrap())
                .collect()
        })
    } else {
        Ok(vec![file_path.canonicalize()?.to_path_buf()])
    }
}

fn get_paths_of_interest(
    format: Option<FileType>,
    file_path: &Path,
) -> anyhow::Result<Vec<PathBuf>> {
    let paths = if file_path.is_file() {
        // single file case
        get_paths_for(format, file_path)?
    } else {
        let mut unique = HashSet::new();

        // collect all files in the directory
        for entry in glob(file_path.join("**/*").to_str().unwrap())? {
            match entry {
                Ok(path) => {
                    if path.is_file() {
                        unique.extend(get_paths_for(format.clone(), &path)?);
                    }
                }
                Err(e) => println!("{:?}", e),
            }
        }

        unique.into_iter().collect::<Vec<PathBuf>>()
    };

    if paths.is_empty() {
        return Err(anyhow!("no compatible paths found"));
    }

    Ok(paths)
}

fn signature_path(file_path: &Path, signature_path: Option<PathBuf>) -> PathBuf {
    if let Some(path) = signature_path {
        path.canonicalize().unwrap()
    } else if file_path.is_file() {
        file_path
            .with_extension("signature")
            .canonicalize()
            .unwrap()
    } else {
        file_path.join("tensor-man.signature")
    }
}

pub(crate) fn sign(args: SignArgs) -> anyhow::Result<()> {
    // load the private key for signing
    let signing_key = crate::core::signing::load_key(&args.key_path)?;
    // get the paths to sign
    let mut paths_to_sign = get_paths_of_interest(args.format, &args.file_path)?;
    let base_path = if args.file_path.is_file() {
        args.file_path.parent().unwrap().to_path_buf()
    } else {
        args.file_path.to_path_buf()
    };
    // create the manifest
    let mut manifest = Manifest::from_signing_key(&base_path, signing_key)?;

    // sign
    let signature = manifest.sign(&mut paths_to_sign)?;
    println!("Signature: {}", signature);

    // write manifest to file
    let signature_path = signature_path(&args.file_path, args.output);

    std::fs::write(&signature_path, serde_json::to_string(&manifest)?)?;

    println!("Manifest written to {}", signature_path.display());

    Ok(())
}

pub(crate) fn verify(args: VerifyArgs) -> anyhow::Result<()> {
    let base_path = if args.file_path.is_file() {
        args.file_path.parent().unwrap().to_path_buf()
    } else {
        args.file_path.to_path_buf()
    };

    // load signature file to verify
    let signature_path = signature_path(&args.file_path, args.signature);

    println!("Verifying signature: {}", signature_path.display());

    let signature = Manifest::from_signature_path(&base_path, &signature_path)?;

    // load the public key to verify against
    let mut manifest = Manifest::from_public_key_path(&base_path, &args.key_path)?;
    // get the paths to verify
    let mut paths_to_verify = get_paths_of_interest(args.format, &args.file_path)?;
    // remove the signature file from the list
    paths_to_verify.retain(|p| p != &signature_path);

    // this will compute the checksums and verify the signature
    manifest.verify(&mut paths_to_verify, &signature)?;

    println!("Signature verified");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile::TempDir;

    #[test]
    fn test_get_paths_single_file() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("model.safetensors");

        File::create(&file_path)?;

        let paths = get_paths_of_interest(None, &file_path)?;
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], file_path.canonicalize()?);

        Ok(())
    }

    #[test]
    fn test_get_paths_directory() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;

        // Create multiple files
        File::create(temp_dir.path().join("model.safetensors"))?;
        File::create(temp_dir.path().join("model.bin"))?;
        File::create(temp_dir.path().join("other.txt"))?;

        let paths = get_paths_of_interest(None, temp_dir.path())?;
        assert_eq!(paths.len(), 3);

        // Sort paths for consistent comparison
        let mut paths: Vec<String> = paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        paths.sort();

        assert_eq!(paths, vec!["model.bin", "model.safetensors", "other.txt"]);

        Ok(())
    }

    #[test]
    fn test_get_paths_with_format_override() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;

        // Create files with different extensions
        File::create(temp_dir.path().join("model.custom"))?;
        File::create(temp_dir.path().join("model.safetensors"))?;

        let paths = get_paths_of_interest(
            Some(FileType::SafeTensors),
            &temp_dir.path().join("model.custom"),
        )?;
        assert_eq!(paths.len(), 1);
        assert!(paths[0].to_string_lossy().ends_with("model.custom"));

        Ok(())
    }

    #[test]
    fn test_get_paths_sharded_files() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;

        // Create sharded files
        File::create(temp_dir.path().join("model-00001-of-00002.safetensors"))?;
        File::create(temp_dir.path().join("model-00002-of-00002.safetensors"))?;
        File::create(temp_dir.path().join("other.txt"))?;

        let paths = get_paths_of_interest(None, temp_dir.path())?;
        assert_eq!(paths.len(), 3);

        let mut paths: Vec<String> = paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        paths.sort();

        assert_eq!(
            paths,
            vec![
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "other.txt"
            ]
        );

        Ok(())
    }

    #[test]
    fn test_get_paths_nonexistent() {
        let result = get_paths_of_interest(None, &PathBuf::from("/nonexistent/path"));
        assert!(result.is_err());
    }

    #[test]
    fn test_get_paths_nested_and_hidden() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;

        // Create nested directory structure
        let nested_dir = temp_dir.path().join("nested");
        let deep_dir = nested_dir.join("deep");
        std::fs::create_dir_all(&deep_dir)?;

        // Create various files including hidden ones
        File::create(temp_dir.path().join(".hidden"))?;
        File::create(temp_dir.path().join("regular.txt"))?;
        File::create(nested_dir.join(".hidden_nested"))?;
        File::create(nested_dir.join("nested.bin"))?;
        File::create(deep_dir.join(".very_hidden"))?;
        File::create(deep_dir.join("deep.dat"))?;

        let paths = get_paths_of_interest(None, temp_dir.path())?;
        assert_eq!(paths.len(), 6); // Should find all 6 files

        let mut paths: Vec<String> = paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        paths.sort();

        assert_eq!(
            paths,
            vec![
                ".hidden",
                ".hidden_nested",
                ".very_hidden",
                "deep.dat",
                "nested.bin",
                "regular.txt"
            ]
        );

        Ok(())
    }

    #[test]
    fn test_paths_are_canonicalized() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("test.txt");
        File::create(&file_path)?;

        // Get paths using absolute path first to verify it works
        let paths = get_paths_of_interest(None, &file_path)?;
        assert_eq!(paths.len(), 1);
        let canonical_path = file_path.canonicalize()?;
        assert_eq!(&paths[0], &canonical_path);

        // Change into temp dir to test relative path
        std::env::set_current_dir(temp_dir.path())?;

        // Get paths using relative path
        let relative_path = PathBuf::from("test.txt");
        let paths = get_paths_of_interest(None, &relative_path)?;

        assert_eq!(paths.len(), 1);
        let returned_path = &paths[0];

        // Verify the returned path is absolute and canonicalized
        assert!(returned_path.is_absolute());
        assert_eq!(returned_path, &canonical_path);

        Ok(())
    }
}
