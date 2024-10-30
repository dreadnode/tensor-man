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

fn get_paths_of_interest(
    format: Option<FileType>,
    file_path: &Path,
) -> anyhow::Result<Vec<PathBuf>> {
    let paths = if file_path.is_file() {
        // single file case
        // determine handler
        let handler = crate::core::handlers::handler_for(format, file_path, Scope::Signing);
        // get the paths to sign or verify
        if let Ok(handler) = handler {
            handler.paths_to_sign(file_path)?
        } else {
            println!("Warning: Unrecognized file format. Signing this file does not ensure that the model data will be signed in its entirety.");
            vec![file_path.to_path_buf()]
        }
    } else {
        let mut unique = HashSet::new();

        // collect all files in the directory
        for entry in glob(file_path.join("**/*.*").to_str().unwrap())? {
            match entry {
                Ok(path) => {
                    if path.is_file() {
                        // determine handler
                        if let Ok(handler) = crate::core::handlers::handler_for(
                            format.clone(),
                            &path,
                            Scope::Signing,
                        ) {
                            // add only if handled
                            unique.extend(handler.paths_to_sign(&path)?);
                        }
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
        path
    } else if file_path.is_file() {
        file_path.with_extension("signature")
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

    // this will compute the checksums and verify the signature
    manifest.verify(&mut paths_to_verify, &signature)?;

    println!("Signature verified");

    Ok(())
}
