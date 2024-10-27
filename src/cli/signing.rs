use crate::core::{safetensors::is_safetensors_index, signing::Manifest, FileType};

use super::{CreateKeyArgs, SignArgs, VerifyArgs};

pub(crate) fn create_key(args: CreateKeyArgs) -> anyhow::Result<()> {
    crate::core::signing::create_key(&args.private_key, &args.public_key)
}

pub(crate) fn sign(args: SignArgs) -> anyhow::Result<()> {
    let signing_key = crate::core::signing::load_key(&args.key_path)?;

    let forced_format = args.format.unwrap_or(FileType::Unknown);
    let file_ext = args
        .file_path
        .extension()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("")
        .to_ascii_lowercase();

    let mut paths_to_sign = if forced_format.is_safetensors()
        || file_ext == "safetensors"
        || is_safetensors_index(&args.file_path)
    {
        crate::core::safetensors::paths_to_sign(&args.file_path)?
    } else if forced_format.is_onnx() || file_ext == "onnx" {
        crate::core::onnx::paths_to_sign(&args.file_path)?
    } else if forced_format.is_gguf() || file_ext == "gguf" {
        crate::core::gguf::paths_to_sign(&args.file_path)?
    } else {
        println!("Warning: Unrecognized file format. Signing this file does not ensure that the model data will be signed in its entirety.");
        vec![args.file_path.clone()]
    };

    paths_to_sign.sort();

    let mut manifest = Manifest::for_signing(signing_key);

    // compute checksums for all files
    for path in paths_to_sign {
        println!("Signing {} ...", path.display());

        manifest.compute_checksum(&path)?;
    }

    // sign
    let signature = manifest.create_signature()?;
    println!("Signature: {}", signature);

    // write manifest to file
    let manifest_path = if let Some(path) = args.output {
        path
    } else {
        args.file_path.with_extension("signature")
    };

    std::fs::write(&manifest_path, serde_json::to_string(&manifest)?)?;

    println!("Manifest written to {}", manifest_path.display());

    Ok(())
}

pub(crate) fn verify(args: VerifyArgs) -> anyhow::Result<()> {
    let manifest_path = if let Some(path) = args.signature {
        path
    } else {
        args.file_path.with_extension("signature")
    };

    let raw = std::fs::read_to_string(&manifest_path)?;
    let ref_manifest: Manifest = serde_json::from_str(&raw)?;

    let forced_format = args.format.unwrap_or(FileType::Unknown);
    let file_ext = args
        .file_path
        .extension()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("")
        .to_ascii_lowercase();

    let raw = std::fs::read(&args.key_path)?;
    let mut manifest = Manifest::for_verifying(raw);

    let mut paths_to_verify = if forced_format.is_safetensors()
        || file_ext == "safetensors"
        || is_safetensors_index(&args.file_path)
    {
        crate::core::safetensors::paths_to_sign(&args.file_path)?
    } else if forced_format.is_onnx() || file_ext == "onnx" {
        crate::core::onnx::paths_to_sign(&args.file_path)?
    } else if forced_format.is_gguf() || file_ext == "gguf" {
        crate::core::gguf::paths_to_sign(&args.file_path)?
    } else {
        println!("Warning: Unrecognized file format. Verifying this file does not ensure that the model data will be verified in its entirety.");
        vec![args.file_path.clone()]
    };

    paths_to_verify.sort();

    // compute checksums for all files
    for path in paths_to_verify {
        println!("Hashing {} ...", path.display());

        manifest.compute_checksum(&path)?;
    }

    // verify
    manifest.verify(&ref_manifest)?;

    println!("Signature verified");

    Ok(())
}
