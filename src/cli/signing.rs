use crate::core::{handlers::Scope, signing::Manifest};

use super::{CreateKeyArgs, SignArgs, VerifyArgs};

pub(crate) fn create_key(args: CreateKeyArgs) -> anyhow::Result<()> {
    crate::core::signing::create_key(&args.private_key, &args.public_key)
}

pub(crate) fn sign(args: SignArgs) -> anyhow::Result<()> {
    // determine handler
    let handler = crate::core::handlers::handler_for(args.format, &args.file_path, Scope::Signing);
    // load the private key for signing
    let signing_key = crate::core::signing::load_key(&args.key_path)?;
    // get the paths to sign
    let mut paths_to_sign = if let Ok(handler) = handler {
        handler.paths_to_sign(&args.file_path)?
    } else {
        println!("Warning: Unrecognized file format. Signing this file does not ensure that the model data will be signed in its entirety.");
        vec![args.file_path.clone()]
    };
    // create the manifest
    let mut manifest = Manifest::from_signing_key(signing_key);
    // sign
    let signature = manifest.sign(&mut paths_to_sign)?;
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
    // determine handler
    let handler = crate::core::handlers::handler_for(args.format, &args.file_path, Scope::Signing);
    // load signature file to verify
    let signature = Manifest::from_signature_path(&if let Some(path) = args.signature {
        path
    } else {
        args.file_path.with_extension("signature")
    })?;
    // load the public key to verify against
    let mut manifest = Manifest::from_public_key_path(&args.key_path)?;
    // get the paths to verify
    let mut paths_to_verify = if let Ok(handler) = handler {
        handler.paths_to_sign(&args.file_path)?
    } else {
        println!("Warning: Unrecognized file format. Signing this file does not ensure that the model data will be signed in its entirety.");
        vec![args.file_path.clone()]
    };
    // this will compute the checksums and verify the signature
    manifest.verify(&mut paths_to_verify, &signature)?;

    println!("Signature verified");

    Ok(())
}
