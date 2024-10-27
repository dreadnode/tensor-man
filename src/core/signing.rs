use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use blake2::{Blake2b512, Digest};
use ring::{
    rand,
    signature::{self, KeyPair, UnparsedPublicKey, ED25519},
};
use serde::{Deserialize, Serialize};

pub(crate) fn create_key(output: &Path) -> anyhow::Result<()> {
    println!("Generating Ed25519 key pair to {}...", output.display());

    let rng = rand::SystemRandom::new();
    let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng)
        .map_err(|e| anyhow::anyhow!("Failed to generate Ed25519 key pair: {}", e))?;

    std::fs::write(output.join("private.key"), &pkcs8)?;

    let pair = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref())
        .map_err(|e| anyhow::anyhow!("Failed to parse Ed25519 key pair: {}", e))?;

    std::fs::write(output.join("public.key"), pair.public_key())?;

    Ok(())
}

pub(crate) fn load_key(path: &PathBuf) -> anyhow::Result<signature::Ed25519KeyPair> {
    println!("Loading signing key from {}...", path.display());

    let pkcs8_bytes =
        std::fs::read(path).map_err(|e| anyhow::anyhow!("Failed to read key file: {}", e))?;
    signature::Ed25519KeyPair::from_pkcs8(&pkcs8_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to parse Ed25519 key pair: {}", e))
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum HashAlgorithm {
    BLAKE2b512,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum SigningAlgorithm {
    Ed25519,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Algorithms {
    hash: HashAlgorithm,
    signature: SigningAlgorithm,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum Version {
    #[serde(rename = "1.0")]
    V1,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Manifest {
    // version of the manifest format
    pub(crate) version: Version,
    // ISO 8601 timestamp of when the signature was created
    pub(crate) signed_at: String,
    // software name and version
    pub(crate) signed_with: String,
    // hex-encoded public key of the signing key
    pub(crate) public_key: Option<String>,
    pub(crate) algorithms: Algorithms,
    pub(crate) checksums: BTreeMap<String, String>,
    pub(crate) signature: String,

    #[serde(skip_serializing, skip_deserializing)]
    signing_key: Option<signature::Ed25519KeyPair>,
    #[serde(skip_serializing, skip_deserializing)]
    verifying_key: Option<UnparsedPublicKey<Vec<u8>>>,
}

impl Manifest {
    pub(crate) fn for_signing(signing_key: signature::Ed25519KeyPair) -> Self {
        let public_key = signing_key.public_key();
        let mut hasher = Blake2b512::new();
        hasher.update(public_key.as_ref());
        let hash = hasher.finalize();

        Self {
            version: Version::V1,
            signed_at: chrono::Utc::now().to_rfc3339(),
            signed_with: format!("tensor-man v{}", env!("CARGO_PKG_VERSION")),
            // blake2b512 hash of the public key
            public_key: Some(hex::encode(hash)),
            algorithms: Algorithms {
                hash: HashAlgorithm::BLAKE2b512,
                signature: SigningAlgorithm::Ed25519,
            },
            checksums: BTreeMap::new(),
            signature: String::new(),
            signing_key: Some(signing_key),
            verifying_key: None,
        }
    }

    pub(crate) fn for_verifying(public_key_bytes: Vec<u8>) -> Self {
        let public_key = UnparsedPublicKey::new(&ED25519, public_key_bytes);
        let mut hasher = Blake2b512::new();
        hasher.update(public_key.as_ref());
        let hash = hasher.finalize();

        Self {
            version: Version::V1,
            signed_at: chrono::Utc::now().to_rfc3339(),
            signed_with: format!("tensor-man v{}", env!("CARGO_PKG_VERSION")),
            // blake2b512 hash of the public key
            public_key: Some(hex::encode(hash)),
            algorithms: Algorithms {
                hash: HashAlgorithm::BLAKE2b512,
                signature: SigningAlgorithm::Ed25519,
            },
            checksums: BTreeMap::new(),
            signature: String::new(),
            signing_key: None,
            verifying_key: Some(public_key),
        }
    }

    pub(crate) fn compute_checksum(&mut self, path: &Path) -> anyhow::Result<()> {
        // print!("  computing checksum for {} ...", path.to_string_lossy());
        // std::io::stdout().flush().unwrap();

        // let start = Instant::now();

        let mut hasher = Blake2b512::new();
        let mut file = std::fs::File::open(path)?;
        let _bytes_hashed = std::io::copy(&mut file, &mut hasher)?;
        let hash_bytes = hasher.finalize();
        let hash = hex::encode(hash_bytes);

        /*
        println!(
            "hashed {} ({}) in {:?}",
            humansize::format_size(bytes_hashed, humansize::DECIMAL),
            _bytes_hashed,
            start.elapsed()
        ); */

        self.checksums.insert(
            path.file_name().unwrap().to_string_lossy().to_string(),
            hash,
        );
        Ok(())
    }

    fn data_to_sign(&self) -> String {
        // sort hashes by lexicographical order and join them with dots
        let mut checksums = self
            .checksums
            .values()
            .map(|s| s.to_owned())
            .collect::<Vec<String>>();
        checksums.sort();
        checksums.join(".")
    }

    pub(crate) fn create_signature(&mut self) -> anyhow::Result<&str> {
        let data_to_sign = self.data_to_sign();
        // sign data
        self.signature = hex::encode(
            self.signing_key
                .as_ref()
                .unwrap()
                .sign(data_to_sign.as_bytes()),
        );

        Ok(&self.signature)
    }

    fn verify_checksums(&self, checksums: &BTreeMap<String, String>) -> anyhow::Result<()> {
        for (path, expected_checksum) in checksums {
            if !self.checksums.contains_key(path) {
                return Err(anyhow::anyhow!("missing checksum for {}", path));
            }

            let actual_checksum = self.checksums.get(path).unwrap();
            if actual_checksum != expected_checksum {
                return Err(anyhow::anyhow!("checksum mismatch for {}", path));
            }
        }
        Ok(())
    }

    fn verify_signature(&self, signature: &str) -> anyhow::Result<()> {
        let data_to_verify = self.data_to_sign();
        let signature_bytes = hex::decode(signature)?;

        self.verifying_key
            .as_ref()
            .unwrap()
            .verify(data_to_verify.as_bytes(), &signature_bytes)
            .map_err(|e| anyhow::anyhow!("signature verification failed: {}", e))
    }

    pub(crate) fn verify(&self, ref_manifest: &Self) -> anyhow::Result<()> {
        // check public key fingerprint if set
        if ref_manifest.public_key != self.public_key {
            anyhow::bail!("public key fingerprint mismatch");
        }
        // verify individual checksums
        self.verify_checksums(&ref_manifest.checksums)?;
        // verify signature
        self.verify_signature(&ref_manifest.signature)
    }
}
