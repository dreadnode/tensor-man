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

pub(crate) fn create_key(private_key: &Path, public_key: &Path) -> anyhow::Result<()> {
    println!("Generating Ed25519 private key ...");

    let rng = rand::SystemRandom::new();
    let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng)
        .map_err(|e| anyhow::anyhow!("Failed to generate Ed25519 key pair: {}", e))?;

    println!("Writing private key to {} ...", private_key.display());
    std::fs::write(private_key, &pkcs8)?;

    println!("Writing public key to {} ...", public_key.display());
    let pair = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref())
        .map_err(|e| anyhow::anyhow!("Failed to parse Ed25519 key pair: {}", e))?;

    std::fs::write(public_key, pair.public_key())?;

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

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct Manifest {
    // version of the manifest format
    pub(crate) version: Version,
    // ISO 8601 timestamp of when the signature was created
    pub(crate) signed_at: String,
    // software name and version
    pub(crate) signed_with: String,
    // hex-encoded public key of the signing key
    pub(crate) public_key: Option<String>,
    // algorithms used for hashing and signing
    pub(crate) algorithms: Algorithms,
    // checksums of the files
    pub(crate) checksums: BTreeMap<String, String>,
    // hex-encoded signature of the checksums
    pub(crate) signature: String,

    #[serde(skip_serializing, skip_deserializing)]
    base_path: PathBuf,
    #[serde(skip_serializing, skip_deserializing)]
    signing_key: Option<signature::Ed25519KeyPair>,
    #[serde(skip_serializing, skip_deserializing)]
    verifying_key: Option<UnparsedPublicKey<Vec<u8>>>,
}

impl Manifest {
    pub(crate) fn from_signature_path(base_path: &Path, path: &Path) -> anyhow::Result<Self> {
        let mut this: Manifest = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        this.base_path = base_path.canonicalize()?;
        Ok(this)
    }

    pub(crate) fn from_signing_key(
        base_path: &Path,
        signing_key: signature::Ed25519KeyPair,
    ) -> anyhow::Result<Self> {
        let public_key = signing_key.public_key();
        let mut hasher = Blake2b512::new();
        hasher.update(public_key.as_ref());
        let hash = hasher.finalize();

        Ok(Self {
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
            base_path: base_path.canonicalize()?,
        })
    }

    pub(crate) fn from_public_key(
        base_path: &Path,
        public_key_bytes: Vec<u8>,
    ) -> anyhow::Result<Self> {
        let public_key = UnparsedPublicKey::new(&ED25519, public_key_bytes);
        let mut hasher = Blake2b512::new();
        hasher.update(public_key.as_ref());
        let hash = hasher.finalize();

        Ok(Self {
            version: Version::V1,
            signed_at: chrono::Utc::now().to_rfc3339(),
            signed_with: format!("{} v{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
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
            base_path: base_path.canonicalize()?,
        })
    }

    pub(crate) fn from_public_key_path(
        base_path: &Path,
        public_key: &Path,
    ) -> anyhow::Result<Self> {
        let public_key_bytes = std::fs::read(public_key)?;
        Self::from_public_key(base_path, public_key_bytes)
    }

    fn compute_checksum(&mut self, path: &Path) -> anyhow::Result<()> {
        // print!("  computing checksum for {} ...", path.to_string_lossy());
        // std::io::stdout().flush().unwrap();

        // let start = Instant::now();

        let path = path.canonicalize()?;

        let mut hasher = Blake2b512::new();
        let mut file = std::fs::File::open(&path)?;
        let _ = std::io::copy(&mut file, &mut hasher)?;
        let hash_bytes = hasher.finalize();
        let hash = hex::encode(hash_bytes);

        /*
        println!(
            "hashed {} ({}) in {:?}",
            humansize::format_size(bytes_hashed, humansize::DECIMAL),
            _bytes_hashed,
            start.elapsed()
        ); */

        if let Err(e) = path.strip_prefix(&self.base_path) {
            panic!(
                "base_path={} path={} error={}",
                self.base_path.display(),
                path.display(),
                e
            );
        }

        self.checksums.insert(
            path.strip_prefix(&self.base_path)
                .unwrap()
                .to_string_lossy()
                .to_string(),
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

    fn create_signature(&mut self) -> anyhow::Result<&str> {
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
        // check if all the required checksums are present, use the checksum value instead
        // of the path as the file name might be different
        let provided_checksums = checksums.values().collect::<Vec<&String>>();
        for (path, required_checksum) in self.checksums.iter() {
            if !provided_checksums.contains(&required_checksum) {
                return Err(anyhow::anyhow!("missing or invalid checksum for {}", path));
            }
        }
        // check if all the provided checksums are valid
        let required_checksums = self.checksums.values().collect::<Vec<&String>>();
        for (path, expected_checksum) in checksums {
            if !required_checksums.contains(&expected_checksum) {
                return Err(anyhow::anyhow!("invalid checksum for {}", path));
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

    pub(crate) fn sign(&mut self, paths: &mut [PathBuf]) -> anyhow::Result<&str> {
        paths.sort();

        // compute checksums for all files
        for path in paths {
            println!("Signing {} ...", path.display());

            self.compute_checksum(path)?;
        }

        // sign
        self.create_signature()
    }

    pub(crate) fn verify(&mut self, paths: &mut [PathBuf], signature: &Self) -> anyhow::Result<()> {
        paths.sort();

        // compute checksums for all files
        for path in paths {
            println!("Hashing {} ...", path.display());

            self.compute_checksum(path)?;
        }

        // check public key fingerprint if set
        if signature.public_key != self.public_key {
            anyhow::bail!("public key fingerprint mismatch");
        }
        // verify individual checksums
        self.verify_checksums(&signature.checksums)?;
        // verify signature
        self.verify_signature(&signature.signature)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use tempfile::NamedTempFile;

    use super::*;

    fn create_test_keypair() -> signature::Ed25519KeyPair {
        let rng = rand::SystemRandom::new();
        let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
        signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap()
    }

    fn create_temp_file_with_content(content: &str) -> anyhow::Result<NamedTempFile> {
        let mut temp_file = NamedTempFile::new()?;
        temp_file.write_all(content.as_bytes())?;
        temp_file.flush()?;
        Ok(temp_file)
    }

    #[test]
    fn test_will_create_a_signature() {
        let keypair = create_test_keypair();

        let temp_file = create_temp_file_with_content("test").unwrap();

        let base_path = temp_file.path().parent().unwrap();

        let mut manifest = Manifest::from_signing_key(&base_path, keypair).unwrap();

        manifest.compute_checksum(&temp_file.path()).unwrap();
        let signature = manifest.create_signature().unwrap();

        assert!(!signature.is_empty());

        assert!(matches!(manifest.version, Version::V1));
        assert!(manifest.signed_at.len() > 0);
        assert!(manifest.signed_with.len() > 0);
        assert!(manifest.public_key.is_some());
        assert!(matches!(
            manifest.algorithms.hash,
            HashAlgorithm::BLAKE2b512
        ));
        assert!(matches!(
            manifest.algorithms.signature,
            SigningAlgorithm::Ed25519
        ));

        assert_eq!(manifest.checksums.len(), 1);
        assert_eq!(manifest.checksums.values().next().unwrap(), "a71079d42853dea26e453004338670a53814b78137ffbed07603a41d76a483aa9bc33b582f77d30a65e6f29a896c0411f38312e1d66e0bf16386c86a89bea572");
    }

    #[test]
    fn test_will_verify_correct_signature() {
        let keypair = create_test_keypair();
        let pub_key = keypair.public_key().as_ref().to_vec();
        let temp_file = create_temp_file_with_content("test").unwrap();
        let base_path = temp_file.path().parent().unwrap();

        let mut ref_manifest = Manifest::from_signing_key(&base_path, keypair).unwrap();

        let mut paths = vec![temp_file.path().to_path_buf()];

        _ = ref_manifest.sign(&mut paths).unwrap();

        let mut manifest = Manifest::from_public_key(&base_path, pub_key).unwrap();

        manifest.verify(&mut paths, &ref_manifest).unwrap();
    }

    #[test]
    fn test_wont_verify_with_wrong_key() {
        let keypair = create_test_keypair();
        let other_keypair = create_test_keypair();
        let pub_key = other_keypair.public_key().as_ref().to_vec();
        let temp_file = create_temp_file_with_content("test").unwrap();
        let base_path = temp_file.path().parent().unwrap();

        let mut ref_manifest = Manifest::from_signing_key(&base_path, keypair).unwrap();

        let mut paths = vec![temp_file.path().to_path_buf()];

        ref_manifest.compute_checksum(&temp_file.path()).unwrap();
        ref_manifest.create_signature().unwrap();

        let mut manifest = Manifest::from_public_key(&base_path, pub_key).unwrap();

        manifest.compute_checksum(&temp_file.path()).unwrap();

        assert!(manifest.verify(&mut paths, &ref_manifest).is_err());
    }

    #[test]
    fn test_wont_verify_a_tampered_file() {
        let keypair = create_test_keypair();
        let pub_key = keypair.public_key().as_ref().to_vec();

        let temp_file = create_temp_file_with_content("test").unwrap();
        let base_path = temp_file.path().parent().unwrap();

        let mut ref_manifest = Manifest::from_signing_key(&base_path, keypair).unwrap();

        let mut paths = vec![temp_file.path().to_path_buf()];

        ref_manifest.compute_checksum(&temp_file.path()).unwrap();
        ref_manifest.create_signature().unwrap();

        let mut manifest = Manifest::from_public_key(&base_path, pub_key).unwrap();

        let temp_file = create_temp_file_with_content("tost").unwrap();

        manifest.compute_checksum(&temp_file.path()).unwrap();

        assert!(manifest.verify(&mut paths, &ref_manifest).is_err());
    }

    #[test]
    fn test_wont_verify_empty_file() {
        let keypair = create_test_keypair();
        let pub_key = keypair.public_key().as_ref().to_vec();

        let temp_file = create_temp_file_with_content("test").unwrap();
        let base_path = temp_file.path().parent().unwrap();

        let mut ref_manifest = Manifest::from_signing_key(&base_path, keypair).unwrap();

        let mut paths = vec![temp_file.path().to_path_buf()];

        ref_manifest.compute_checksum(&temp_file.path()).unwrap();
        ref_manifest.create_signature().unwrap();

        let mut manifest = Manifest::from_public_key(&base_path, pub_key).unwrap();

        let empty_file = create_temp_file_with_content("").unwrap();
        manifest.compute_checksum(&empty_file.path()).unwrap();

        assert!(manifest.verify(&mut paths, &ref_manifest).is_err());
    }

    #[test]
    fn test_wont_verify_extra_file() {
        let keypair = create_test_keypair();
        let pub_key = keypair.public_key().as_ref().to_vec();

        let temp_file = create_temp_file_with_content("test").unwrap();
        let base_path = temp_file.path().parent().unwrap();

        let mut ref_manifest = Manifest::from_signing_key(&base_path, keypair).unwrap();

        let mut paths = vec![temp_file.path().to_path_buf()];

        ref_manifest.compute_checksum(&temp_file.path()).unwrap();
        ref_manifest.create_signature().unwrap();

        let mut manifest = Manifest::from_public_key(&base_path, pub_key).unwrap();

        // Compute checksum for original file
        manifest.compute_checksum(&temp_file.path()).unwrap();

        // Add checksum for an extra file
        let extra_file = create_temp_file_with_content("extra").unwrap();
        manifest.compute_checksum(&extra_file.path()).unwrap();

        assert!(manifest.verify(&mut paths, &ref_manifest).is_err());
    }

    #[test]
    fn test_wont_verify_without_signature() {
        let keypair = create_test_keypair();
        let pub_key = keypair.public_key().as_ref().to_vec();

        let temp_file = create_temp_file_with_content("test").unwrap();
        let base_path = temp_file.path().parent().unwrap();

        let mut ref_manifest = Manifest::from_signing_key(&base_path, keypair).unwrap();

        ref_manifest.compute_checksum(&temp_file.path()).unwrap();
        // Deliberately skip creating signature

        let mut manifest = Manifest::from_public_key(&base_path, pub_key).unwrap();
        manifest.compute_checksum(&temp_file.path()).unwrap();

        let mut paths = vec![temp_file.path().to_path_buf()];

        assert!(manifest.verify(&mut paths, &ref_manifest).is_err());
    }

    #[test]
    fn test_inner_folder_name_preserved() {
        let keypair = create_test_keypair();

        // Create a temporary directory with a nested file structure
        let temp_dir = tempfile::tempdir().unwrap();
        let inner_dir = temp_dir.path().join("inner");
        std::fs::create_dir(&inner_dir).unwrap();

        let test_file = inner_dir.join("test.txt");
        std::fs::write(&test_file, "test content").unwrap();

        let base_path = temp_dir.path();

        let mut manifest = Manifest::from_signing_key(&base_path, keypair).unwrap();

        manifest.compute_checksum(&test_file).unwrap();

        // Verify the checksum key preserves the inner folder name
        assert!(manifest.checksums.contains_key("inner/test.txt"));
    }
}
