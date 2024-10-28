`tensor-man` is a utility to inspect, validate, sign and verify machine learning model files.

## Supported Formats

* [safetensors](https://github.com/huggingface/safetensors)
* [ONNX](https://onnx.ai/)
* [GGUF](https://huggingface.co/docs/hub/gguf)
* [PyTorch](https://pytorch.org/)

> [!IMPORTANT]
> PyTorch models are loaded and inspected in a networkless Docker container in order to prevent [unintended code execution](https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models) on the host machine.

## Install with Cargo

```bash
cargo install tensor-man
```

## Build Docker image

To build the Docker image for the tool, run:

```bash
docker build . -t tman  
```

## Build from source

Alternatively you can build the project from source, in which case you'll need to have Rust and Cargo [installed on your system](https://rustup.rs/).

Once you have those set up, clone the repository and build the project:

```bash
cargo build --release
```

The compiled binary will be available in the `target/release` directory. You can run it directly or add it to your system's PATH:

```bash
# Run directly
./target/release/tman

# Or, copy to a directory in your PATH (e.g., /usr/local/bin)
sudo cp target/release/tman /usr/local/bin/
```

## Usage

### Inspect

Inspect a file and print a brief summary:

```bash
tman inspect /path/to/whatever/llama-3.1-8b-instruct.safetensors
```

Print detailed information about each tensor:

```bash
tman inspect /path/to/whatever/llama-3.1-8b-instruct.safetensors --detail full
```

Filter by tensor name:

```bash
tman inspect /path/to/whatever/llama-3.1-8b-instruct.onnx -D full --filter "q_proj"
```

Save the output as JSON:

```bash
tman inspect /path/to/whatever/llama-3.1-8b-instruct.gguf -D full --to-json output.json
```

### Sign and Verify

The tool allows you to generate an Ed25519 key pair to sign your models:

```bash
tman create-key --private-key private.key --public-key public.key
```

Then you can use the private key to sign a model (this will automatically include and sign external data files if referenced by the format):

```bash
# this will generate the tinyyolov2-8.signature file
tman sign /path/to/whatever/tinyyolov2-8.onnx -K /path/to/private.key

# you can provide a safetensors index file and all files referenced by it will be signed as well
tman sign /path/to/whatever/Meta-Llama-3-8B/model.safetensors.index.json -K /path/to/private.key
```
And the public one to verify the signature:

```bash
# will verify the signature in tinyyolov2-8.signature
tman verify /path/to/whatever/tinyyolov2-8.onnx -K /path/to/public.key

# will verify with an alternative signature file 
tman verify /path/to/whatever/tinyyolov2-8.onnx -K /path/to/public.key --signature /path/to/your.signature
```

### Inference Graph

Generate a .dot file for the execution graph of an ONNX model:

```bash
tman graph /path/to/whatever/tinyyolov2-8.onnx --output tinyyolov2-8.dot
```

### More

For the full list of commands and options, run:

```bash
tman --help

# get command specific help
tman inspect --help
```

## License

This tool is released under the GPL 3 license. To see the licenses of the project dependencies, install cargo license with `cargo install cargo-license` and then run `cargo license`.