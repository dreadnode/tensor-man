
<p align="center">
  <a href="https://github.com/dreadnode/tensor-man/releases/latest"><img alt="Release" src="https://img.shields.io/github/release/dreadnode/tensor-man.svg?style=fl_pathat-square"></a>
  <a href="https://crates.io/crates/tensor-man"><img alt="Crate" src="https://img.shields.io/crates/v/tensor-man.svg"></a>
  <a href="https://hub.docker.com/r/dreadnode/tensor-man"><img alt="Docker Hub" src="https://img.shields.io/docker/v/dreadnode/tensor-man?logo=docker"></a>
  <a href="https://rust-reportcard.xuri.me/report/github.com/dreadnode/tensor-man"><img alt="Rust Report" src="https://rust-reportcard.xuri.me/badge/github.com/dreadnode/tensor-man"></a>
  <a href="#"><img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/dreadnode/tensor-man/test.yml"></a>
  <a href="https://github.com/dreadnode/tensor-man/blob/master/LICENSE.md"><img alt="Software License" src="https://img.shields.io/badge/license-GPL3-brightgreen.svg?style=flat-square"></a>
</p>

`tensor-man` is a utility to inspect, validate, sign and verify machine learning model files.

## Supported Formats

* [safetensors](https://github.com/huggingface/safetensors)
* [ONNX](https://onnx.ai/)
* [GGUF](https://huggingface.co/docs/hub/gguf)
* [PyTorch](https://pytorch.org/)

> [!IMPORTANT]
> PyTorch models are loaded and inspected in a networkless Docker container in order to prevent [unintended code execution](https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models) on the host machine.

## Install with Cargo

This is the recommended way to install and use the tool:

```bash
cargo install tensor-man
```

## Pull from Docker Hub

```bash
docker pull dreadnode/tensor-man:latest
```

## Build Docker image

To build your own Docker image for the tool, run:

```bash
docker build . -t tman  
```

## Note about Docker

If you want to inspect PyTorch models and you are using `tensor-man` inside a container, make sure to share the docker socket from the host machine with the container:

```bash
docker run -it \
  # these paths must match
  -v/path/to/pytorch_model.bin:/path/to/pytorch_model.bin \
  # allow the container itself to instrument docker on the host
  -v/var/run/docker.sock:/var/run/docker.sock \
  # the rest of the command line
  tman inspect /path/to/pytorch_model.bin
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

# this will sign the entire model folder with every file in it
tman sign /path/to/whatever/Meta-Llama-3-8B/ -K /path/to/private.key
```
And the public one to verify the signature:

```bash
# will verify the signature in tinyyolov2-8.signature
tman verify /path/to/whatever/tinyyolov2-8.onnx -K /path/to/public.key

# will verify with an alternative signature file 
tman verify /path/to/whatever/tinyyolov2-8.onnx -K /path/to/public.key --signature /path/to/your.signature

# this will verify every file in the model folder
tman sign /path/to/whatever/Meta-Llama-3-8B/ -K /path/to/public.key
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
