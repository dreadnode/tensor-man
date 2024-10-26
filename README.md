`tensor-man` is an utility to inspect and validate [safetensors](https://github.com/huggingface/safetensors), [ONNX](https://onnx.ai/) and [GGUF](https://huggingface.co/docs/hub/gguf) files.

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

Generate a .dot file for the execution graph of an ONNX model:

```bash
tman graph /path/to/whatever/tinyyolov2-8.onnx --output tinyyolov2-8.dot
```

For the full list of commands and options, run:

```bash
tman --help

# get command specific help
tman inspect --help
```

## License

This tool is released under the GPL 3 license. To see the licenses of the project dependencies, install cargo license with `cargo install cargo-license` and then run `cargo license`.