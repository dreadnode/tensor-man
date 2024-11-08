import torch
import json
import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Inspect PyTorch model files")
    parser.add_argument("file", help="Path to PyTorch model file")
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed tensor information"
    )
    parser.add_argument("--filter", help="Filter tensors by name pattern")

    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    file_size = os.path.getsize(file_path)

    try:
        model = torch.load(
            file_path, weights_only=True, mmap=True, map_location=torch.device("cpu")
        )
    except RuntimeError:
        # handle: RuntimeError: mmap can only be used with files saved with `torch.save(/model.bin, _use_new_zipfile_serialization=True),
        # please torch.save your checkpoint with this option in order to use mmap.
        model = torch.load(
            file_path, weights_only=True, map_location=torch.device("cpu")
        )

    all_metadata = getattr(model, "_metadata") if hasattr(model, "_metadata") else {}
    model_metadata = all_metadata[""] if "" in all_metadata else {}

    inspection = {
        "file_path": file_path,
        "file_type": "PyTorch",
        "file_size": file_size,
        "version": str(
            model_metadata["version"] if "version" in model_metadata else ""
        ),
        "num_tensors": len(model.items()),
        "data_size": 0,
        "unique_shapes": [],
        "unique_dtypes": [],
        "metadata": {k: str(v) for (k, v) in model_metadata.items()},
        "tensors": [] if args.detailed else None,
    }

    # handle nested dictionary case
    if "model" in model:
        model = model["model"]

    for tensor_name, tensor in model.items():
        # make sure it's a tensor
        if not isinstance(tensor, torch.Tensor):
            try:
                tensor = torch.tensor(tensor)
            except:
                continue

        inspection["data_size"] += tensor.shape.numel() * tensor.element_size()

        shape = list(tensor.shape)
        if shape != []:
            if shape not in inspection["unique_shapes"]:
                inspection["unique_shapes"].append(shape)

        dtype = str(tensor.dtype).replace("torch.", "")
        if dtype not in inspection["unique_dtypes"]:
            inspection["unique_dtypes"].append(dtype)

        if args.detailed:
            if args.filter and args.filter not in tensor_name:
                continue

            layer_name = tensor_name.split(".")[0]
            inspection["tensors"].append(
                {
                    "id": tensor_name,
                    "shape": shape,
                    "dtype": dtype,
                    "size": tensor.shape.numel() * tensor.element_size(),
                    "metadata": {
                        k: str(v) for (k, v) in all_metadata[layer_name].items()
                    }
                    if layer_name in all_metadata
                    else {},
                }
            )

    # data can be compressed or shared among multiple vectors(?) in which case this would be negative
    inspection["header_size"] = (
        inspection["file_size"] - inspection["data_size"]
        if inspection["data_size"] < inspection["file_size"]
        else 0
    )

    print(json.dumps(inspection))


if __name__ == "__main__":
    main()
