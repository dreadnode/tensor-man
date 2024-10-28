import torch
import json
import os
import argparse

parser = argparse.ArgumentParser(description="Inspect PyTorch model files")
parser.add_argument("file", help="Path to PyTorch model file")
parser.add_argument(
    "--detailed", action="store_true", help="Show detailed tensor information"
)
parser.add_argument("--filter", help="Filter tensors by name pattern")

args = parser.parse_args()

file_path = os.path.abspath(args.file)
file_size = os.path.getsize(file_path)

model = torch.load(
    file_path, weights_only=True, mmap=True, map_location=torch.device("cpu")
)

model_metadata = model._metadata[""]
inspection = {
    "file_path": file_path,
    "file_type": "PyTorch",
    "file_size": file_size,
    "version": str(model_metadata["version"] if "version" in model_metadata else ""),
    "num_tensors": len(model.items()),
    "data_size": 0,
    "unique_shapes": [],
    "unique_dtypes": [],
    "metadata": {k: str(v) for (k, v) in model_metadata.items()},
    "tensors": [] if args.detailed else None,
}

for tensor_name, tensor in model.items():
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
                    k: str(v) for (k, v) in model._metadata[layer_name].items()
                }
                if layer_name in model._metadata
                else None,
            }
        )

inspection["header_size"] = inspection["file_size"] - inspection["data_size"]

print(json.dumps(inspection))
