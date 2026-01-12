#!/usr/bin/env python3
"""
Convert PyTorch checkpoint to safetensors format.

Simple conversion that preserves original PyTorch structure:
- Keeps all original key names (no remapping)
- Preserves tensor shapes (no transpositions)
- Includes all keys (preprocessor, LSTM biases, etc.)
- Optional float16 casting

Usage:
    # Basic conversion (keep original dtypes)
    python convert_ckpt.py checkpoint.ckpt output.safetensors

    # Convert float32 tensors to float16
    python convert_ckpt.py checkpoint.ckpt output.safetensors --dtype float16
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_checkpoint(path: str) -> dict[str, Any]:
    """
    Load PyTorch checkpoint and return state dict.

    Extracts the state_dict from NeMo/PyTorch checkpoints,
    skipping non-tensor data like optimizer states.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for checkpoint loading. "
            "Install with: pip install torch"
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"Path is not a file: {path}")

    print(f"Loading checkpoint from {path}...")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # Extract state_dict - NeMo checkpoints store it under 'state_dict' key
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise RuntimeError(
            f"Unexpected checkpoint format. Expected dict, got {type(checkpoint)}"
        )

    print(f"Loaded {len(state_dict)} keys from checkpoint")
    return state_dict


def convert_weights(
    state_dict: dict[str, Any],
    dtype: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Convert PyTorch state dict to numpy arrays for safetensors.

    Args:
        state_dict: PyTorch state dict with tensor values
        dtype: Optional dtype to cast float32 tensors to ('float16' or None)

    Returns:
        Dictionary mapping key names to numpy arrays
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for tensor conversion")

    result = {}
    skipped_count = 0
    converted_count = 0
    cast_count = 0

    for key, value in state_dict.items():
        # Skip non-tensor values (optimizer states, scalars, etc.)
        if not isinstance(value, (torch.Tensor, np.ndarray)):
            skipped_count += 1
            continue

        # Convert to numpy
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = value

        # Cast to float16 if requested
        if dtype == "float16" and arr.dtype == np.float32:
            arr = arr.astype(np.float16)
            cast_count += 1

        result[key] = arr
        converted_count += 1

    print(f"Converted {converted_count} tensors")
    if cast_count > 0:
        print(f"Cast {cast_count} tensors from float32 to float16")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} non-tensor values")

    return result


def save_safetensors(weights: dict[str, np.ndarray], output_path: str) -> None:
    """Save weights in safetensors format."""
    try:
        from safetensors.numpy import save_file
    except ImportError:
        raise ImportError(
            "safetensors is required for saving. Install with: pip install safetensors"
        )

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate total size
    total_bytes = sum(arr.nbytes for arr in weights.values())
    print(f"Saving {len(weights)} tensors ({total_bytes / 1024 / 1024:.2f} MB)")

    save_file(weights, output_path)

    file_size = os.path.getsize(output_path)
    print(f"Saved to {output_path} ({file_size / 1024 / 1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoint to safetensors format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion (keep original dtypes)
    python convert_ckpt.py checkpoint.ckpt output.safetensors
    
    # Convert float32 tensors to float16
    python convert_ckpt.py checkpoint.ckpt output.safetensors --dtype float16
        """,
    )

    parser.add_argument(
        "checkpoint_path", type=str, help="Path to the PyTorch checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "output_path", type=str, help="Path to save the safetensors file"
    )
    parser.add_argument(
        "--dtype",
        "-d",
        type=str,
        choices=["float16", "float32"],
        default=None,
        help="Cast float32 tensors to this dtype. Default: keep original",
    )

    args = parser.parse_args()

    # Don't cast if float32 is explicitly requested (it's already float32)
    dtype = args.dtype if args.dtype != "float32" else None

    try:
        state_dict = load_checkpoint(args.checkpoint_path)
        weights = convert_weights(state_dict, dtype=dtype)
        save_safetensors(weights, args.output_path)
        print("Conversion complete!")
    except (FileNotFoundError, RuntimeError, ImportError, IOError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
