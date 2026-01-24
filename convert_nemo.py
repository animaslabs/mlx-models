#!/usr/bin/env python3
"""
Convert Parakeet TDT 0.6B v3 PyTorch checkpoint to MLX-compatible safetensors format.

This script loads a NeMo/PyTorch checkpoint and converts the weights to safetensors
format matching the mlx-community format:
- Key names follow MLX conventions (encoder.layers, encoder.pre_encode)
- Conv weights are transposed to MLX layout (OHWI for Conv2d, OKI for Conv1d)
- LSTM biases are combined into single bias tensors
- Optional quantization to 4-bit or 8-bit using MLX's native quantization format

Quantization notes:
- Only nn.Linear layers are quantized (MLX only supports QuantizedLinear and QuantizedEmbedding)
- LSTM and Conv1d weights are NOT quantized (kept as float32)
- Scales/biases use float16 to save space

Usage:
    python scripts/convert_parakeet_weights.py <checkpoint_path> <output_path>
    python scripts/convert_parakeet_weights.py <checkpoint_path> <output_path> --quantize 8
    
Example:
    python scripts/convert_parakeet_weights.py \\
        models/parakeet-tdt-0.6b-v3/model_weights.ckpt \\
        models/parakeet-tdt-0.6b-v3/model_weights.safetensors
    
    # With 8-bit quantization (MLX native format)
    python scripts/convert_parakeet_weights.py \\
        models/parakeet-tdt-0.6b-v3/model_weights.ckpt \\
        models/parakeet-tdt-0.6b-v3-q8/model.safetensors \\
        --quantize 8
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np


# Quantization constants
QUANTIZE_GROUP_SIZE = 64  # Group size for block-wise quantization (MLX default)


def check_file_exists(path: str) -> None:
    """Check if a file exists and raise an error if not."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"Path is not a file: {path}")


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load PyTorch checkpoint and return state dict."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for checkpoint loading. "
            "Install with: pip install torch"
        )

    check_file_exists(path)

    try:
        print(f"Loading checkpoint...")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to parse checkpoint: {e}")

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


def remap_key_to_mlx(key: str) -> Optional[str]:
    """
    Remap a single PyTorch/NeMo key to MLX model structure.

    MLX format uses:
    - encoder.layers.N (not encoder.blocks.N)
    - encoder.pre_encode.conv.N (not encoder.subsampling.convN)
    - decoder.prediction.dec_rnn.lstm.N.Wx/Wh/bias (not weight_ih_lN etc)
    """
    # Skip non-model keys
    skip_prefixes = [
        "optimizer",
        "lr_scheduler",
        "global_step",
        "epoch",
        "pytorch-lightning",
    ]
    for prefix in skip_prefixes:
        if key.startswith(prefix):
            return None

    # Skip preprocessor (MLX computes mel filterbank at runtime)
    if key.startswith("preprocessor."):
        return None

    # Skip num_batches_tracked (not needed for inference)
    if "num_batches_tracked" in key:
        return None

    # Encoder subsampling: keep as pre_encode with dot notation
    # NeMo: encoder.pre_encode.conv.0.weight -> MLX: encoder.pre_encode.conv.0.weight
    if key.startswith("encoder.pre_encode."):
        # Already in correct format for MLX
        return key

    # Encoder layers: keep as layers (MLX uses encoder.layers.N)
    if key.startswith("encoder.layers."):
        return key

    # LSTM keys are handled specially in convert_lstm_weights
    # Skip individual LSTM weight/bias keys here - they'll be combined
    if "dec_rnn.lstm.weight_" in key or "dec_rnn.lstm.bias_" in key:
        return None  # Handled by convert_lstm_weights

    # Other decoder/joint keys pass through
    if (
        key.startswith("decoder.")
        or key.startswith("joint.")
        or key.startswith("encoder.")
    ):
        return key

    print(f"WARNING: Unhandled key '{key}' - keeping as-is", file=sys.stderr)
    return key


def transpose_conv2d(weight: np.ndarray) -> np.ndarray:
    """
    Transpose Conv2d weights from PyTorch to MLX format.
    PyTorch: OIHW [out_channels, in_channels, height, width]
    MLX:     OHWI [out_channels, height, width, in_channels]
    """
    return np.transpose(weight, (0, 2, 3, 1))


def transpose_conv1d(weight: np.ndarray) -> np.ndarray:
    """
    Transpose Conv1d weights from PyTorch to MLX format.
    PyTorch: OIK [out_channels, in_channels, kernel_size]
    MLX:     OKI [out_channels, kernel_size, in_channels]
    """
    return np.transpose(weight, (0, 2, 1))


def is_conv2d_weight(key: str, shape: Tuple[int, ...]) -> bool:
    """Check if a weight tensor is a Conv2d weight based on key and shape."""
    # Conv2d weights have 4 dimensions and are in pre_encode (subsampling)
    if len(shape) != 4:
        return False
    if "pre_encode.conv" in key and ".weight" in key:
        return True
    return False


def is_conv1d_weight(key: str, shape: Tuple[int, ...]) -> bool:
    """Check if a weight tensor is a Conv1d weight based on key and shape."""
    # Conv1d weights have 3 dimensions
    if len(shape) != 3:
        return False
    # Conformer conv module weights and decoder Conv1d layers
    if any(x in key for x in ["pointwise_conv", "depthwise_conv", "decoder_layers"]):
        return ".weight" in key
    return False


def is_pointwise_conv1d(key: str, shape: Tuple[int, ...]) -> bool:
    """
    Check if a weight tensor is a pointwise (kernel_size=1) Conv1d.
    These can be reshaped to 2D and quantized as linear layers.

    PyTorch shape: (out_channels, in_channels, 1)
    """
    if len(shape) != 3:
        return False
    if "pointwise_conv" not in key:
        return False
    if ".weight" not in key:
        return False
    # Kernel size is the last dimension in PyTorch format
    return shape[2] == 1


def should_quantize_tensor(
    key: str, shape: Tuple[int, ...], quantize_convs: bool = False
) -> bool:
    """
    Determine if a tensor should be quantized using MLX native quantization.

    Only 2D tensors (Linear weights) are quantized. Skip:
    - Small tensors (< 1024 elements)
    - Biases (1D tensors)
    - Layer norms and batch norms
    - Conv weights (3D/4D tensors) - unless quantize_convs is True for pointwise
    - Tensors where input_dim < group_size (can't be quantized)
    """
    # Only quantize 2D tensors (Linear weights)
    if len(shape) != 2:
        return False

    # Skip small tensors
    total_elements = np.prod(shape)
    if total_elements < 1024:
        return False

    # Skip tensors where input dimension is smaller than group_size
    # MLX quantization requires input_dim >= group_size
    input_dim = shape[1]  # For Linear weights: [out_features, in_features]
    if input_dim < QUANTIZE_GROUP_SIZE:
        return False

    # Skip normalization layers and biases
    skip_patterns = [
        "layer_norm",
        "layernorm",
        "ln_",
        "_ln",
        "batch_norm",
        "batchnorm",
        "bn_",
        "_bn",
        ".bias",
        ".running_mean",
        ".running_var",
        "pos_bias",  # Skip positional biases (small)
    ]
    key_lower = key.lower()
    for pattern in skip_patterns:
        if pattern in key_lower:
            return False

    return True


def quantize_tensor_mlx(
    arr: np.ndarray, bits: int, group_size: int = QUANTIZE_GROUP_SIZE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize a tensor using MLX's native affine quantization format.

    MLX uses affine quantization: x_q = round((x - bias) / scale)
    Dequantization: x = x_q * scale + bias

    The quantized weights are packed into uint32 values.

    Args:
        arr: Input tensor (float32), shape [out_features, in_features]
        bits: Number of bits (4 or 8)
        group_size: Number of elements per quantization group

    Returns:
        Tuple of (quantized_weights, scales, biases)
        - quantized_weights: uint32 packed values, shape [out_features, in_features * bits / 32]
        - scales: float32 scale factors, shape [out_features, num_groups]
        - biases: float32 bias values, shape [out_features, num_groups]
    """
    try:
        import mlx.core as mx
    except ImportError:
        raise ImportError(
            "MLX is required for native quantization. Install with: pip install mlx"
        )

    # Convert to MLX array
    weight = mx.array(arr.astype(np.float32))

    # Use MLX's native quantize function
    quantized, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)

    # Convert back to numpy
    return (
        np.array(quantized),  # uint32
        np.array(scales),  # float32
        np.array(biases),  # float32
    )


def convert_lstm_weights(
    state_dict: Dict[str, Any],
    quantize_bits: Optional[int] = None,
    group_size: int = QUANTIZE_GROUP_SIZE,
) -> Dict[str, np.ndarray]:
    """
    Convert PyTorch LSTM weights to MLX format.

    PyTorch format:
        weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
        weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1

    MLX format:
        0.Wx, 0.Wh, 0.bias
        1.Wx, 1.Wh, 1.bias

    Note: LSTM weights are NOT quantized because MLX doesn't have QuantizedLSTM.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required")

    result = {}
    prefix = "decoder.prediction.dec_rnn.lstm"

    # Find all LSTM layers
    layer_indices = set()
    for key in state_dict.keys():
        match = re.search(r"dec_rnn\.lstm\.weight_ih_l(\d+)", key)
        if match:
            layer_indices.add(int(match.group(1)))

    for layer_idx in sorted(layer_indices):
        # Get PyTorch weights
        weight_ih = state_dict.get(f"{prefix}.weight_ih_l{layer_idx}")
        weight_hh = state_dict.get(f"{prefix}.weight_hh_l{layer_idx}")
        bias_ih = state_dict.get(f"{prefix}.bias_ih_l{layer_idx}")
        bias_hh = state_dict.get(f"{prefix}.bias_hh_l{layer_idx}")

        if weight_ih is None or weight_hh is None:
            print(f"WARNING: Missing LSTM weights for layer {layer_idx}")
            continue

        # Convert to numpy
        if isinstance(weight_ih, torch.Tensor):
            weight_ih = weight_ih.detach().cpu().numpy()
        if isinstance(weight_hh, torch.Tensor):
            weight_hh = weight_hh.detach().cpu().numpy()

        # Store as float32 (MLX doesn't support quantized LSTM)
        result[f"{prefix}.{layer_idx}.Wx"] = weight_ih
        result[f"{prefix}.{layer_idx}.Wh"] = weight_hh

        # Combine biases (PyTorch has separate ih and hh biases)
        if bias_ih is not None and bias_hh is not None:
            if isinstance(bias_ih, torch.Tensor):
                bias_ih = bias_ih.detach().cpu().numpy()
            if isinstance(bias_hh, torch.Tensor):
                bias_hh = bias_hh.detach().cpu().numpy()
            combined_bias = bias_ih + bias_hh
            result[f"{prefix}.{layer_idx}.bias"] = combined_bias
        elif bias_ih is not None:
            if isinstance(bias_ih, torch.Tensor):
                bias_ih = bias_ih.detach().cpu().numpy()
            result[f"{prefix}.{layer_idx}.bias"] = bias_ih

    return result


def convert_weights_to_mlx(
    state_dict: Dict[str, Any],
    quantize_bits: Optional[int] = None,
    group_size: int = QUANTIZE_GROUP_SIZE,
) -> Dict[str, np.ndarray]:
    """
    Convert all weights to MLX format with proper transpositions.

    Args:
        state_dict: PyTorch state dict
        quantize_bits: If set, quantize weights to this bit width (4 or 8)
        group_size: Group size for quantization (default: 64)
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for tensor conversion")

    result = {}
    conv2d_count = 0
    conv1d_count = 0
    skipped_count = 0
    quantized_count = 0

    # First, handle LSTM weights specially (with optional quantization)
    lstm_weights = convert_lstm_weights(state_dict, quantize_bits, group_size)
    result.update(lstm_weights)
    print(
        f"Converted {len([k for k in lstm_weights.keys() if not k.endswith('.scales') and not k.endswith('.biases')])} LSTM tensors"
    )

    # Process remaining weights
    for key, value in state_dict.items():
        # Remap key to MLX format
        new_key = remap_key_to_mlx(key)
        if new_key is None:
            skipped_count += 1
            continue

        # Convert tensor to numpy
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            skipped_count += 1
            continue

        # Convert float16 to float32
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)

        shape = arr.shape

        # Transpose Conv2d weights: OIHW -> OHWI
        if is_conv2d_weight(key, shape):
            arr = transpose_conv2d(arr)
            conv2d_count += 1

        # Handle Conv1d weights (including pointwise)
        # Note: Conv1d weights are NOT quantized because MLX doesn't have QuantizedConv1d
        elif is_pointwise_conv1d(key, shape) or is_conv1d_weight(key, shape):
            # Transpose to MLX format: OIK -> OKI
            arr = transpose_conv1d(arr)
            conv1d_count += 1

        # Apply MLX native quantization if requested
        if quantize_bits is not None and should_quantize_tensor(new_key, arr.shape):
            quantized, scales, biases = quantize_tensor_mlx(
                arr, quantize_bits, group_size
            )
            result[new_key] = quantized
            # MLX expects scales/biases as siblings of weight, not children
            # e.g., linear.weight, linear.scales, linear.biases (not linear.weight.scales)
            base_key = (
                new_key.rsplit(".", 1)[0] if new_key.endswith(".weight") else new_key
            )
            result[f"{base_key}.scales"] = scales.astype(np.float16)
            result[f"{base_key}.biases"] = biases.astype(np.float16)
            quantized_count += 1
        else:
            # For non-quantized small tensors in quantized models, use float16
            if quantize_bits is not None and arr.dtype == np.float32:
                # Keep biases, running stats, and small tensors as float16
                if (
                    arr.ndim == 1
                    or np.prod(arr.shape) < 10000
                    or "running_" in new_key
                    or "pos_bias" in new_key
                ):
                    arr = arr.astype(np.float16)
            result[new_key] = arr

    print(f"Transposed {conv2d_count} Conv2d weights (OIHW -> OHWI)")
    print(f"Transposed {conv1d_count} Conv1d weights (OIK -> OKI)")
    print(f"Skipped {skipped_count} keys (optimizer states, preprocessor, etc.)")
    if quantize_bits is not None:
        print(
            f"Quantized {quantized_count} Linear layer weights to {quantize_bits}-bit"
        )

    return result


def save_safetensors(weights: Dict[str, np.ndarray], output_path: str) -> None:
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


def main(
    checkpoint_path: str,
    output_path: str,
    quantize_bits: Optional[int] = None,
    group_size: int = QUANTIZE_GROUP_SIZE,
) -> None:
    """Main conversion function."""
    quant_str = (
        f" with {quantize_bits}-bit quantization (group_size={group_size})"
        if quantize_bits
        else ""
    )
    print(f"Converting {checkpoint_path} to MLX format{quant_str}...")

    state_dict = load_checkpoint(checkpoint_path)

    print("Converting weights to MLX format...")
    mlx_weights = convert_weights_to_mlx(state_dict, quantize_bits, group_size)
    print(f"Total: {len(mlx_weights)} tensors")

    # Show key prefixes for verification
    prefixes = set()
    for key in mlx_weights.keys():
        parts = key.split(".")
        if len(parts) >= 2:
            prefixes.add(f"{parts[0]}.{parts[1]}")
    print(f"\nKey prefixes: {sorted(prefixes)}")

    # Show quantization info
    if quantize_bits:
        quant_keys = [k for k in mlx_weights.keys() if k.endswith(".scales")]
        print(f"\nQuantized {len(quant_keys)} weight tensors")

        # Show breakdown by type
        linear_quant = [
            k for k in quant_keys if "linear" in k.lower() or "joint" in k.lower()
        ]
        conv_quant = [k for k in quant_keys if "pointwise" in k.lower()]
        lstm_quant = [k for k in quant_keys if "lstm" in k.lower()]
        embed_quant = [k for k in quant_keys if "embed" in k.lower()]
        other_quant = [
            k
            for k in quant_keys
            if k not in linear_quant + conv_quant + lstm_quant + embed_quant
        ]

        print(f"  Linear layers: {len(linear_quant)}")
        print(f"  Pointwise convs: {len(conv_quant)}")
        print(f"  LSTM weights: {len(lstm_quant)}")
        print(f"  Embeddings: {len(embed_quant)}")
        print(f"  Other: {len(other_quant)}")

        # Show dtype breakdown
        dtype_sizes = {}
        for key, arr in mlx_weights.items():
            dtype = str(arr.dtype)
            dtype_sizes[dtype] = dtype_sizes.get(dtype, 0) + arr.nbytes

        print(f"\nSize by dtype:")
        for dtype, size in sorted(dtype_sizes.items(), key=lambda x: -x[1]):
            print(f"  {dtype}: {size / 1024 / 1024:.2f} MB")

    save_safetensors(mlx_weights, output_path)
    print("Conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Parakeet PyTorch checkpoint to MLX safetensors format"
    )
    parser.add_argument(
        "checkpoint_path", type=str, help="Path to the PyTorch checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "output_path", type=str, help="Path to save the safetensors file"
    )
    parser.add_argument(
        "--quantize",
        "-q",
        type=int,
        choices=[4, 8],
        default=None,
        help="Quantize weights to specified bit width (4 or 8). Default: no quantization",
    )
    parser.add_argument(
        "--group-size",
        "-g",
        type=int,
        default=64,
        help="Group size for block-wise quantization (default: 64)",
    )

    args = parser.parse_args()

    try:
        main(args.checkpoint_path, args.output_path, args.quantize, args.group_size)
    except (FileNotFoundError, RuntimeError, ImportError, IOError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
