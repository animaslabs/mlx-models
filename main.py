#!/usr/bin/env python3
"""
Main script for converting NeMo models to safetensors format.

Workflow:
1. Download a model from Hugging Face Hub
2. Unpack the NeMo file (tar archive)
3. Convert checkpoint to safetensors (MLX or generic PyTorch format)
4. Optionally upload converted model(s) to Hugging Face Hub

Usage:
    # MLX format (default) - full precision
    python main.py nvidia/parakeet-tdt-0.6b-v3 --output-repo username/model-mlx

    # MLX format with 8-bit quantization
    python main.py nvidia/parakeet-tdt-0.6b-v3 --quantize 8 --output-repo username/model-mlx-8bit

    # MLX format with multiple quantizations
    python main.py nvidia/parakeet-tdt-0.6b-v3 --quantize 4 8 --output-repo username/model-mlx

    # Generic PyTorch safetensors (no MLX transformations)
    python main.py nvidia/parakeet-tdt-0.6b-v3 --format pytorch --output-repo username/model

    # Generic PyTorch safetensors with float16
    python main.py nvidia/parakeet-tdt-0.6b-v3 --format pytorch --dtype float16 --output-repo username/model

    # Without upload
    python main.py nvidia/parakeet-tdt-0.6b-v3 --quantize 8 --no-upload
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

from hf import (
    download_model,
    fetch_model_metadata,
    fetch_source_info,
    list_existing_repos,
    unpack_nemo,
    upload_model,
)
from model_card import detect_family, write_model_card
from convert_nemo import (
    load_checkpoint,
    convert_weights_to_mlx,
    save_safetensors,
    QUANTIZE_GROUP_SIZE,
)
from convert_ckpt import convert_weights as convert_weights_pytorch


def find_nemo_file(model_dir: str) -> Optional[Path]:
    """Find the .nemo file in the downloaded model directory."""
    model_path = Path(model_dir)
    nemo_files = list(model_path.glob("*.nemo"))

    if not nemo_files:
        # Check subdirectories
        nemo_files = list(model_path.glob("**/*.nemo"))

    if not nemo_files:
        return None

    if len(nemo_files) > 1:
        print(f"Found multiple .nemo files, using first: {nemo_files[0]}")

    return nemo_files[0]


def find_checkpoint_file(unpacked_dir: str) -> Optional[Path]:
    """Find the checkpoint file in the unpacked NeMo directory."""
    unpacked_path = Path(unpacked_dir)

    # Look for common checkpoint patterns
    patterns = ["*.ckpt", "model_weights.ckpt", "**/*.ckpt"]

    for pattern in patterns:
        ckpt_files = list(unpacked_path.glob(pattern))
        if ckpt_files:
            return ckpt_files[0]

    return None


def copy_tokenizer_files(src_dir: str, dst_dir: str) -> None:
    """Copy tokenizer and config files to the output directory."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    # Files to copy
    patterns = [
        "*.vocab",
        "*.model",
        "vocab.txt",
        "*vocab*.txt",
        "model_config.yaml",
        "config.yaml",
        "tokenizer*",
    ]

    copied = []
    for pattern in patterns:
        for src_file in src_path.glob(pattern):
            if src_file.is_file():
                # Clean up filenames (remove hash prefixes)
                dst_name = src_file.name
                # Remove hash prefix like "0ee587b5d4b94f48993dcccf9868ea77_"
                if "_" in dst_name and len(dst_name.split("_")[0]) == 32:
                    dst_name = "_".join(dst_name.split("_")[1:])

                dst_file = dst_path / dst_name
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                    copied.append(dst_name)

    if copied:
        print(f"Copied tokenizer files: {', '.join(copied)}")


def _sanitize_for_json(obj):
    """Recursively replace inf/-inf/nan with None for JSON compatibility."""
    import math

    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def nemotron_asr_config(nemo_config: dict) -> dict:
    """Build a config.json in mlx-audio's native nemotron_asr layout.

    Mirrors mlx_audio.stt.models.nemotron_asr.convert._build_config, with the
    overrides mlx-audio's transformers-config path uses for this model: no
    language prompt, constant STFT padding, and the model's own attention context.
    """
    pre = nemo_config["preprocessor"]
    enc = nemo_config["encoder"]
    dec = nemo_config["decoder"]
    joint = nemo_config["joint"]
    att_context_size = [list(c) for c in enc["att_context_size"]]
    encoder_keys = [
        "feat_in", "n_layers", "d_model", "n_heads", "ff_expansion_factor",
        "subsampling_factor", "subsampling_conv_channels", "conv_kernel_size",
        "causal_downsampling", "conv_context_size", "conv_norm_type",
        "self_attention_model", "att_context_style", "pos_emb_max_len",
        "use_bias", "xscaling",
    ]
    return {
        "model_type": "nemotron_asr",
        "target": nemo_config.get("target", ""),
        "preprocessor": {
            "sample_rate": pre.get("sample_rate", 16000),
            "features": pre.get("features", 128),
            "n_fft": pre.get("n_fft", 512),
            "window_size": pre.get("window_size", 0.025),
            "window_stride": pre.get("window_stride", 0.01),
            "window": pre.get("window", "hann"),
            "preemph": pre.get("preemph") or 0.97,
            "dither": pre.get("dither", 1.0e-05),
            "normalize": str(pre.get("normalize", "NA")),
            "log_zero_guard_value": 2.0**-24,
            "pad_to": pre.get("pad_to", 0),
            "pad_value": pre.get("pad_value", 0.0),
            "pad_mode": "constant",
        },
        "encoder": {k: enc[k] for k in encoder_keys if k in enc}
        | {"att_context_size": att_context_size},
        "prompt": {"num_prompts": 0, "prompt_hidden": 0, "prompt_dictionary": {}},
        "decoder": {
            "pred_hidden": dec["prednet"]["pred_hidden"],
            "pred_rnn_layers": dec["prednet"]["pred_rnn_layers"],
            "vocab_size": dec["vocab_size"],
            "blank_as_pad": dec.get("blank_as_pad", True),
        },
        "joint": {
            "joint_hidden": joint["jointnet"]["joint_hidden"],
            "activation": joint["jointnet"].get("activation", "relu"),
            "encoder_hidden": joint["jointnet"]["encoder_hidden"],
            "pred_hidden": joint["jointnet"]["pred_hidden"],
            "num_classes": joint["num_classes"],
        },
        "vocabulary": list(joint["vocabulary"]),
        "default_language": "en",
        "default_att_context_size": att_context_size[0],
        "max_symbols": nemo_config.get("decoding", {}).get("greedy", {}).get("max_symbols", 10),
    }


def create_config_json(
    output_dir: str,
    unpacked_dir: str,
    quantize_bits: Optional[int] = None,
    group_size: int = QUANTIZE_GROUP_SIZE,
) -> None:
    """Create a config.json file for the MLX model from the NeMo config.

    Reads the model_config.yaml from the unpacked NeMo directory and converts
    it to config.json format expected by inference libraries like parakeet-mlx.
    Cache-aware streaming models get mlx-audio's nemotron_asr layout instead,
    since parakeet-mlx can't run them.
    """
    import json
    import yaml
    from typing import Any

    config: dict[str, Any] = {}

    # Try to load config from model_config.yaml (NeMo format)
    yaml_path = Path(unpacked_dir) / "model_config.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            nemo_config: dict[str, Any] = yaml.safe_load(f)

    if yaml_path.exists() and detect_family(nemo_config) == "cache-aware-rnnt":
        config = nemotron_asr_config(nemo_config)
        print(f"Created config.json in mlx-audio's nemotron_asr layout")
    elif yaml_path.exists():
        # Copy relevant config sections (same structure as mlx-community models)
        config = nemo_config.copy()

        # Clean up tokenizer paths to use local files instead of nemo: prefix
        if "tokenizer" in config:
            tokenizer: dict[str, Any] = config["tokenizer"]
            if isinstance(tokenizer.get("model_path"), str):
                tokenizer["model_path"] = "nemo:tokenizer.model"
            if isinstance(tokenizer.get("vocab_path"), str):
                tokenizer["vocab_path"] = "nemo:vocab.txt"
            if isinstance(tokenizer.get("spe_tokenizer_vocab"), str):
                tokenizer["spe_tokenizer_vocab"] = "nemo:tokenizer.vocab"

        print(f"Created config.json from NeMo config")
    else:
        # Fallback to minimal config if no YAML found
        config = {
            "model_type": "parakeet-tdt",
            "framework": "mlx",
        }
        print(f"Created minimal config.json (no NeMo config found)")

    if quantize_bits:
        config["quantization"] = {
            "bits": quantize_bits,
            "group_size": group_size,
        }

    config_path = Path(output_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(_sanitize_for_json(config), f, indent=2)


def get_output_suffix(
    format: str,
    quantize_bits: Optional[int] = None,
    dtype: Optional[str] = None,
    quantize_pointwise: bool = False,
) -> str:
    """Get the output directory suffix based on format and quantization.

    Quantized variants with pointwise convs quantized as linear layers get a
    -pw suffix: smaller, but only runtimes that build those convs as linear
    layers can load them.
    """
    if format == "pytorch":
        if dtype == "float16":
            return "-fp16"
        return ""
    else:  # mlx format
        if quantize_bits is None:
            return "-mlx"
        return f"-mlx-{quantize_bits}bit" + ("-pw" if quantize_pointwise else "")


def convert_model(
    repo_id: str,
    output_repo: Optional[str] = None,
    quantize_bits: list[Optional[int]] | None = None,
    upload: bool = True,
    local_dir: Optional[str] = None,
    private: bool = False,
    group_size: int = QUANTIZE_GROUP_SIZE,
    format: str = "mlx",
    dtype: Optional[str] = None,
    pointwise: str = "unquantized",
) -> list[str]:
    """
    Main conversion pipeline.

    Args:
        repo_id: Source HuggingFace repository ID
        output_repo: Base output repository ID (suffixes added for quantization)
        quantize_bits: List of quantization bit widths (None, 4, 8) - MLX format only
        upload: Whether to upload to HuggingFace
        local_dir: Local directory for downloads
        private: Whether to create private repos
        group_size: Quantization group size (MLX format only)
        format: Output format ('mlx' or 'pytorch')
        dtype: Target dtype for pytorch format ('float16' or None)
        pointwise: For quantized MLX variants, 'unquantized' leaves pointwise convs
            in float, 'quantized' quantizes them as linear layers (-pw repos), 'both'
            builds each

    Returns:
        List of output directory paths
    """
    # Step 1: Download the model
    print(f"\n{'=' * 60}")
    print(f"Step 1: Downloading {repo_id}")
    print(f"{'=' * 60}")

    if local_dir is None:
        local_dir = f"models/{repo_id}"

    model_dir = download_model(repo_id, local_dir)

    # Step 2: Find and unpack NeMo file
    print(f"\n{'=' * 60}")
    print(f"Step 2: Unpacking NeMo file")
    print(f"{'=' * 60}")

    nemo_file = find_nemo_file(model_dir)
    if nemo_file is None:
        raise FileNotFoundError(f"No .nemo file found in {model_dir}")

    print(f"Found NeMo file: {nemo_file}")
    unpacked_dir = unpack_nemo(str(nemo_file))

    # Step 3: Find checkpoint
    ckpt_file = find_checkpoint_file(unpacked_dir)
    if ckpt_file is None:
        raise FileNotFoundError(f"No checkpoint file found in {unpacked_dir}")

    print(f"Found checkpoint: {ckpt_file}")

    # Load checkpoint once (reuse for all conversions)
    print(f"\nLoading checkpoint...")
    state_dict = load_checkpoint(str(ckpt_file))

    # Fetch source model metadata once (reuse for all conversions)
    print(f"\nFetching model metadata from {repo_id}...")
    metadata = fetch_model_metadata(repo_id)
    source_revision, nemo_sha256 = fetch_source_info(repo_id, nemo_file.name)

    # Step 4: Convert based on format
    output_dirs = []

    if format == "pytorch":
        # Generic PyTorch safetensors conversion (single output)
        suffix = get_output_suffix(format, dtype=dtype)
        dtype_str = f"float16" if dtype == "float16" else "original dtype"

        print(f"\n{'=' * 60}")
        print(f"Step 3: Converting to PyTorch safetensors ({dtype_str})")
        print(f"{'=' * 60}")

        # Determine output directory
        repo_name = repo_id.split("/")[-1]
        output_dir = f"models/{repo_name}{suffix}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Convert weights (generic, no MLX transformations)
        print(f"Converting weights...")
        weights = convert_weights_pytorch(state_dict, dtype=dtype)

        # Save safetensors
        output_file = Path(output_dir) / "model.safetensors"
        save_safetensors(weights, str(output_file))

        # Copy tokenizer files
        copy_tokenizer_files(unpacked_dir, output_dir)

        # Generate model card README.md
        target_repo = (
            f"{output_repo}{suffix}" if output_repo else f"{repo_name}{suffix}"
        )
        write_model_card(
            output_dir,
            repo_id,
            target_repo,
            metadata,
            fmt="pytorch",
            source_revision=source_revision,
            nemo_sha256=nemo_sha256,
        )

        # No config.json for pytorch format (MLX-specific)

        output_dirs.append(output_dir)

        # Upload if requested
        if upload and output_repo:
            print(f"\n{'=' * 60}")
            print(f"Step 4: Uploading to HuggingFace")
            print(f"{'=' * 60}")

            target_repo = f"{output_repo}{suffix}"
            upload_model(output_dir, target_repo, private=private)

    else:
        # MLX format conversion (possibly multiple outputs for different quantizations)
        # Default quantization: just full precision
        if quantize_bits is None:
            quantize_bits = [None]

        # parakeet-mlx and mlx-audio build pointwise convs as nn.Conv1d, so only
        # the -pw variants quantize them as linear layers
        pointwise_options = {"unquantized": [False], "quantized": [True], "both": [False, True]}[pointwise]
        builds = [(b, pw) for b in quantize_bits for pw in (pointwise_options if b else [False])]

        repo_base = output_repo or repo_id.split("/")[-1]

        def repo_for(bits, pw):
            return f"{repo_base}{get_output_suffix(format, quantize_bits=bits, quantize_pointwise=pw)}"

        building = [repo_for(b, pw) for b, pw in builds]
        candidates = [repo_for(b, pw) for b, pw in [(None, False), (8, False), (4, False), (8, True), (4, True)]]
        existing = list_existing_repos([c for c in candidates if c not in building]) if output_repo else []
        variants = [c for c in candidates if c in building or c in existing]

        for bits, quantize_pointwise in builds:
            suffix = get_output_suffix(format, quantize_bits=bits, quantize_pointwise=quantize_pointwise)
            quant_str = f"{bits}-bit quantized" if bits else "full precision"
            if quantize_pointwise:
                quant_str += ", pointwise convs quantized"

            print(f"\n{'=' * 60}")
            print(f"Step 3: Converting to MLX format ({quant_str})")
            print(f"{'=' * 60}")

            # Determine output directory
            repo_name = repo_id.split("/")[-1]
            output_dir = f"models/{repo_name}{suffix}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Convert weights (MLX transformations + optional quantization)
            print(f"Converting weights...")
            mlx_weights = convert_weights_to_mlx(
                state_dict, bits, group_size, quantize_pointwise
            )

            # Save safetensors
            output_file = Path(output_dir) / "model.safetensors"
            save_safetensors(mlx_weights, str(output_file))

            # Copy tokenizer files
            copy_tokenizer_files(unpacked_dir, output_dir)

            # Generate model card README.md
            target_repo = (
                f"{output_repo}{suffix}" if output_repo else f"{repo_name}{suffix}"
            )
            write_model_card(
                output_dir,
                repo_id,
                target_repo,
                metadata,
                bits=bits,
                group_size=group_size,
                variants=variants,
                source_revision=source_revision,
                nemo_sha256=nemo_sha256,
            )

            # Create config (MLX-specific)
            create_config_json(output_dir, unpacked_dir, bits, group_size)

            output_dirs.append(output_dir)

            # Upload if requested
            if upload and output_repo:
                print(f"\n{'=' * 60}")
                print(f"Step 4: Uploading to HuggingFace")
                print(f"{'=' * 60}")

                target_repo = f"{output_repo}{suffix}"
                upload_model(output_dir, target_repo, private=private)

    return output_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeMo models to safetensors format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # MLX format (default) - full precision
    python main.py nvidia/parakeet-tdt-0.6b-v3 --output-repo username/parakeet-mlx
    
    # MLX format with 8-bit quantization
    python main.py nvidia/parakeet-tdt-0.6b-v3 -q 8 --output-repo username/parakeet-mlx
    
    # MLX format with multiple quantizations (full, 4-bit, 8-bit)
    python main.py nvidia/parakeet-tdt-0.6b-v3 -q 4 8 --output-repo username/parakeet-mlx
    
    # Generic PyTorch safetensors (no MLX transformations)
    python main.py nvidia/parakeet-tdt-0.6b-v3 --format pytorch --output-repo username/model
    
    # Generic PyTorch safetensors with float16
    python main.py nvidia/parakeet-tdt-0.6b-v3 --format pytorch --dtype float16 --output-repo username/model
    
    # Local conversion only (no upload)
    python main.py nvidia/parakeet-tdt-0.6b-v3 -q 8 --no-upload
        """,
    )

    parser.add_argument(
        "repo_id",
        help="Source HuggingFace repository ID (e.g., nvidia/parakeet-tdt-0.6b-v3)",
    )
    parser.add_argument(
        "--output-repo",
        "-o",
        help="Base output repository ID (suffixes added for quantization)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["mlx", "pytorch"],
        default="mlx",
        help="Output format: 'mlx' (with transpositions/quantization) or 'pytorch' (raw safetensors)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default=None,
        help="Cast tensors to dtype (pytorch format only). Default: keep original",
    )
    parser.add_argument(
        "--quantize",
        "-q",
        type=int,
        nargs="+",
        choices=[4, 8],
        default=None,
        help="Quantization bit widths (4 and/or 8). MLX format only. Omit for full precision.",
    )
    parser.add_argument(
        "--include-full-precision",
        action="store_true",
        help="Include full precision model when quantizing (MLX format only)",
    )
    parser.add_argument(
        "--no-upload", action="store_true", help="Skip uploading to HuggingFace"
    )
    parser.add_argument(
        "--local-dir", "-d", help="Local directory for downloaded model"
    )
    parser.add_argument(
        "--private", "-p", action="store_true", help="Create private repositories"
    )
    parser.add_argument(
        "--group-size",
        "-g",
        type=int,
        default=QUANTIZE_GROUP_SIZE,
        help=f"Quantization group size (default: {QUANTIZE_GROUP_SIZE})",
    )
    parser.add_argument(
        "--pointwise",
        choices=["unquantized", "quantized", "both"],
        default="unquantized",
        help="Pointwise convs in quantized variants: 'quantized' stores them as quantized "
        "linear layers in -pw repos (smaller, but parakeet-mlx and mlx-audio can't load "
        "them), 'both' builds each. Default: unquantized",
    )

    args = parser.parse_args()

    # Validate format-specific options
    quantize_bits: list[Optional[int]] | None = None
    dtype: str | None = None

    if args.format == "pytorch":
        if args.quantize:
            print(
                "Warning: --quantize is ignored for pytorch format (no quantization support)",
                file=sys.stderr,
            )
        if args.include_full_precision:
            print(
                "Warning: --include-full-precision is ignored for pytorch format",
                file=sys.stderr,
            )
        # Don't cast if float32 is explicitly requested (it's already float32)
        dtype = args.dtype if args.dtype != "float32" else None
    else:  # mlx format
        if args.dtype:
            print(
                "Warning: --dtype is ignored for mlx format (use --quantize instead)",
                file=sys.stderr,
            )
        # Build quantization list for MLX format
        quantize_bits = []
        if args.include_full_precision or args.quantize is None:
            quantize_bits.append(None)
        if args.quantize:
            quantize_bits.extend(args.quantize)

    # Validate upload requirements
    upload = not args.no_upload
    if upload and not args.output_repo:
        print("Error: --output-repo is required for upload (or use --no-upload)")
        sys.exit(1)

    try:
        output_dirs = convert_model(
            repo_id=args.repo_id,
            output_repo=args.output_repo,
            quantize_bits=quantize_bits,
            upload=upload,
            local_dir=args.local_dir,
            private=args.private,
            group_size=args.group_size,
            format=args.format,
            dtype=dtype,
            pointwise=args.pointwise,
        )

        print(f"\n{'=' * 60}")
        print("Conversion complete!")
        print(f"{'=' * 60}")
        print(f"Output directories:")
        for d in output_dirs:
            print(f"  - {d}")

    except (FileNotFoundError, RuntimeError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
