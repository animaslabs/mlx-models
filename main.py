#!/usr/bin/env python3
"""
Main script for converting NeMo models to MLX-compatible safetensors format.

Workflow:
1. Download a model from Hugging Face Hub
2. Unpack the NeMo file (tar archive)
3. Convert checkpoint to safetensors with optional quantization
4. Optionally upload converted model(s) to Hugging Face Hub

Usage:
    # Basic conversion (no quantization)
    python main.py nvidia/parakeet-tdt-0.6b-v3 --output-repo username/model-mlx
    
    # With 8-bit quantization
    python main.py nvidia/parakeet-tdt-0.6b-v3 --quantize 8 --output-repo username/model-mlx-8bit
    
    # Multiple quantizations
    python main.py nvidia/parakeet-tdt-0.6b-v3 --quantize 4 8 --output-repo username/model-mlx
    
    # Without upload
    python main.py nvidia/parakeet-tdt-0.6b-v3 --quantize 8 --no-upload
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

from hf import download_model, unpack_nemo, upload_model
from convert_nemo import (
    load_checkpoint,
    convert_weights_to_mlx,
    save_safetensors,
    QUANTIZE_GROUP_SIZE,
)


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


def copy_model_card(output_dir: str, quantize_bits: Optional[int] = None, model_card_path: str = "MODEL_CARD.md") -> None:
    """Copy MODEL_CARD.md to output directory as README.md, adding quantize tag if needed."""
    src_path = Path(model_card_path)
    if not src_path.exists():
        print(f"Warning: {model_card_path} not found, skipping README.md")
        return
    
    content = src_path.read_text()
    
    # Add quantize tag for quantized models
    if quantize_bits is not None:
        # Insert "quantized" tag after existing tags
        if "tags:" in content:
            content = content.replace("tags:\n", "tags:\n- quantized\n", 1)
    
    dst_path = Path(output_dir) / "README.md"
    dst_path.write_text(content)
    print(f"Copied {model_card_path} to README.md" + (f" (added quantized tag)" if quantize_bits else ""))


def create_config_json(output_dir: str, quantize_bits: Optional[int] = None) -> None:
    """Create a config.json file for the MLX model."""
    import json
    
    config = {
        "model_type": "parakeet-tdt",
        "framework": "mlx",
    }
    
    if quantize_bits:
        config["quantization"] = {
            "bits": quantize_bits,
            "group_size": QUANTIZE_GROUP_SIZE,
        }
    
    config_path = Path(output_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Created config.json")


def get_output_suffix(quantize_bits: Optional[int]) -> str:
    """Get the output directory suffix based on quantization."""
    if quantize_bits is None:
        return "-mlx"
    return f"-mlx-{quantize_bits}bit"


def convert_model(
    repo_id: str,
    output_repo: Optional[str] = None,
    quantize_bits: list[int] | None = None,
    upload: bool = True,
    local_dir: Optional[str] = None,
    private: bool = False,
    group_size: int = QUANTIZE_GROUP_SIZE,
) -> list[str]:
    """
    Main conversion pipeline.
    
    Args:
        repo_id: Source HuggingFace repository ID
        output_repo: Base output repository ID (suffixes added for quantization)
        quantize_bits: List of quantization bit widths (None, 4, 8)
        upload: Whether to upload to HuggingFace
        local_dir: Local directory for downloads
        private: Whether to create private repos
        group_size: Quantization group size
    
    Returns:
        List of output directory paths
    """
    # Default quantization: just full precision
    if quantize_bits is None:
        quantize_bits = [None]
    
    # Step 1: Download the model
    print(f"\n{'='*60}")
    print(f"Step 1: Downloading {repo_id}")
    print(f"{'='*60}")
    
    if local_dir is None:
        local_dir = f"models/{repo_id}"
    
    model_dir = download_model(repo_id, local_dir)
    
    # Step 2: Find and unpack NeMo file
    print(f"\n{'='*60}")
    print(f"Step 2: Unpacking NeMo file")
    print(f"{'='*60}")
    
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
    
    # Load checkpoint once (reuse for all quantizations)
    print(f"\nLoading checkpoint...")
    state_dict = load_checkpoint(str(ckpt_file))
    
    # Step 4: Convert for each quantization level
    output_dirs = []
    
    for bits in quantize_bits:
        suffix = get_output_suffix(bits)
        quant_str = f"{bits}-bit quantized" if bits else "full precision"
        
        print(f"\n{'='*60}")
        print(f"Step 3: Converting to {quant_str}")
        print(f"{'='*60}")
        
        # Determine output directory
        repo_name = repo_id.split("/")[-1]
        output_dir = f"models/{repo_name}{suffix}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert weights
        print(f"Converting weights...")
        mlx_weights = convert_weights_to_mlx(state_dict, bits, group_size)
        
        # Save safetensors
        output_file = Path(output_dir) / "model.safetensors"
        save_safetensors(mlx_weights, str(output_file))
        
        # Copy tokenizer files
        copy_tokenizer_files(unpacked_dir, output_dir)
        
        # Copy model card as README.md
        copy_model_card(output_dir, bits)
        
        # Create config
        create_config_json(output_dir, bits)
        
        output_dirs.append(output_dir)
        
        # Step 5: Upload if requested
        if upload and output_repo:
            print(f"\n{'='*60}")
            print(f"Step 4: Uploading to HuggingFace")
            print(f"{'='*60}")
            
            target_repo = f"{output_repo}{suffix}"
            upload_model(output_dir, target_repo, private=private)
    
    return output_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeMo models to MLX safetensors format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion (full precision only)
    python main.py nvidia/parakeet-tdt-0.6b-v3 --output-repo username/parakeet-mlx
    
    # With 8-bit quantization
    python main.py nvidia/parakeet-tdt-0.6b-v3 -q 8 --output-repo username/parakeet-mlx
    
    # Multiple quantizations (full, 4-bit, 8-bit)
    python main.py nvidia/parakeet-tdt-0.6b-v3 -q 4 8 --output-repo username/parakeet-mlx
    
    # Local conversion only (no upload)
    python main.py nvidia/parakeet-tdt-0.6b-v3 -q 8 --no-upload
        """
    )
    
    parser.add_argument(
        "repo_id",
        help="Source HuggingFace repository ID (e.g., nvidia/parakeet-tdt-0.6b-v3)"
    )
    parser.add_argument(
        "--output-repo", "-o",
        help="Base output repository ID (suffixes added for quantization)"
    )
    parser.add_argument(
        "--quantize", "-q",
        type=int,
        nargs="+",
        choices=[4, 8],
        default=None,
        help="Quantization bit widths (4 and/or 8). Omit for full precision only."
    )
    parser.add_argument(
        "--include-full-precision",
        action="store_true",
        help="Include full precision model when quantizing"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to HuggingFace"
    )
    parser.add_argument(
        "--local-dir", "-d",
        help="Local directory for downloaded model"
    )
    parser.add_argument(
        "--private", "-p",
        action="store_true",
        help="Create private repositories"
    )
    parser.add_argument(
        "--group-size", "-g",
        type=int,
        default=QUANTIZE_GROUP_SIZE,
        help=f"Quantization group size (default: {QUANTIZE_GROUP_SIZE})"
    )
    
    args = parser.parse_args()
    
    # Build quantization list
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
        )
        
        print(f"\n{'='*60}")
        print("Conversion complete!")
        print(f"{'='*60}")
        print(f"Output directories:")
        for d in output_dirs:
            print(f"  - {d}")
            
    except (FileNotFoundError, RuntimeError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
