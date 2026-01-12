#!/usr/bin/env python3
"""
Example: Transcribe audio using Parakeet MLX.

This script downloads a sample audio file and transcribes it using either
a full-precision or quantized Parakeet model.

Usage:
    python examples/transcribe_librispeech.py
    python examples/transcribe_librispeech.py --model mlx-community/parakeet-tdt-0.6b-v3
    python examples/transcribe_librispeech.py --quantized --model path/to/quantized-model
"""

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def download_sample_audio(output_path: Path) -> Path:
    """Download a sample speech audio file."""
    # Open Speech Repository - public domain American English speech samples
    # Source: https://www.voiptroubleshooter.com/open_speech/american.html
    sample_url = (
        "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav"
    )

    print(f"Downloading sample audio...")

    # Some servers require a proper User-Agent header
    request = urllib.request.Request(
        sample_url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
    )
    with urllib.request.urlopen(request) as response:
        with open(output_path, "wb") as f:
            f.write(response.read())

    print(f"Downloaded to {output_path}")

    return output_path


def load_model_standard(model_id: str):
    """Load a standard (non-quantized) model."""
    from parakeet_mlx.utils import from_pretrained

    print(f"Loading model: {model_id}")
    model = from_pretrained(model_id)
    return model


def load_model_quantized(model_path: str, bits: int = 8, group_size: int = 64):
    """Load a quantized model with proper nn.quantize() call."""
    from parakeet_mlx.utils import from_config

    # Handle both local paths and HuggingFace repos
    if Path(model_path).exists():
        config_path = Path(model_path) / "config.json"
        weight_path = Path(model_path) / "model.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_path, "config.json")
        weight_path = hf_hub_download(model_path, "model.safetensors")

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Get quantization settings from config or use defaults
    quant_config = config.get("quantization", {})
    bits = quant_config.get("bits", bits)
    group_size = quant_config.get("group_size", group_size)

    print(f"Loading quantized model: {model_path}")
    print(f"  Quantization: {bits}-bit, group_size={group_size}")

    # Build model with standard layers
    model = from_config(config)

    # Convert to quantized layers
    nn.quantize(model, bits=bits, group_size=group_size)

    # Load quantized weights
    model.load_weights(str(weight_path))

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a LibriSpeech sample using Parakeet MLX"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/parakeet-tdt-0.6b-v3",
        help="Model ID (HuggingFace repo or local path)",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Load as quantized model (requires quantized weights)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Quantization bits (default: 8)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file (downloads sample if not provided)",
    )
    args = parser.parse_args()

    # Get audio file
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            return 1
    else:
        # Download sample audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = Path(f.name)
        download_sample_audio(audio_path)

    # Load model
    if args.quantized:
        model = load_model_quantized(args.model, bits=args.bits)
    else:
        model = load_model_standard(args.model)

    # Transcribe
    print(f"\nTranscribing: {audio_path}")
    result = model.transcribe(audio_path)

    # Print results
    print(f"\n{'=' * 60}")
    print("TRANSCRIPTION")
    print(f"{'=' * 60}")
    print(result.text)
    print(f"{'=' * 60}")

    # Print timing info if available
    if hasattr(result, "sentences") and result.sentences:
        print(f"\nSegments ({len(result.sentences)} sentences):")
        for i, sentence in enumerate(result.sentences):
            start = sentence.tokens[0].start if sentence.tokens else 0
            end = sentence.tokens[-1].end if sentence.tokens else 0
            print(f"  [{start:.2f}s - {end:.2f}s] {sentence.text}")

    # Cleanup temp file if we downloaded it
    if not args.audio:
        audio_path.unlink()

    return 0


if __name__ == "__main__":
    exit(main())
