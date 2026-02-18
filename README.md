# mlx-models

Convert NeMo speech recognition models to MLX format (optimized for Apple Silicon) or generic PyTorch safetensors, and publish them to HuggingFace Hub.

## Requirements

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/)

## Authentication

Set your HuggingFace token for faster downloads and uploading to the Hub:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

Generate a token at https://huggingface.co/settings/tokens.

Alternatively, log in once via the CLI:

```bash
uv run huggingface-cli login
```

## Commands

### `main.py` — Full conversion pipeline

Downloads a NeMo model from HuggingFace, converts it, and uploads the result.

```bash
# MLX format, full precision
uv run main.py nvidia/parakeet-tdt-0.6b-v3 -o username/model-mlx

# MLX format with 8-bit quantization
uv run main.py nvidia/parakeet-tdt-0.6b-v3 -q 8 -o username/model-mlx

# MLX format with 4-bit and 8-bit quantization variants
uv run main.py nvidia/parakeet-tdt-0.6b-v3 -q 4 8 -o username/model-mlx

# Include full precision alongside quantized versions
uv run main.py nvidia/parakeet-tdt-0.6b-v3 -q 4 8 --include-full-precision -o username/model-mlx

# Generic PyTorch safetensors (no MLX transforms)
uv run main.py nvidia/parakeet-tdt-0.6b-v3 --format pytorch -o username/model

# PyTorch with float16 casting
uv run main.py nvidia/parakeet-tdt-0.6b-v3 --format pytorch --dtype float16 -o username/model

# Local only, skip upload
uv run main.py nvidia/parakeet-tdt-0.6b-v3 -q 8 --no-upload

# Private repo with custom quantization group size
uv run main.py nvidia/parakeet-tdt-0.6b-v3 -q 8 -o username/model -g 32 --private
```

| Flag | Short | Description |
|------|-------|-------------|
| `--output-repo` | `-o` | Base output repository ID (suffixes like `-mlx-8bit` added automatically) |
| `--format` | `-f` | Output format: `mlx` (default) or `pytorch` |
| `--quantize` | `-q` | Quantization bits: `4`, `8`, or both (MLX only) |
| `--include-full-precision` | | Also output a full precision model when quantizing |
| `--dtype` | | Cast tensors to `float16` or `float32` (PyTorch only) |
| `--no-upload` | | Skip uploading to HuggingFace |
| `--local-dir` | `-d` | Local directory for downloads (default: `models/<repo_id>`) |
| `--private` | `-p` | Create private HuggingFace repos |
| `--group-size` | `-g` | Quantization group size (default: 64) |

### `hf.py` — HuggingFace Hub utilities

```bash
# Download a model
uv run hf.py download nvidia/parakeet-tdt-0.6b-v3

# Download to a specific directory
uv run hf.py download nvidia/parakeet-tdt-0.6b-v3 -d ./my-models

# Download a specific revision
uv run hf.py download nvidia/parakeet-tdt-0.6b-v3 -r main

# Upload a converted model
uv run hf.py upload models/my-model username/my-model-mlx

# Upload as private
uv run hf.py upload models/my-model username/my-model-mlx --private

# Upload with a custom commit message
uv run hf.py upload models/my-model username/my-model-mlx -m "Add 8-bit quantized model"

# Unpack a .nemo file
uv run hf.py unpack models/model.nemo -o ./unpacked
```

### `convert_ckpt.py` — Standalone checkpoint conversion

Converts a PyTorch checkpoint to safetensors without any MLX-specific transforms.

```bash
# Basic conversion, keep original dtypes
uv run convert_ckpt.py checkpoint.ckpt output.safetensors

# Cast float32 tensors to float16
uv run convert_ckpt.py checkpoint.ckpt output.safetensors --dtype float16
```

### `examples/transcribe.py` — Test a converted model

```bash
# Transcribe with the default model and a sample audio file
uv run examples/transcribe.py

# Use a specific model
uv run examples/transcribe.py --model mlx-community/parakeet-tdt-0.6b-v3

# Use a quantized model
uv run examples/transcribe.py --model username/model-mlx-8bit --quantized --bits 8

# Transcribe your own audio file
uv run examples/transcribe.py --audio recording.wav
```

## Typical workflow

```bash
# 1. Convert a NeMo model to MLX with full precision + quantized variants
uv run main.py nvidia/parakeet-tdt-0.6b-v3 \
  -q 4 8 --include-full-precision \
  -o username/parakeet-tdt-0.6b-v3

# 2. Test the converted model locally
uv run examples/transcribe.py \
  --model models/parakeet-tdt-0.6b-v3-mlx \
  --audio samples/test.wav
```
