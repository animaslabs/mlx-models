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
uv run hf auth login
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
| `--pointwise` | | Pointwise convs in quantized variants: `unquantized` (default), `quantized` (builds `-pw` repos: smaller, but parakeet-mlx and mlx-audio can't load them) or `both` |

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

### `samples/fetch_eval.py` — Download an evaluation set

Downloads 13 clips with reference transcripts to `samples/eval/` (gitignored), as 16 kHz mono WAV + `.txt`, with a `manifest.jsonl`:

| Category | Clips | Source |
|----------|-------|--------|
| Multilingual (~90 s each) | de, fr, es, it, pt, nl, pl | Multilingual LibriSpeech test |
| Long-form, single speaker (~6 min) | 2 TED talks | TED-LIUM long-form |
| Multi-speaker (3–22 min) | 2 earnings calls, 2 podcasts | Earnings-21, Rev16 |

```bash
uv run samples/fetch_eval.py
```

The built set is mirrored to the private HF dataset [`animaslabs/asr-eval`](https://huggingface.co/datasets/animaslabs/asr-eval), so other repos (e.g. mlx-voice-rs, via `cargo xtask download-eval`) test against identical audio. After changing the set, re-push and bump `EVAL_REVISION` in `samples/score.py` and in mlx-voice-rs `xtask/src/main.rs`:

```bash
uv run samples/fetch_eval.py --push animaslabs/asr-eval
```

### `samples/score.py` — Score hypotheses (WER)

The one scorer for every port: normalizes reference and hypothesis with Whisper's normalizer (English for `en`, basic otherwise), then computes per-clip and per-category corpus WER with jiwer. Input is JSONL, one `{"name": ..., "text": ...}` per clip; pass several files to compare them side by side. References come from the pinned dataset revision.

```bash
uv run samples/score.py ../mlx-voice-rs/hypotheses.jsonl
uv run samples/score.py rust.jsonl python.jsonl --json scores.json
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

## Published models

Current state of the [animaslabs](https://huggingface.co/animaslabs) speech repos, maintained by hand. Last checked 2026-10-14 by transcribing `samples/Speaker26_004_16k.wav` with each repo in each runtime (parakeet-mlx 0.5.3, mlx-audio 0.5.9, mlx-voice-rs `8a5b440`).

| Model | Variants | parakeet-mlx | mlx-audio | mlx-voice-rs |
|---|---|---|---|---|
| [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | mlx, 8bit, 4bit | ✅ | ✅ | ❌ no TDT decoding |
| [parakeet-rnnt-0.6b](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) | mlx, 8bit, 4bit | ✅ | ✅ | ✅ |
| [parakeet-ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b) | mlx, pytorch | ✅ (mlx) | ✅ (mlx) | ✅ |
| [nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) | mlx, 8bit, 4bit | ❌ | ❌ | ✅ |
| [multitalker-parakeet-streaming-0.6b-v1](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) | mlx, 8bit, 4bit | ❌ | ❌ | 🚧 in progress |

Notes:

- **Quantization layout and `-pw` repos.** Quantized repos without a suffix leave pointwise convs unquantized, because parakeet-mlx and mlx-audio build them as `nn.Conv1d`. Repos ending in `-pw` (e.g. `-mlx-4bit-pw`) also quantize pointwise convs as linear layers, which is about 40% smaller at 4 bits but only loads in mlx-voice-rs. Build them with `--pointwise quantized` or `--pointwise both`. The published quantized nemotron repos predate this scheme: they use the `-pw` layout without the suffix. The quantized parakeet-tdt, parakeet-rnnt and multitalker repos also store the LSTM as float32; the current converter stores it as float16.
- **nemotron in mlx-audio.** The published nemotron repos have a NeMo-layout `config.json`, which mlx-audio can't read (it loads the original `nvidia/nemotron-speech-streaming-en-0.6b` directly). The converter now writes cache-aware streaming models' `config.json` in mlx-audio's native `nemotron_asr` layout. Local rebuilds of the full-precision, 8-bit and 4-bit repos transcribe the sample in mlx-audio 0.5.9 and mlx-voice-rs (and the `-pw` builds in mlx-voice-rs only). The table will change once the repos are re-uploaded.
- **Compatibility on model cards** comes from `COMPATIBILITY` in `model_card.py`. Update it (and this table) when you re-check a runtime.
