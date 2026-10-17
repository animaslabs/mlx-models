"""
Model card (README.md) generation for converted models.

Compatibility is maintained by hand in COMPATIBILITY below. When you re-check a
runtime, update the entry, COMPATIBILITY_CHECKED and RUNTIME_VERSIONS, then
regenerate the cards.
"""

import re
from pathlib import Path
from typing import Optional

import yaml
from safetensors import safe_open

GITHUB_REPO = "https://github.com/animaslabs/mlx-models"
VOICE_RS_REPO = "https://github.com/animaslabs/mlx-voice-rs"
VOICE_RS_CRATE = "mlx-voice"

RUNTIMES = ["mlx-voice-rs", "mlx-audio", "parakeet-mlx"]

# Checked by transcribing samples/Speaker26_004_16k.wav with every published
# repo in every runtime and comparing against a reference transcript.
COMPATIBILITY_CHECKED = "2026-10-14"
RUNTIME_VERSIONS = {
    "parakeet-mlx": "0.5.3",
    "mlx-audio": "0.5.9",
    "mlx-voice-rs": "8a5b440",
}

# family -> runtime -> (works, note). {base_model} in a note is filled in.
COMPATIBILITY = {
    "tdt": {
        "parakeet-mlx": (True, ""),
        "mlx-audio": (True, ""),
        "mlx-voice-rs": (False, "TDT decoding isn't implemented"),
    },
    "rnnt": {
        "parakeet-mlx": (True, ""),
        "mlx-audio": (True, ""),
        "mlx-voice-rs": (True, ""),
    },
    "ctc": {
        "parakeet-mlx": (True, ""),
        "mlx-audio": (True, ""),
        "mlx-voice-rs": (True, ""),
    },
    # Placeholder: multi-talker support in mlx-voice-rs is in progress, and the
    # published multitalker cards haven't been regenerated yet.
    "multitalker": {
        "parakeet-mlx": (False, "Multi-talker models aren't supported"),
        "mlx-audio": (False, "Multi-talker models aren't supported"),
        "mlx-voice-rs": (False, "Multi-talker support is in progress"),
    },
    "cache-aware-rnnt": {
        "parakeet-mlx": (False, "Cache-aware streaming models aren't supported"),
        "mlx-audio": (True, ""),
        "mlx-voice-rs": (True, ""),
    },
}

# parakeet-mlx and mlx-audio build pointwise convs as nn.Conv1d, which can't load
# pointwise weights that were quantized as linear layers (the -pw repos).
QUANTIZED_POINTWISE_NOTE = "Pointwise convs are quantized as linear layers, which this runtime can't load"

# mlx-voice-rs model type per family
VOICE_RS_MODELS = {
    "tdt": "EncDecRNNTModel",
    "rnnt": "EncDecRNNTModel",
    "ctc": "EncDecCTCModel",
    "multitalker": "CacheAwareRNNTPipeline",
    "cache-aware-rnnt": "CacheAwareRNNTPipeline",
}


def detect_family(nemo_config: dict) -> str:
    """Classify a NeMo ASR model from its model_config.yaml contents."""
    target = nemo_config.get("target", "")
    if "MultiTalker" in target:
        return "multitalker"
    if "CTC" in target:
        return "ctc"
    if nemo_config.get("model_defaults", {}).get("tdt_durations"):
        return "tdt"
    if nemo_config.get("encoder", {}).get("causal_downsampling"):
        return "cache-aware-rnnt"
    return "rnnt"


DTYPE_NAMES = {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}


def describe_weights(weights_path: Path) -> dict:
    """Read the safetensors header: which layer groups are quantized, and float dtypes."""
    quantized, float_dtypes = set(), {}
    with safe_open(str(weights_path), "np") as f:
        for key in f.keys():
            if "pointwise_conv" in key:
                group = "pointwise"
            elif "lstm" in key and not key.endswith("bias"):
                group = "lstm"  # weight matrices; biases are tracked under "other"
            else:
                group = "other"
            if key.endswith(".scales"):
                quantized.add(group)
            elif not key.endswith(".biases"):
                dtype = f.get_slice(key).get_dtype()
                if dtype in DTYPE_NAMES:  # skip integer buffers like num_batches_tracked
                    float_dtypes.setdefault(group, set()).add(DTYPE_NAMES[dtype])
    return {"quantized": quantized, "float_dtypes": float_dtypes}


def compatibility(family: str, weights: dict, base_model: str) -> dict:
    table = {}
    for runtime in RUNTIMES:
        works, note = COMPATIBILITY[family][runtime]
        if works and runtime != "mlx-voice-rs" and "pointwise" in weights["quantized"]:
            works, note = False, QUANTIZED_POINTWISE_NOTE
        table[runtime] = (works, note.format(base_model=base_model))
    return table


def usage_snippet(runtime: str, repo: str, family: str, bits: Optional[int], fmt: str) -> str:
    name = repo.split("/")[-1]
    if runtime == "parakeet-mlx" and bits:
        return f'''```python
# pip install -U parakeet-mlx  (transcribe() needs FFmpeg on PATH)
import json
import mlx.nn as nn
from huggingface_hub import hf_hub_download
from parakeet_mlx.utils import from_config

config = json.load(open(hf_hub_download("{repo}", "config.json")))
model = from_config(config)
nn.quantize(model, bits=config["quantization"]["bits"], group_size=config["quantization"]["group_size"])
model.load_weights(hf_hub_download("{repo}", "model.safetensors"))

print(model.transcribe("audio.wav").text)
```'''
    if runtime == "parakeet-mlx":
        return f'''```python
# pip install -U parakeet-mlx  (transcribe() needs FFmpeg on PATH)
from parakeet_mlx import from_pretrained

model = from_pretrained("{repo}")
print(model.transcribe("audio.wav").text)
```'''
    if runtime == "mlx-audio":
        return f'''```python
# pip install -U mlx-audio
from mlx_audio.stt import load

model = load("{repo}")
print(model.generate("audio.wav").text)
```'''
    crate = VOICE_RS_CRATE.replace("-", "_")
    model_type = VOICE_RS_MODELS[family]
    setup = f"""```bash
cargo add {VOICE_RS_CRATE}
hf download {repo} --local-dir {name}
```"""
    if model_type == "CacheAwareRNNTPipeline":
        code = f'''```rust
use {crate}::asr::CacheAwareRNNTPipeline;
use {crate}::audio::load_wav;

fn main() -> {crate}::Result<()> {{
    // Mode 0 is the lowest-latency streaming mode
    let mut pipeline = CacheAwareRNNTPipeline::restore_from_directory("{name}", 0)?;
    let (samples, _sample_rate) = load_wav("audio.wav")?; // 16 kHz mono

    // Feed audio in 100 ms chunks, as it would arrive from a microphone
    let chunks: Vec<&[f32]> = samples.chunks(1600).collect();
    for (i, chunk) in chunks.iter().enumerate() {{
        pipeline.add_audio(chunk);
        if i == chunks.len() - 1 {{
            pipeline.end_stream();
        }}
        while let Some(output) = pipeline.transcribe_step()? {{
            if output.is_final {{
                println!("{{}}", output.text);
            }}
        }}
    }}
    Ok(())
}}
```'''
    else:
        code = f'''```rust
use {crate}::asr::{model_type};

fn main() -> {crate}::Result<()> {{
    let mut model = {model_type}::restore_from_directory("{name}")?;
    let texts = model.transcribe(&["audio.wav"])?;
    println!("{{}}", texts[0]);
    Ok(())
}}
```'''
    return f"{setup}\n\n{code}"


def describe_format(weights: dict, bits: Optional[int], group_size: int, fmt: str) -> str:
    dtypes = sorted({d for ds in weights["float_dtypes"].values() for d in ds})
    if fmt == "pytorch":
        return (
            "PyTorch-layout weights, exported unchanged from the NeMo checkpoint "
            f"(dtypes: {', '.join(dtypes)}). No MLX transposes or quantization."
        )
    if not bits:
        return f"MLX-layout weights, full precision ({', '.join(dtypes)})."
    layers = "Linear and embedding layers"
    if "pointwise" in weights["quantized"]:
        layers += ", plus pointwise convs (stored as linear layers)"
    lstm = sorted(weights["float_dtypes"].get("lstm", set()))
    lstm_note = f" LSTM weight matrices are {', '.join(lstm)}." if lstm else ""
    return (
        f"MLX-layout weights. {layers} are {bits}-bit affine quantized with group size "
        f"{group_size}; the rest stays in floating point.{lstm_note}"
    )


def render_body(
    repo: str,
    base_model: str,
    family: str,
    weights: dict,
    bits: Optional[int],
    group_size: int,
    fmt: str,
    variants: list[str],
    source_revision: Optional[str],
    nemo_sha256: Optional[str],
    files: list[str],
) -> str:
    kind = f"{bits}-bit quantized MLX" if bits else "MLX" if fmt == "mlx" else "PyTorch safetensors"
    lines = [
        f"# {repo.split('/')[-1]}",
        "",
        f"{kind} conversion of [{base_model}](https://huggingface.co/{base_model}). "
        "See the original model card for training data, evaluation and intended use.",
    ]

    if len(variants) > 1:
        lines += ["", "**Variants:** " + " · ".join(
            f"**{v.split('/')[-1]}**" if v == repo else f"[{v.split('/')[-1]}](https://huggingface.co/{v})"
            for v in variants
        )]
        if any(v.endswith("-pw") for v in variants):
            lines += ["", "Variants ending in `-pw` also quantize the pointwise convs. "
                      "They're smaller, but only mlx-voice-rs can load them."]

    table = compatibility(family, weights, base_model) if fmt == "mlx" else {
        "mlx-voice-rs": COMPATIBILITY[family]["mlx-voice-rs"],
    }
    lines += [
        "",
        "## Compatibility",
        "",
        "| Runtime | Works | Notes |",
        "|---|---|---|",
    ]
    for runtime, (works, note) in table.items():
        lines.append(f"| {runtime} ({RUNTIME_VERSIONS[runtime]}) | {'✅' if works else '❌'} | {note} |")
    lines += ["", f"Last checked {COMPATIBILITY_CHECKED} by transcribing a sample recording in each runtime."]

    working = [r for r, (works, _) in table.items() if works]
    lines += ["", "## Usage"]
    if not working:
        lines += ["", "None of the runtimes above can run this model yet."]
    for runtime in working:
        lines += ["", f"### {runtime}", "", usage_snippet(runtime, repo, family, bits, fmt)]
    if fmt == "pytorch":
        lines += [
            "",
            "### Loading the weights directly",
            "",
            "```python\nfrom safetensors.torch import load_file\n\nweights = load_file(\"model.safetensors\")\n```",
        ]

    file_notes = {
        "model.safetensors": describe_format(weights, bits, group_size, fmt),
        "config.json": (
            "The model config in mlx-audio's `nemotron_asr` layout"
            if family == "cache-aware-rnnt"
            else "The NeMo model config as JSON (the layout parakeet-mlx reads)"
        ) + (", plus a `quantization` block" if bits else "") + ".",
        "model_config.yaml": "The original NeMo config from the `.nemo` archive. mlx-voice-rs reads this one.",
        "tokenizer.model": "SentencePiece tokenizer.",
        "tokenizer.vocab": "SentencePiece vocabulary.",
        "vocab.txt": "Token list.",
    }
    lines += ["", "## Files", ""]
    lines += [f"- `{f}`: {file_notes[f]}" for f in file_notes if f in files]

    source = f"Converted from [{base_model}](https://huggingface.co/{base_model})"
    if source_revision:
        source += f" at revision [`{source_revision[:10]}`](https://huggingface.co/{base_model}/tree/{source_revision})"
    if nemo_sha256:
        source += f" (`.nemo` sha256 `{nemo_sha256[:12]}…`)"
    lines += ["", "## Source", "", f"{source} using [animaslabs/mlx-models]({GITHUB_REPO})."]
    return "\n".join(lines) + "\n"


def write_model_card(
    output_dir: str,
    source_repo: str,
    output_repo: str,
    metadata: dict,
    bits: Optional[int] = None,
    group_size: int = 64,
    fmt: str = "mlx",
    variants: Optional[list[str]] = None,
    source_revision: Optional[str] = None,
    nemo_sha256: Optional[str] = None,
    template_path: str = "MODEL_CARD.md",
) -> None:
    """Write README.md into output_dir. Expects model.safetensors and model_config.yaml there."""
    out = Path(output_dir)
    nemo_config = yaml.safe_load((out / "model_config.yaml").read_text())
    family = detect_family(nemo_config)
    weights = describe_weights(out / "model.safetensors")

    # Upstream library tags (e.g. NVIDIA's Transformers and GGUF files) don't apply here
    tags = [t for t in metadata.get("tags", []) if t.lower() not in ("transformers", "gguf")]
    if fmt == "mlx":
        library_name = "mlx"
        tags += [t for t in ["mlx"] if t not in tags]
        if bits and "quantized" not in tags:
            tags.insert(0, "quantized")
    else:
        library_name = "safetensors"
        tags += [t for t in ["safetensors"] if t not in tags]

    license_lines = [f"license: {metadata['license']}"]
    for key in ("license_name", "license_link"):
        if metadata.get(key):
            license_lines.append(f"{key}: {metadata[key]}")

    def yaml_list(items):
        return "\n".join(f"- {i}" for i in items) if items else "[]"

    # Also files written after the card (config.json is created later in main.py)
    files = sorted({p.name for p in out.iterdir()} | ({"config.json"} if fmt == "mlx" else set()))
    body = render_body(
        output_repo, source_repo, family, weights, bits, group_size, fmt,
        variants or [output_repo], source_revision, nemo_sha256, files,
    )
    content = Path(template_path).read_text().format(
        license_block="\n".join(license_lines),
        languages=yaml_list(metadata.get("language", ["en"])),
        pipeline_tag=metadata.get("pipeline_tag", "automatic-speech-recognition"),
        library_name=library_name,
        datasets=yaml_list(metadata.get("datasets", [])),
        tags=yaml_list(tags),
        base_model=source_repo,
        base_model_relation="quantized" if bits else "finetune",
        body=body,
    )
    (out / "README.md").write_text(content)
    print(f"Generated README.md ({family}, {'pytorch' if fmt == 'pytorch' else f'{bits}-bit' if bits else 'full precision'})")
