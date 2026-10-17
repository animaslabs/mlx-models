#!/usr/bin/env python3
"""
Benchmark mlx-voice-rs against parakeet-mlx, mlx-audio and NVIDIA NeMo:
speed (load time, wall time, RTF, streaming chunk latency) and WER on the
shared eval set, for each model in MODELS.

Every library gets the same clips, one untimed warm-up on the first clip, and
is timed from WAV path to text. Results land in bench/results/<timestamp>/
as one JSONL per library x model, plus report.md and report.json.

Usage:
    uv run bench/bench.py                                # all models, English clips
    uv run bench/bench.py --model nemotron --only nemo   # partial run
    uv run bench/bench.py --max-duration 400 --repeats 3
    uv run bench/bench.py --all-clips                    # ignore per-model max_duration
    uv run bench/bench.py --report bench/results/<timestamp>   # rebuild report only

Needs a mlx-voice-rs checkout (default ../mlx-voice-rs), `uv sync --project
bench/nemo` for NeMo, and access to the gated nvidia/nemotron model on the Hub.
"""

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
sys.path[:0] = [str(BENCH), str(ROOT / "samples")]

from common import EVAL_DIR, load_clips  # noqa: E402

# Each model runs from the same full-precision weights everywhere: the original
# .nemo for mlx-voice-rs and NeMo, its unquantized MLX conversion for the MLX
# libraries. `mode` indexes encoder.att_context_size (nemotron 0 = [70, 13] and
# nemotron-3.5 3 = [56, 13]: 1.12 s chunks). `target_lang` is the language prompt
# of a multilingual model ("auto" detects it) and `languages` picks the clips
# (default English only; None = every language). A bare "mlx" name is a local
# conversion under models/; models without one skip the MLX libraries.
# Offline runs use full attention over the whole file, so memory grows with the
# square of the clip length: an 18 min clip peaks near 40 GB in mlx-voice-rs, and
# NeMo on MPS bogs the whole machine down on the 22 min clips. `max_duration`
# keeps the default run light; --all-clips lifts it.
MODELS = {
    "parakeet-rnnt": {
        "streaming": False,
        "max_duration": 400,
        "nemo": ("nvidia/parakeet-rnnt-0.6b", "parakeet-rnnt-0.6b.nemo"),
        "mlx": "animaslabs/parakeet-rnnt-0.6b-mlx",
    },
    "nemotron": {
        "streaming": True,
        "mode": 0,
        "nemo": ("nvidia/nemotron-speech-streaming-en-0.6b", "nemotron-speech-streaming-en-0.6b.nemo"),
        "mlx": "animaslabs/nemotron-speech-streaming-en-0.6b-mlx",
    },
    # Not on the Hub yet: convert with `python -m mlx_audio.stt.models.nemotron_asr.convert
    # --nemo-path <.nemo> --mlx-path models/nemotron-3.5-asr-streaming-0.6b-mlx --dtype float32`.
    "nemotron-3.5": {
        "streaming": True,
        "mode": 3,
        "languages": None,
        "target_lang": "auto",
        "nemo": ("nvidia/nemotron-3.5-asr-streaming-0.6b", "nemotron-3.5-asr-streaming-0.6b.nemo"),
        "mlx": "nemotron-3.5-asr-streaming-0.6b-mlx",
    },
}
MLX_LIBRARIES = ["parakeet-mlx", "mlx-audio"]
LIBRARIES = ["mlx-voice-rs", "parakeet-mlx", "mlx-audio", "nemo"]
GROUPS = ["multilingual", "long-form", "multi-speaker", "all"]


def nemo_path(spec) -> str:
    local = ROOT / "models" / "nvidia" / spec[0].split("/")[1] / spec[1]
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(spec[0], spec[1])


def mlx_path(spec) -> str | None:
    # A bare name is a local conversion under models/; anything else is a Hub id.
    name = spec.get("mlx")
    if not name or "/" in name:
        return name
    local = ROOT / "models" / name
    return str(local) if local.exists() else None


def run_library(lib, key, spec, clips, args, out: Path) -> None:
    common = ["--warmup", str(args.warmup), "--repeats", str(args.repeats),
              "--eval-dir", str(args.eval_dir), "--out", str(out)]
    for clip in clips:
        common += ["--clip", clip["name"]]
    if spec["streaming"]:
        common += ["--mode", str(spec.get("mode", 0))]
    if "target_lang" in spec:
        common += ["--target-lang", spec["target_lang"]]
    if lib in MLX_LIBRARIES and not mlx_path(spec):
        out.write_text(json.dumps({"library": lib, "model": key, "unsupported": "no MLX conversion"}) + "\n")
        return

    if lib == "mlx-voice-rs":
        binary = args.rust_repo / "target" / "release" / "examples" / "eval"
        cmd = [str(binary), "--model", nemo_path(spec["nemo"])] + common
        if not spec["streaming"]:
            cmd.append("--offline")
    else:
        model = nemo_path(spec["nemo"]) if lib == "nemo" else mlx_path(spec)
        project = ["--project", str(BENCH / "nemo")] if lib == "nemo" else []
        cmd = ["uv", "run", *project, str(BENCH / "run.py"), "--library", lib,
               "--model", model, "--model-name", key] + common
        if spec["streaming"]:
            cmd.append("--streaming")
    print(f"\n=== {lib} / {key}\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"!!! {lib} / {key} exited with {result.returncode}")
        if not out.exists() or not out.read_text().strip():
            out.write_text(json.dumps({"library": lib, "unsupported": f"runner exited with {result.returncode}"}) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_report(results_dir: Path, eval_dir: Path) -> None:
    from score import make_normalizers, score

    manifest = [json.loads(line) for line in (eval_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    normalize = make_normalizers()
    report, md = {}, [f"# ASR benchmark — {results_dir.name}\n"]

    for key in MODELS:
        files = sorted(results_dir.glob(f"*__{key}.jsonl"),
                       key=lambda p: LIBRARIES.index(p.name.split("__")[0]) if p.name.split("__")[0] in LIBRARIES else 99)
        if not files:
            continue
        entries = {}
        for path in files:
            lib = path.name.split("__")[0]
            rows = read_rows(path)
            unsupported = next((r["unsupported"] for r in rows if "unsupported" in r), None)
            ok = [r for r in rows if "text" in r]
            errors = {r["name"]: r["error"] for r in rows if "error" in r}
            entry = {"unsupported": unsupported, "errors": errors, "clips": len(ok),
                     "attempted": len(ok) + len(errors)}
            if ok:
                audio = sum(r["duration"] for r in ok)
                elapsed = sum(r["elapsed"] for r in ok)
                scores = score(manifest, eval_dir, {r["name"]: r["text"] for r in ok}, normalize)
                entry |= {
                    "device": ok[0].get("device"), "dtype": ok[0].get("dtype"),
                    "load_time": ok[0].get("load_time"), "audio": audio, "elapsed": elapsed,
                    "rtf": elapsed / audio,
                    "wer": {g: v["wer"] for g, v in scores["groups"].items()},
                    "clip_wer": {n: v["wer"] for n, v in scores["clips"].items()},
                    "clip_rtf": {r["name"]: r["elapsed"] / r["duration"] for r in ok},
                }
                chunk = [r["chunk_ms_mean"] for r in ok if "chunk_ms_mean" in r]
                if chunk:
                    entry["chunk_ms_mean"] = statistics.fmean(chunk)
                    entry["chunk_ms_p95"] = max(r["chunk_ms_p95"] for r in ok if "chunk_ms_p95" in r)
            entries[lib] = entry
        report[key] = entries

        spec = MODELS[key]
        groups = [g for g in GROUPS if any(g in e.get("wer", {}) for e in entries.values())]
        mode = f"streaming, mode {spec.get('mode', 0)}" if spec["streaming"] else "offline"
        if spec.get("target_lang"):
            mode += f", target_lang {spec['target_lang']}"
        md.append(f"\n## {key} ({mode})\n")
        cols = ["library", "device", "clips", "load s", "audio s", "elapsed s", "RTF", "× realtime"]
        if spec["streaming"]:
            cols += ["chunk ms mean", "chunk ms p95 (max)"]
        cols += [f"WER {g}" for g in groups]
        md += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for lib, e in entries.items():
            if e["unsupported"]:
                md.append(f"| {lib} | n/a ({e['unsupported']}) |" + " |" * (len(cols) - 2))
                continue
            if not e["clips"]:
                md.append(f"| {lib} | all clips failed |" + " |" * (len(cols) - 2))
                continue
            cells = [lib, f"{e['device']} {e['dtype']}", f"{e['clips']}/{e['attempted']}",
                     f"{e['load_time']:.2f}", f"{e['audio']:.0f}", f"{e['elapsed']:.2f}",
                     f"{e['rtf']:.4f}", f"{1 / e['rtf']:.0f}×"]
            if spec["streaming"]:
                cells += [f"{e['chunk_ms_mean']:.1f}" if "chunk_ms_mean" in e else "—",
                          f"{e['chunk_ms_p95']:.1f}" if "chunk_ms_p95" in e else "—"]
            cells += [f"{e['wer'][g] * 100:.2f}%" if g in e["wer"] else "—" for g in groups]
            md.append("| " + " | ".join(cells) + " |")

        # Per-clip WER / RTF, one column per library that produced output.
        libs = [lib for lib, e in entries.items() if e.get("clip_wer")]
        names = [m["name"] for m in manifest if any(m["name"] in entries[l]["clip_wer"] for l in libs)]
        if libs:
            md += ["", "| clip | " + " | ".join(f"{l} WER / RTF" for l in libs) + " |",
                   "|---|" + "---|" * len(libs)]
            for name in names:
                cells = []
                for lib in libs:
                    e = entries[lib]
                    if name in e["clip_wer"]:
                        cells.append(f"{e['clip_wer'][name] * 100:.2f}% / {e['clip_rtf'][name]:.4f}")
                    else:
                        cells.append("error" if name in e["errors"] else "—")
                md.append(f"| {name} | " + " | ".join(cells) + " |")
        for lib, e in entries.items():
            for name, err in e["errors"].items():
                md.append(f"\n- {lib} failed on {name}: `{err[:200]}`")

    md.append("\nWER is corpus WER over the clips each library completed (samples/score.py normalization); "
              "a different clip count means the WERs are not directly comparable. "
              "Times exclude model load and include audio decoding; RTF = elapsed / audio.")
    (results_dir / "report.md").write_text("\n".join(md) + "\n")
    (results_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("\n" + "\n".join(md))
    print(f"\nWrote {results_dir / 'report.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", action="append", choices=MODELS, help="Only these models (repeatable)")
    parser.add_argument("--only", action="append", choices=LIBRARIES, help="Only these libraries (repeatable)")
    parser.add_argument("--language", action="append", help="Clip languages (default: each model's languages)")
    parser.add_argument("--clip", action="append", default=[], help="Only this clip (repeatable)")
    parser.add_argument("--max-duration", type=float, help="Skip clips longer than this (seconds)")
    parser.add_argument("--all-clips", action="store_true", help="Ignore each model's max_duration")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--rust-repo", type=Path, default=ROOT.parent / "mlx-voice-rs")
    parser.add_argument("--out-dir", type=Path, help="Results directory (default: bench/results/<timestamp>)")
    parser.add_argument("--report", type=Path, metavar="DIR", help="Only rebuild the report for DIR")
    args = parser.parse_args()
    args.eval_dir = args.eval_dir.resolve()

    if args.report:
        build_report(args.report, args.eval_dir)
        return

    models = args.model or list(MODELS)
    clips = {key: load_clips(args.eval_dir, args.language or MODELS[key].get("languages", ["en"]) or [],
                             args.clip, args.max_duration) for key in models}
    if not any(clips.values()):
        raise SystemExit(f"no clips selected (run `uv run samples/fetch_eval.py` or "
                         f"`cargo xtask download-eval` for {args.eval_dir})")
    libraries = args.only or LIBRARIES
    results_dir = args.out_dir or BENCH / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"results → {results_dir}")

    if "mlx-voice-rs" in libraries:
        subprocess.run(["cargo", "build", "--release", "--example", "eval",
                        "--manifest-path", str(args.rust_repo / "Cargo.toml")], check=True)

    for key in models:
        limit = None if args.all_clips else MODELS[key].get("max_duration")
        model_clips = [c for c in clips[key] if limit is None or c["duration"] <= limit]
        print(f"{key}: {len(model_clips)}/{len(clips[key])} clips, "
              f"{sum(c['duration'] for c in model_clips):.0f}s audio"
              + (f" (max_duration {limit}s; --all-clips for all)" if len(model_clips) < len(clips[key]) else ""))
        for lib in libraries:
            run_library(lib, key, MODELS[key], model_clips, args, results_dir / f"{lib}__{key}.jsonl")

    build_report(results_dir, args.eval_dir)


if __name__ == "__main__":
    main()
