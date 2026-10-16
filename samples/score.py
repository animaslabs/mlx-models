#!/usr/bin/env python3
"""
Score ASR hypotheses against the shared eval set (animaslabs/asr-eval).

The one scorer for every port, so WER from mlx-voice-rs and mlx-models is
directly comparable: both sides go through Whisper's text normalizer (the
English one for `en`, the basic one otherwise) before jiwer computes WER.

Hypotheses are JSON lines, one per clip; any other fields are ignored:
    {"name": "mls_de", "text": "eins ein kühnes unternehmen ..."}

Usage:
    uv run samples/score.py hyps.jsonl
    uv run samples/score.py rust.jsonl python.jsonl      # side by side
    uv run samples/score.py hyps.jsonl --eval-dir samples/eval
    uv run samples/score.py hyps.jsonl --json scores.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import jiwer
from huggingface_hub import hf_hub_download, snapshot_download
from transformers.models.whisper.english_normalizer import (
    BasicTextNormalizer,
    EnglishTextNormalizer,
)

EVAL_REPO = "animaslabs/asr-eval"
# Bump when the dataset changes; scores are only comparable at the same revision.
EVAL_REVISION = "b74107914b1f6530a7096fd6f6b1bcb047486237"


def load_eval(eval_dir: Path | None, revision: str) -> tuple[Path, list[dict]]:
    if eval_dir is None:
        eval_dir = Path(snapshot_download(
            EVAL_REPO, repo_type="dataset", revision=revision,
            allow_patterns=["manifest.jsonl", "*.txt"],
        ))
    manifest = [json.loads(line) for line in (eval_dir / "manifest.jsonl").read_text().splitlines()]
    return eval_dir, manifest


def make_normalizers():
    # The British→American spelling map ships with Whisper's tokenizer files.
    mapping = json.loads(Path(hf_hub_download("openai/whisper-small", "normalizer.json")).read_text())
    english, basic = EnglishTextNormalizer(mapping), BasicTextNormalizer()
    return lambda text, lang: (english if lang == "en" else basic)(text)


def load_hypotheses(path: Path) -> dict[str, str]:
    hyps = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            hyps[row["name"]] = row["text"]
    return hyps


def score(manifest, eval_dir, hyps, normalize) -> dict:
    clips, totals = {}, defaultdict(lambda: defaultdict(int))
    for entry in manifest:
        name = entry["name"]
        if name not in hyps:
            continue
        ref = normalize((eval_dir / entry["reference"]).read_text(), entry["language"])
        hyp = normalize(hyps[name], entry["language"])
        out = jiwer.process_words(ref, hyp)
        counts = {
            "ref_words": out.hits + out.substitutions + out.deletions,
            "substitutions": out.substitutions,
            "deletions": out.deletions,
            "insertions": out.insertions,
        }
        clips[name] = {"wer": out.wer, **counts}
        # Corpus WER per group = total errors / total reference words.
        for group in (entry["category"], "all"):
            for k, v in counts.items():
                totals[group][k] += v
    groups = {}
    for group, t in totals.items():
        errors = t["substitutions"] + t["deletions"] + t["insertions"]
        groups[group] = {"wer": errors / max(t["ref_words"], 1), **t}
    return {"clips": clips, "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("hypotheses", type=Path, nargs="+", help="JSONL hypothesis files")
    parser.add_argument("--eval-dir", type=Path,
                        help="Local eval set (default: download references from the Hub)")
    parser.add_argument("--revision", default=EVAL_REVISION, help="Dataset revision")
    parser.add_argument("--json", type=Path, help="Also write scores as JSON")
    args = parser.parse_args()

    eval_dir, manifest = load_eval(args.eval_dir, args.revision)
    normalize = make_normalizers()
    results = {
        str(path): score(manifest, eval_dir, load_hypotheses(path), normalize)
        for path in args.hypotheses
    }

    labels = [p.stem for p in args.hypotheses]
    header = f"{'clip':30} {'lang':4} {'words':>6}  " + "  ".join(f"{l[:12]:>12}" for l in labels)
    print(header)
    print("-" * len(header))
    for entry in manifest:
        cells = [results[str(p)]["clips"].get(entry["name"]) for p in args.hypotheses]
        if not any(cells):
            continue
        words = next(c["ref_words"] for c in cells if c)
        wers = "  ".join(f"{c['wer'] * 100:11.2f}%" if c else f"{'—':>12}" for c in cells)
        print(f"{entry['name']:30} {entry['language']:4} {words:6}  {wers}")
    print("-" * len(header))
    for group in ("multilingual", "long-form", "multi-speaker", "all"):
        cells = [results[str(p)]["groups"].get(group) for p in args.hypotheses]
        if not any(cells):
            continue
        wers = "  ".join(f"{c['wer'] * 100:11.2f}%" if c else f"{'—':>12}" for c in cells)
        print(f"{group:30} {'':4} {'':6}  {wers}")

    missing = {
        p.stem: sorted({e["name"] for e in manifest} - set(results[str(p)]["clips"]))
        for p in args.hypotheses
    }
    for label, names in missing.items():
        if names:
            print(f"\n{label}: no hypothesis for {', '.join(names)}")

    if args.json:
        args.json.write_text(json.dumps({"revision": args.revision, "results": results}, indent=2))


if __name__ == "__main__":
    main()
