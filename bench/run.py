#!/usr/bin/env python3
"""
Run one library on the eval set and write timed hypotheses as JSONL.

MLX libraries run in the root env; NeMo runs in its own (Python 3.12) project:
    uv run bench/run.py --library parakeet-mlx --model animaslabs/parakeet-rnnt-0.6b-mlx --out x.jsonl
    uv run --project bench/nemo bench/run.py --library nemo --model path/to/model.nemo --streaming --out y.jsonl

Usually driven by bench/bench.py, which runs every library and builds the report.
"""

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import EVAL_DIR, run_benchmark  # noqa: E402

RUNNERS = {"parakeet-mlx": "parakeet_mlx", "mlx-audio": "mlx_audio", "nemo": "nemo"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--library", required=True, choices=RUNNERS)
    parser.add_argument("--model", required=True, help="HF id, MLX directory or .nemo path")
    parser.add_argument("--model-name", help="Label in the output (default: model basename)")
    parser.add_argument("--streaming", action="store_true", help="Cache-aware streaming")
    parser.add_argument("--mode", type=int, default=0, help="Index into encoder.att_context_size")
    parser.add_argument("--target-lang", help="Language prompt of a multilingual model (e.g. en-US, auto)")
    parser.add_argument("--dtype", help="float32 / bfloat16 (default: the library's default)")
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--language", action="append", default=[], help="Filter by language (repeatable)")
    parser.add_argument("--clip", action="append", default=[], help="Only this clip (repeatable)")
    parser.add_argument("--max-duration", type=float, help="Skip clips longer than this (seconds)")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed runs on the first clip")
    parser.add_argument("--repeats", type=int, default=1, help="Timed runs per clip (median)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.model_name = args.model_name or Path(args.model.rstrip("/")).name.removesuffix(".nemo")

    runner = importlib.import_module(f"runners.{RUNNERS[args.library]}")
    run_benchmark(args, args.library, lambda: runner.load(args))


if __name__ == "__main__":
    main()
