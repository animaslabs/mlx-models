"""
Shared harness for the cross-library benchmark: clip selection, timing and
the JSONL record every runner writes (mlx-voice-rs's `examples/eval.rs` writes
the same fields).

Standard library only, so the NeMo sub-project (its own Python) can import it.

Record per clip:
    {"name", "text", "library", "model", "streaming", "mode", "device", "dtype",
     "duration", "elapsed", "load_time", "chunk_ms_mean"?, "chunk_ms_p95"?}
A clip that raises writes {"name", ..., "error"} instead; a model the library
can't run writes a single {"library", "model", "unsupported"} line.
"""

import json
import math
import statistics
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

EVAL_DIR = Path(__file__).resolve().parent.parent / "samples" / "eval"


@dataclass
class Result:
    text: str
    # Wall time of each streaming step (seconds), when the runner streams.
    chunk_times: Optional[list[float]] = None


@dataclass
class Runner:
    device: str
    dtype: str
    transcribe: Callable[[Path], Result]
    extra: dict = field(default_factory=dict)
    # Untimed cleanup after every transcription, including failed ones.
    after_clip: Optional[Callable[[], None]] = None


class Unsupported(Exception):
    """Raised by a runner's `load` when the library can't run this model/mode."""


def load_clips(eval_dir: Path, languages: list[str], names: list[str],
               max_duration: Optional[float] = None) -> list[dict]:
    manifest = [json.loads(line) for line in (eval_dir / "manifest.jsonl").read_text().splitlines()
                if line.strip()]
    return [
        e for e in manifest
        if (not languages or e["language"] in languages)
        and (not names or e["name"] in names)
        and (max_duration is None or e["duration"] <= max_duration)
    ]


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def run_benchmark(args, library: str, load: Callable[[], Runner]) -> None:
    """Load the model, warm up, then time every selected clip into args.out."""
    clips = load_clips(args.eval_dir, args.language, args.clip, args.max_duration)
    if not clips:
        raise SystemExit("no clips selected")
    out = open(args.out, "w")
    base = {"library": library, "model": args.model_name,
            "streaming": args.streaming, "mode": args.mode}

    def write(row):
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()

    start = time.perf_counter()
    try:
        runner = load()
    except Unsupported as e:
        print(f"{library} / {args.model_name}: unsupported: {e}")
        write({**base, "unsupported": str(e)})
        return
    load_time = time.perf_counter() - start
    base |= {"device": runner.device, "dtype": runner.dtype, **runner.extra}
    print(f"{library} / {args.model_name}: loaded in {load_time:.2f}s on {runner.device} ({runner.dtype})")

    def timed(audio: Path) -> tuple[float, Result]:
        t0 = time.perf_counter()
        try:
            result = runner.transcribe(audio)
            return time.perf_counter() - t0, result
        finally:
            if runner.after_clip:
                runner.after_clip()

    first = args.eval_dir / clips[0]["audio"]
    for _ in range(args.warmup):
        timed(first)

    total_audio = total_elapsed = 0.0
    for entry in clips:
        audio = args.eval_dir / entry["audio"]
        try:
            elapsed, result = [], None
            for _ in range(args.repeats):
                secs, result = timed(audio)
                elapsed.append(secs)
        except Exception as e:
            traceback.print_exc()
            write({**base, "name": entry["name"], "duration": entry["duration"],
                   "error": f"{type(e).__name__}: {e}"})
            continue
        secs = statistics.median(elapsed)
        row = {**base, "name": entry["name"], "text": result.text.strip(),
               "duration": entry["duration"], "elapsed": round(secs, 3),
               "load_time": round(load_time, 2)}
        if result.chunk_times:
            ms = [t * 1000 for t in result.chunk_times]
            row |= {"chunk_ms_mean": round(statistics.fmean(ms), 2),
                    "chunk_ms_p95": round(p95(ms), 2)}
        write(row)
        total_audio += entry["duration"]
        total_elapsed += secs
        print(f"  {entry['name']:30} {entry['duration']:8.1f}s audio  {secs:7.2f}s  RTF {secs / entry['duration']:.4f}")
    if total_audio:
        print(f"  total {total_audio:.1f}s audio in {total_elapsed:.2f}s, RTF {total_elapsed / total_audio:.4f}")
