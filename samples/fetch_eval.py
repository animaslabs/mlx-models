#!/usr/bin/env python3
"""
Download an evaluation set of audio clips with reference transcripts.

Covers three axes the single bundled sample doesn't:
  - languages:      Multilingual LibriSpeech test (de, fr, es, it, pt, nl, pl),
                    consecutive utterances from one speaker joined to ~90 s
  - long-form:      full TED talks from TED-LIUM (~6 min, single speaker)
  - multi-speaker:  earnings calls (Earnings-21) and podcasts (Rev16), 3-22 min

Every clip is written as 16 kHz mono WAV next to a .txt reference, plus a
manifest.jsonl describing all of them. Files go to samples/eval/ (gitignored;
several sources are non-redistributable, so the audio isn't committed).

Usage:
    uv run samples/fetch_eval.py
    uv run samples/fetch_eval.py --only mls
    uv run samples/fetch_eval.py --mls-seconds 60
    uv run samples/fetch_eval.py --push animaslabs/asr-eval
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from huggingface_hub import HfApi, HfFileSystem

SAMPLE_RATE = 16000
SILENCE_BETWEEN_UTTERANCES = 0.4  # seconds

MLS_LANGUAGES = {
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
    "pl": "polish",
}

# (clip name, dataset, parquet glob, key column, key value, metadata)
LONG_FORM = [
    ("ted_gary_flake", "distil-whisper/tedlium-long-form", "data/test-*.parquet",
     "speaker_id", "GaryFlake",
     {"category": "long-form", "speakers": 1, "license": "CC-BY-NC-ND-3.0"}),
    ("ted_robert_gupta", "distil-whisper/tedlium-long-form", "data/test-*.parquet",
     "speaker_id", "RobertGupta",
     {"category": "long-form", "speakers": 1, "license": "CC-BY-NC-ND-3.0"}),
    ("earnings21_cumulus_media", "distil-whisper/earnings21", "full/test-*.parquet",
     "file_id", "4386541",
     {"category": "multi-speaker", "speakers": 5, "license": "CC-BY-SA-4.0"}),
    ("earnings21_zagg", "distil-whisper/earnings21", "full/test-*.parquet",
     "file_id", "4387332",
     {"category": "multi-speaker", "speakers": 6, "license": "CC-BY-SA-4.0"}),
    ("rev16_crimetown_trailer", "distil-whisper/rev16", "full/test-*.parquet",
     "file_number", "14",
     {"category": "multi-speaker", "speakers": None, "license": "research-only"}),
    ("rev16_the_daily_wildfires", "distil-whisper/rev16", "full/test-*.parquet",
     "file_number", "6",
     {"category": "multi-speaker", "speakers": None, "license": "research-only"}),
]

TEXT_COLUMNS = ("transcription", "text", "transcript")


def decode(audio_bytes: bytes) -> np.ndarray:
    """Decode any libsndfile-readable audio to 16 kHz mono float32."""
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        mono = soxr.resample(mono, sr, SAMPLE_RATE)
    return mono


def write_clip(out_dir: Path, name: str, audio: np.ndarray, text: str, meta: dict) -> dict:
    wav_path = out_dir / f"{name}.wav"
    sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
    (out_dir / f"{name}.txt").write_text(text.strip() + "\n")
    entry = {
        "name": name,
        "audio": wav_path.name,
        "reference": f"{name}.txt",
        "duration": round(len(audio) / SAMPLE_RATE, 2),
        **meta,
    }
    print(f"  {name}: {entry['duration']:.0f}s, {len(text.split())} words")
    return entry


def fetch_long_form(fs: HfFileSystem, out_dir: Path) -> list[dict]:
    entries = []
    for name, dataset, pattern, key_col, key_val, meta in LONG_FORM:
        for path in sorted(fs.glob(f"datasets/{dataset}/{pattern}")):
            pf = pq.ParquetFile(fs.open(path))
            found = None
            for rg in range(pf.num_row_groups):
                keys = pf.read_row_group(rg, columns=[key_col]).column(0).to_pylist()
                if str(key_val) in map(str, keys):
                    found = (rg, list(map(str, keys)).index(str(key_val)))
                    break
            if found is None:
                continue
            rg, idx = found
            row = pf.read_row_group(rg).slice(idx, 1).to_pylist()[0]
            text = next(row[c] for c in TEXT_COLUMNS if c in row)
            audio = decode(row["audio"]["bytes"])
            meta = {**meta, "language": "en", "source": dataset}
            if row.get("unique_speakers"):
                meta["speakers"] = int(row["unique_speakers"])
            entries.append(write_clip(out_dir, name, audio, text, meta))
            break
        else:
            print(f"  {name}: NOT FOUND in {dataset}")
    return entries


def fetch_mls(fs: HfFileSystem, out_dir: Path, target_seconds: float) -> list[dict]:
    entries = []
    gap = np.zeros(int(SILENCE_BETWEEN_UTTERANCES * SAMPLE_RATE), dtype=np.float32)
    for lang, folder in MLS_LANGUAGES.items():
        path = f"datasets/facebook/multilingual_librispeech/{folder}/test-00000-of-00001.parquet"
        pf = pq.ParquetFile(fs.open(path))
        # Utterances from one speaker/chapter, in reading order, so the joined
        # clip is continuous narration rather than unrelated sentences.
        rows = pf.read_row_group(0).to_pylist()
        speaker, chapter = rows[0]["speaker_id"], rows[0]["chapter_id"]
        rows = sorted(
            (r for r in rows if r["speaker_id"] == speaker and r["chapter_id"] == chapter),
            key=lambda r: (r["original_path"], r["begin_time"]),
        )
        pieces, texts, total = [], [], 0.0
        for r in rows:
            if total >= target_seconds:
                break
            pieces += [decode(r["audio"]["bytes"]), gap]
            texts.append(r["transcript"])
            total += r["audio_duration"] + SILENCE_BETWEEN_UTTERANCES
        audio = np.concatenate(pieces[:-1])
        meta = {
            "category": "multilingual",
            "language": lang,
            "speakers": 1,
            "source": "facebook/multilingual_librispeech",
            "license": "CC-BY-4.0",
            "utterances": [r["id"] for r in rows[: len(texts)]],
        }
        entries.append(write_clip(out_dir, f"mls_{lang}", audio, " ".join(texts), meta))
    return entries


def push(out_dir: Path, repo_id: str):
    """Upload the eval set as a private dataset; prints the commit to pin."""
    api = HfApi()
    # private=True only applies on creation; an existing repo keeps its visibility.
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        raise SystemExit(f"{repo_id} is public; refusing to upload non-redistributable clips")
    api.upload_file(
        path_or_fileobj=Path(__file__).parent / "eval_card.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    commit = api.upload_folder(
        folder_path=out_dir,
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["manifest.jsonl", "*.wav", "*.txt"],
        commit_message="Update eval clips",
    )
    print(f"Pushed to {repo_id} at revision {commit.oid}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--out-dir", type=Path, default=Path(__file__).parent / "eval")
    parser.add_argument("--only", choices=["mls", "long"], help="Fetch only one group")
    parser.add_argument("--mls-seconds", type=float, default=90.0,
                        help="Target length of each joined MLS clip (default: 90)")
    parser.add_argument("--push", metavar="REPO_ID",
                        help="Upload the existing out-dir to this private HF dataset "
                             "(e.g. animaslabs/asr-eval) instead of fetching")
    args = parser.parse_args()

    if args.push:
        push(args.out_dir, args.push)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"
    existing = {}
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            entry = json.loads(line)
            existing[entry["name"]] = entry

    fs = HfFileSystem()
    if args.only != "long":
        print("Multilingual LibriSpeech:")
        existing.update({e["name"]: e for e in fetch_mls(fs, args.out_dir, args.mls_seconds)})
    if args.only != "mls":
        print("Long-form / multi-speaker:")
        existing.update({e["name"]: e for e in fetch_long_form(fs, args.out_dir)})

    manifest_path.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in existing.values())
    )
    print(f"Wrote {len(existing)} clips to {args.out_dir}")


if __name__ == "__main__":
    main()
