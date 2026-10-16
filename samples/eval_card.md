---
pretty_name: ASR eval set (multilingual, long-form, multi-speaker)
task_categories:
  - automatic-speech-recognition
language: [en, de, fr, es, it, pt, nl, pl]
license: other
license_name: mixed
viewer: false
---

# ASR eval set

A small evaluation set for checking ASR ports (mlx-voice-rs, mlx-models)
against the same audio and references. Built by
`samples/fetch_eval.py` in mlx-models; every clip is 16 kHz mono 16-bit WAV
with a plain-text reference next to it.

**This repository is private.** Some clips come from sources that may not be
redistributed (see the licenses below). Don't make it public without removing
them.

## Layout

```
manifest.jsonl     one JSON object per clip
<name>.wav         audio
<name>.txt         reference transcript
```

Manifest fields: `name`, `audio`, `reference`, `duration` (seconds),
`category` (`multilingual` | `long-form` | `multi-speaker`), `language`,
`speakers`, `source`, `license`, and for MLS clips the joined `utterances`.

## Clips

| Category | Clips | Source | License |
|----------|-------|--------|---------|
| Multilingual (~90 s) | `mls_{de,fr,es,it,pt,nl,pl}` | [Multilingual LibriSpeech](https://www.openslr.org/94/) test, consecutive utterances from one speaker/chapter joined with 0.4 s gaps | CC BY 4.0 |
| Long-form (~6 min) | `ted_gary_flake`, `ted_robert_gupta` | [TED-LIUM 3](https://www.openslr.org/51/) via `distil-whisper/tedlium-long-form` | CC BY-NC-ND 3.0 |
| Multi-speaker (18–22 min) | `earnings21_cumulus_media`, `earnings21_zagg` | [Earnings-21](https://github.com/revdotcom/speech-datasets) via `distil-whisper/earnings21` | CC BY-SA 4.0 |
| Multi-speaker (3–22 min) | `rev16_crimetown_trailer`, `rev16_the_daily_wildfires` | [Rev16](https://github.com/openai/whisper) podcast subset via `distil-whisper/rev16` | research use only |

References keep each source's own formatting: MLS is lower-case without
punctuation, the others are cased and punctuated. Normalize both sides before
scoring (mlx-models `samples/score.py` does this).

## Citations

- V. Pratap et al., "MLS: A Large-Scale Multilingual Dataset for Speech
  Research," Interspeech 2020.
- F. Hernandez et al., "TED-LIUM 3: twice as much data and corpus repartition
  for experiments on speaker adaptation," SPECOM 2018.
- M. Del Rio et al., "Earnings-21: A Practical Benchmark for ASR in the Wild,"
  Interspeech 2021.
- A. Radford et al., "Robust Speech Recognition via Large-Scale Weak
  Supervision," 2022 (Rev16 subset).
