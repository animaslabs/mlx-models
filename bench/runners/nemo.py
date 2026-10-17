"""
NVIDIA NeMo (PyTorch): offline `transcribe`, or cache-aware streaming with
`conformer_stream_step`, following NeMo's
examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py.

Runs on MPS when available (override with BENCH_NEMO_DEVICE=cpu).

Prompt-conditioned multilingual models (nemotron-3.5-asr-streaming) take
--target-lang (default auto). Their language tags (e.g. <en-US>) are stripped
from the text, like NeMo's strip_lang_tags.

Offline NeMo runs full attention over the whole file, so its memory grows with
the square of the clip length: about 13 GB for 6 min, 69 GB for 18 min, and
around 100 GB for the 22 min clips. PyTorch's MPS allocator lets a process grow
to 1.7x the GPU working set by default, which is more than physical RAM, so a
long clip would drag the whole machine into swap. Instead the process gets a
hard budget, 60% of RAM by default or BENCH_NEMO_MEMORY_GB. A clip that doesn't
fit fails fast with an MPS out-of-memory error, which the harness records as
that clip's error.
"""

import gc
import os
import re
import time
from pathlib import Path

from common import Result, Runner, Unsupported


# NeMo's default strip_lang_tags pattern (rnnt_decoding.py)
LANG_TAG = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>")


def _text(hyp) -> str:
    if isinstance(hyp, (list, tuple)):  # some versions return (best, all) or nested lists
        hyp = hyp[0]
    text = hyp if isinstance(hyp, str) else hyp.text
    return LANG_TAG.sub("", text).strip()


def _limit_mps_memory(torch) -> dict:
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    budget = float(os.environ.get("BENCH_NEMO_MEMORY_GB", 0)) * 2**30 or 0.6 * ram
    torch.mps.set_per_process_memory_fraction(min(budget / torch.mps.recommended_max_memory(), 2.0))
    print(f"nemo: MPS memory budget {budget / 2**30:.0f} GB of {ram / 2**30:.0f} GB RAM")
    return {"memory_budget_gb": round(budget / 2**30)}


def _release(torch, device) -> None:
    # Return cached blocks after each clip, so a long clip (or a failed one)
    # doesn't leave tens of GB reserved while the next one runs.
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def load(args) -> Runner:
    import torch
    from nemo.collections.asr.models import ASRModel

    device = os.environ.get("BENCH_NEMO_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    extra = _limit_mps_memory(torch) if device == "mps" else {}
    if args.dtype not in (None, "float32"):
        raise Unsupported(f"benchmark runs NeMo in float32 only, not {args.dtype}")
    try:
        model = ASRModel.restore_from(args.model, map_location=torch.device(device))
    except Exception as e:
        raise Unsupported(f"load failed: {type(e).__name__}: {e}")
    model.eval()

    prompt = {}
    if hasattr(model, "set_inference_prompt"):
        target_lang = args.target_lang or "auto"
        model.set_inference_prompt(target_lang)  # streaming skips the prompt without it
        prompt = {"target_lang": target_lang}
        extra["target_lang"] = target_lang
    elif args.target_lang:
        raise Unsupported("model has no language prompt")

    if not args.streaming:
        def transcribe(path: Path) -> Result:
            with torch.inference_mode():
                out = model.transcribe([str(path)], batch_size=1, verbose=False, **prompt)
            return Result(_text(out[0] if isinstance(out, list) else out))

        return Runner(device=device, dtype="float32", transcribe=transcribe, extra=extra,
                      after_clip=lambda: _release(torch, device))

    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    encoder = model.encoder
    if not hasattr(encoder, "set_default_att_context_size"):
        raise Unsupported("encoder has no cache-aware streaming")
    acs = encoder.att_context_size_all if hasattr(encoder, "att_context_size_all") else [encoder.att_context_size]
    att = list(acs[args.mode])
    encoder.set_default_att_context_size(att)
    extra["att_context_size"] = att

    def transcribe(path: Path) -> Result:
        steps = []
        with torch.inference_mode():
            buffer = CacheAwareStreamingAudioBuffer(
                model=model, online_normalization=False, pad_and_drop_preencoded=False)
            buffer.append_audio_file(str(path), stream_id=-1)
            cache_channel, cache_time, cache_channel_len = encoder.get_initial_cache_state(batch_size=1)
            hyps, pred_out, texts = None, None, [""]
            t = time.perf_counter()
            for step, (chunk, chunk_len) in enumerate(iter(buffer)):
                (pred_out, texts, cache_channel, cache_time, cache_channel_len, hyps) = model.conformer_stream_step(
                    processed_signal=chunk,
                    processed_signal_length=chunk_len,
                    cache_last_channel=cache_channel,
                    cache_last_time=cache_time,
                    cache_last_channel_len=cache_channel_len,
                    keep_all_outputs=buffer.is_buffer_empty(),
                    previous_hypotheses=hyps,
                    previous_pred_out=pred_out,
                    drop_extra_pre_encoded=0 if step == 0 else encoder.streaming_cfg.drop_extra_pre_encoded,
                    return_transcription=True,
                )
                if device == "mps":
                    torch.mps.synchronize()
                now = time.perf_counter()
                steps.append(now - t)
                t = now
        return Result(_text(texts[0]) if texts else "", steps)

    return Runner(device=device, dtype="float32", transcribe=transcribe, extra=extra,
                  after_clip=lambda: _release(torch, device))
