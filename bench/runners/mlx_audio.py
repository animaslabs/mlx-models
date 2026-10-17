"""mlx-audio: offline Parakeet, or cache-aware streaming for Nemotron."""

import time
from pathlib import Path

from common import Result, Runner, Unsupported


def load(args) -> Runner:
    import mlx.core as mx
    from mlx_audio.stt import load as load_stt

    try:
        model = load_stt(args.model)
    except Exception as e:
        raise Unsupported(f"load failed: {type(e).__name__}: {e}")
    # Multilingual Nemotron: language prompt key such as "en-US" or "auto".
    lang = {"language": args.target_lang} if args.target_lang else {}

    if args.streaming:
        if not hasattr(model, "stream_generate") or not hasattr(model, "default_att_context_size"):
            raise Unsupported("no cache-aware streaming for this model type")
        acs = model.encoder_config.att_context_size
        att = acs[args.mode] if isinstance(acs[0], list) else acs
        dtype = getattr(mx, args.dtype or "float32")

        def transcribe(path: Path) -> Result:
            # stream_generate yields once per native encoder chunk; time each step.
            steps, result = [], None
            t = time.perf_counter()
            for result in model.stream_generate(path, att_context_size=att, dtype=dtype, **lang):
                now = time.perf_counter()
                steps.append(now - t)
                t = now
            return Result(result.text if result else "", steps)

        return Runner(device="metal", dtype=str(dtype).removeprefix("mlx.core."),
                      transcribe=transcribe, extra={"att_context_size": att, "target_lang": args.target_lang})

    dtype = getattr(mx, args.dtype or "bfloat16")
    kwargs = {"dtype": dtype, **lang}
    if hasattr(model, "default_att_context_size"):
        # Nemotron's generate chunks to 30s by default; run the whole file like the rest.
        kwargs["chunk_duration"] = None

    def transcribe(path: Path) -> Result:
        return Result(model.generate(path, **kwargs).text)

    return Runner(device="metal", dtype=str(dtype).removeprefix("mlx.core."), transcribe=transcribe)
