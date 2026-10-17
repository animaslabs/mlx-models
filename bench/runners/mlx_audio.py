"""mlx-audio: offline Parakeet, or cache-aware streaming for Nemotron."""

import time
from pathlib import Path

from common import Result, Runner, Unsupported


def compute_dtype(model, input_dtype) -> str:
    """The dtype the model actually computes in.

    `load_stt` keeps the checkpoint's weights as they are (float32 for the
    unquantized conversions) and `generate(dtype=...)` only casts the audio, so
    float32 weights promote everything back to float32 after the first layer.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    dtypes = {v.dtype for _, v in tree_flatten(model.parameters()) if mx.issubdtype(v.dtype, mx.floating)}
    computed = mx.float32 if mx.float32 in dtypes else input_dtype
    return str(computed).removeprefix("mlx.core.")


def load(args) -> Runner:
    import mlx.core as mx
    from mlx_audio.stt import load as load_stt

    try:
        model = load_stt(args.model)
    except Exception as e:
        raise Unsupported(f"load failed: {type(e).__name__}: {e}")

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
            for result in model.stream_generate(path, att_context_size=att, dtype=dtype):
                now = time.perf_counter()
                steps.append(now - t)
                t = now
            return Result(result.text if result else "", steps)

        return Runner(device="metal", dtype=compute_dtype(model, dtype),
                      transcribe=transcribe, extra={"att_context_size": att,
                                                    "input_dtype": str(dtype).removeprefix("mlx.core.")})

    dtype = getattr(mx, args.dtype or "bfloat16")
    kwargs = {"dtype": dtype}
    if hasattr(model, "default_att_context_size"):
        # Nemotron's generate chunks to 30s by default; run the whole file like the rest.
        kwargs["chunk_duration"] = None

    def transcribe(path: Path) -> Result:
        return Result(model.generate(path, **kwargs).text)

    return Runner(device="metal", dtype=compute_dtype(model, dtype), transcribe=transcribe,
                  extra={"input_dtype": str(dtype).removeprefix("mlx.core.")})
