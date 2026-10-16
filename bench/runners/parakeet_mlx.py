"""parakeet-mlx: offline transcription of the whole file (no chunking)."""

from pathlib import Path

from common import Result, Runner, Unsupported


def load(args) -> Runner:
    import mlx.core as mx
    from parakeet_mlx.utils import from_pretrained

    if args.streaming:
        # transcribe_stream is local-attention streaming, not NeMo's cache-aware
        # chunked_limited scheme, so it wouldn't be the same model behaviour.
        raise Unsupported("no cache-aware streaming")
    dtype = getattr(mx, args.dtype or "bfloat16")
    try:
        model = from_pretrained(args.model, dtype=dtype)
    except Exception as e:
        raise Unsupported(f"load failed: {type(e).__name__}: {e}")

    import soundfile as sf
    from parakeet_mlx.audio import get_logmel

    sample_rate = model.preprocessor_config.sample_rate

    def transcribe(path: Path) -> Result:
        # Same as model.transcribe(path) without chunking, but decodes with
        # soundfile instead of shelling out to ffmpeg. Eval clips are 16 kHz mono.
        samples, sr = sf.read(path, dtype="float32")
        if sr != sample_rate:
            raise ValueError(f"{path.name}: expected {sample_rate} Hz, got {sr}")
        mel = get_logmel(mx.array(samples), model.preprocessor_config)
        return Result(model.generate(mel)[0].text)

    return Runner(device="metal", dtype=str(dtype).removeprefix("mlx.core."), transcribe=transcribe)
