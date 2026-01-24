import json
import mlx.nn as nn
from huggingface_hub import hf_hub_download
from parakeet_mlx.utils import from_config

# Download and load config
config_path = hf_hub_download("animaslabs/nemotron-speech-streaming-en-0.6b-mlx", "config.json")
with open(config_path) as f:
    config = json.load(f)
print(config['encoder'])
config["encoder"]["att_context_size"] = [70,13]

# Build model and apply quantization structure
model = from_config(config)
nn.quantize(
    model,
    bits=config["quantization"]["bits"],
    group_size=config["quantization"]["group_size"],
)

# Load quantized weights
weights_path = hf_hub_download("animaslabs/nemotron-speech-streaming-en-0.6b-mlx", "model.safetensors")
model.load_weights(weights_path)

# Transcribe
result = model.transcribe("Speaker26_004_16k.wav")
print(result.text)

