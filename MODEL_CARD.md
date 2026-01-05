---
license: cc-by-4.0
track_downloads: true
language:
- en
- es
- fr
- de
- bg
- hr
- cs
- da
- nl
- et
- fi
- el
- hu
- it
- lv
- lt
- mt
- pl
- pt
- ro
- sk
- sl
- sv
- ru
- uk

pipeline_tag: automatic-speech-recognition
library_name: mlx
datasets:
- nvidia/Granary
- nemo/asr-set-3.0
tags:
- automatic-speech-recognition
- speech
- audio
- Transducer
- TDT
- FastConformer
- Conformer
- pytorch
- Parakeet
base_model: nvidia/parakeet-tdt-0.6b-v3
---

# **animaslabs/parakeet-tdt-0.6b-v3-mlx**

This model was converted to MLX format from [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) using the scripts in this [github repo](https://github.com/animaslabs/mlx-models). Please refer to [original model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) for more details on the model.

## Use with mlx

NOTE: parakeet-mlx and mlx-audio do not support quantized models.

### parakeet-mlx

```bash
pip install -U parakeet-mlx
```

```bash
parakeet-mlx audio.wav --model mlx-community/parakeet-tdt-0.6b-v3
```

### mlx-audio

```bash
pip install -U mlx-audio
```

```bash
python -m mlx_audio.stt.generate --model mlx-community/parakeet-tdt-0.6b-v3 --audio audio.wav --output somewhere
```

