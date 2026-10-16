# Pointwise Conv Quantization

Last updated 2026-10-16.

Quantizing the pointwise convs cuts a parakeet 0.6B 4-bit file from 594 MiB to 347 MiB, and the trick behind it is exact: a kernel-1 Conv1d computes the same function as a Linear layer. Only the quantization adds error, the same error every other quantized Linear layer already has. The catch is loading: parakeet-mlx and mlx-audio build these layers as `nn.Conv1d`, which can't hold quantized weights. So the converter now leaves them unquantized for model families those runtimes support, and quantizes them only for families that run in mlx-voice-rs alone.

## Where pointwise convs sit

Each of the 24 Conformer layers in parakeet 0.6B has a convolution module with two pointwise convs wrapped around one depthwise conv. Together they hold 3.1M weights per layer, 75.5M across the encoder.

1. `pointwise_conv1` expands each frame from 1024 to 2048 channels, and a GLU halves it back to 1024.
2. `depthwise_conv` mixes over time: each channel gets its own kernel spanning several neighbouring frames.
3. `pointwise_conv2` mixes channels again, 1024 to 1024.

| Layer | Kernel size | Weight shape, MLX layout [out, kernel, in] | Weights per layer |
| --- | --- | --- | --- |
| `pointwise_conv1` | 1 | [2048, 1, 1024] | 2,097,152 |
| `pointwise_conv2` | 1 | [1024, 1, 1024] | 1,048,576 |

The depthwise conv is small (one short kernel per channel) and is never quantized. The pointwise convs are the big ones, and "pointwise" means kernel size 1: they look at one frame at a time and only mix channels.

## Why a kernel-1 conv is a Linear layer

With kernel size 1, the convolution's sum over kernel positions has a single term, and what remains is exactly a Linear layer's formula. Take input x with T frames and I channels, and weights W of shape [O, I, K]. A 1D convolution (stride 1, no padding) computes, for each output frame t and output channel o:

```math
y_{t,o} = b_o + \sum_{k=0}^{K-1} \sum_{i=0}^{I-1} W_{o,i,k} \, x_{t+k,\,i}
```

Set K = 1. The sum over k has one term, k = 0, so the frame index is just t:

```math
y_{t,o} = b_o + \sum_{i=0}^{I-1} W_{o,i,0} \, x_{t,i}
```

A Linear layer with weight matrix W' of shape [O, I], applied to every frame, computes:

```math
y_{t,o} = b_o + \sum_{i=0}^{I-1} W'_{o,i} \, x_{t,i}
```

These are the same equation when W' is W with its size-1 kernel axis dropped. In matrix form, the conv is y = x W'ᵀ + b, applied to the [T, I] input in one matmul. Stride, padding and dilation don't change this: with K = 1 there is no window to stride over, pad or dilate, and parakeet-mlx builds these layers with stride 1 and padding 0 anyway. When a model's convs have no bias, drop b from both sides.

Dropping the axis is a reshape, not a numeric operation. Every weight value is kept bit for bit, so before any quantization the two layers give identical outputs.

## How the converter quantizes them

The converter squeezes each pointwise weight to 2D, then hands it to the same `mx.quantize` path every Linear weight goes through. In `convert_nemo.py`:

1. **Squeeze.** The PyTorch weight [O, I, 1] becomes [O, I] by dropping the kernel axis (`arr.squeeze(axis=2)`). It skips the Conv1d transpose, since a Linear weight is already [out, in].
2. **Group.** Each row of I = 1024 input weights is split into groups of 64 consecutive values, so 16 groups per row.
3. **Fit a scale and offset per group.** MLX affine quantization stores each weight as a b-bit integer q in [0, 2^b − 1], with a per-group scale s and offset β (MLX calls it the bias) chosen so the integers span the group's min to max.
4. **Pack.** Eight 4-bit integers (or four 8-bit ones) are packed into each uint32, so [2048, 1024] becomes [2048, 128] uint32 at 4 bits.
5. **Store.** The file gets `pointwise_conv1.weight` (packed uint32), plus `pointwise_conv1.scales` and `pointwise_conv1.biases`, each [2048, 16] float16.

Each weight is reconstructed as:

```math
\hat{w} = s_g \, q + \beta_g, \qquad q \in \{0, 1, \dots, 2^b - 1\}
```

where g is the weight's group. At inference a runtime that builds these layers as `QuantizedLinear` calls `mx.quantized_matmul`, which dequantizes groups on the fly inside the matmul. The full float matrix is never materialized.

## How much space it saves

Left unquantized, the pointwise convs take 288 MiB as float32, 48% of a 4-bit tdt file. At 4 bits they shrink to 40.5 MiB, an 86% cut. The overhead of one float16 scale and one float16 offset per 64 weights adds 0.5 bits per weight, so 4-bit costs 4.5 bits and 8-bit costs 8.5.

| Pointwise storage | Bits per weight | Per layer (MiB) | 24 layers (MiB) |
| --- | --- | --- | --- |
| float32 (what tdt, rnnt, ctc use now) | 32 | 12.0 | 288.0 |
| float16 | 16 | 6.0 | 144.0 |
| 8-bit quantized | 8.5 | 3.19 | 76.5 |
| 4-bit quantized | 4.5 | 1.69 | 40.5 |

Per layer is 3,145,728 weights × bits per weight. The float32 and 4-bit totals match the tensor sizes read from the safetensors headers of the tdt and nemotron 4-bit files.

Measured whole-file sizes for parakeet-tdt-0.6b-v3 at 4 bits, both from the current converter:

- Pointwise unquantized (the new default for tdt): 594.3 MiB
- Pointwise quantized (`--pointwise quantized`, the `-pw` repos): 346.8 MiB, 42% smaller

At 8 bits the same switch would take the file from about 866 MiB to about 655 MiB. That figure is computed from the table above, not measured.

## What is exact and what isn't

The reshape is exact; the quantization is not, and its error is the same kind every quantized Linear layer already carries.

- **Exact:** treating the conv as a Linear layer. Squeezing [O, I, 1] to [O, I] moves no values, and the two layers compute the same function (see the derivation above).
- **Lossy:** rounding to b bits. Each weight lands on the nearest of 2^b levels in its group, so it moves by at most half a step, s / 2. With 64 weights per group, one outlier widens the step for the other 63.
- **Not measured yet:** how much accuracy the pointwise layers specifically lose. The 2026-10-14 compatibility check passed the nemotron 4-bit and 8-bit repos in mlx-voice-rs, but no WER comparison of pointwise quantized against unquantized has been run.

The same pass stores LSTM weights as float16, a separate and much smaller rounding. A rebuilt tdt 4-bit with float16 LSTM gives the same transcript of the test sample as the published build in both parakeet-mlx and mlx-audio (first 120 characters compared).

## Why runtimes disagree, and the flag

The math is the same, but a runtime must build the layer as a Linear for the quantized weights to load, and parakeet-mlx and mlx-audio build it as `nn.Conv1d`. Their loading sequence is: build the model, call `nn.quantize()`, then load the weights. `nn.quantize()` only swaps layers that have a quantized form, and in mlx 0.32.3 `Conv1d` has none. So the pointwise layers stay float convs expecting a [O, 1, I] weight, and the file hands them packed uint32 plus scales and offsets.

| Runtime | Builds pointwise as | Pointwise quantized | Pointwise unquantized |
| --- | --- | --- | --- |
| parakeet-mlx 0.5.3 | `nn.Conv1d` | Fails: "Received 96 parameters not in model" | Loads and transcribes |
| mlx-audio 0.5.9 | `nn.Conv1d` | Fails: "Invalid weight array with 2 dimensions for 1D convolution" | Loads and transcribes |
| mlx-voice-rs 8a5b440 | Accepts both layouts (source not inspected) | Loads (nemotron repos, per the README check) | Loads (rnnt repos, per the README check) |

The 96 unexpected parameters are the scales and offsets: 24 layers × 2 pointwise convs × 2 tensors.

## Naming: the `-pw` variants

Both layouts are published as separate repos, and the name says which one you're getting. The rule is the same for every model family:

- **No suffix** (`-mlx-4bit`, `-mlx-8bit`): pointwise convs unquantized. Loads in every runtime that supports the model.
- **`-pw` suffix** (`-mlx-4bit-pw`, `-mlx-8bit-pw`): pointwise convs quantized as linear layers. Smaller, but only mlx-voice-rs loads it.

`main.py --pointwise` picks which to build: `unquantized` (the default), `quantized` or `both`. Full-precision repos have no pointwise variant.

Rebuilt nemotron-speech-streaming-en-0.6b on 2026-10-16 with `--pointwise both`, transcribing the sample:

| Variant | Size (MiB) | mlx-audio 0.5.9 | mlx-voice-rs 271f965 |
| --- | --- | --- | --- |
| `-mlx` | 2358 | Loads and transcribes | Loads and transcribes |
| `-mlx-8bit` | 845 | Loads and transcribes | Loads and transcribes |
| `-mlx-8bit-pw` | 633 | Fails: "Received 96 parameters not in model" | Loads and transcribes |
| `-mlx-4bit` | 589 | Loads and transcribes | Loads and transcribes |
| `-mlx-4bit-pw` | 342 | Fails: "Received 96 parameters not in model" | Loads and transcribes |

Storing the unquantized pointwise weights as float16 would save another 144 MiB and should still load as `nn.Conv1d`, but that hasn't been tried.
