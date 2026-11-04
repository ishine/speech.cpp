# FASTDIAR encoder files preserve the streaming frontend's kernels and expand grouped downsampling

## Context

Microsoft FASTDIAR's causal ReDimNet2 generates normalized speaker embeddings, which its VAD and online clustering
turn into diarization. The training-time forward differs from StreamingInference in audio normalization and stem
context. The checkpoint holds the frontend's fixed STFT and mel kernels, BatchNorm statistics and training counters.
ReDimNet2 alternates depthwise residual convolutions, grouped strided downsampling and shared feature maps.
The grouped downsampling has channel multipliers and is not the depthwise operation exposed by ggml.

## Decision

- The GGUF holds every tensor read by StreamingInference, with the fixed frontend kernels and normalization
  constants. Only BatchNorm's training-only num_batches_tracked counters are omitted; evaluation reads running_mean
  and running_var instead. No inference head or statistics are discarded.
- Grouped downsampling kernels are expanded with exact zeros into ordinary dense convolution kernels. Residual
  depthwise kernels remain compact. The group count stays in metadata so the reader checks the configuration.
- Files are F32 only. Other storage types are not offered before stage checks establish their error. Python is used
  only for conversion and reference dumps, with a pinned official commit and checkpoint SHA-256.
- F, C, out_channels, block_1d_type, stages_setup, embed_dim, hop_length and time_upsampling are model properties,
  not request options. The pinned small and large checkpoints use pointwise fc context blocks. The checkpoint loader's
  model choice becomes a model path; its device becomes the network backend; its dtype is F32 for this layout.
- Audio normalization, pre-emphasis, STFT and causal mean subtraction run on the CPU; the backbone and projection
  run on the selected backend. This is explicit in the encoder interface, rather than a backend fallback.
- The encoder buffers incomplete input and computes fixed steps, four coarse frames by default. End-of-stream
  processes the remaining whole coarse frames and leaves a shorter final frame unused, as StreamingEncoder.flush
  does. Reset starts a new recording without reloading weights. A failed computation requires a reset.
- This file describes an encoder, not a complete diarizer. StreamingDiarizer's VAD and clustering parameters are not
  silently added as encoder options, and no incomplete FASTDIAR family is registered in the C API or model catalog.

The alternatives were turned down:

- Exporting the training forward. Its normalization and context differ from the streaming reference.
- Recreating the filterbank and STFT kernels from approximate constants. Their exact float32 values already travel
  in the official checkpoint, so reconstruction would introduce a second source of truth.
- Splitting every grouped convolution into one graph branch per group. It retains smaller weights but multiplies
  graph nodes and launches on the short live steps. Zero expansion uses existing backend operations and has no
  arithmetic approximation; the reader's layout states the expanded shape explicitly.
- Folding BatchNorm or softmax aggregation into weights in the file. Keeping the original parameters preserves the
  stage boundaries used to compare with the reference and avoids discarding inference state during this port.

## Consequences

The expanded downsampling tensors use more storage and computation than an eventual native grouped operation.
A future compact representation requires a layout update. The first reader is the fork's encoder implementation;
files are not uploaded or advertised as runnable by an existing released speech.cpp CLI.
