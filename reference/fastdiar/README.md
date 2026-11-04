# FASTDIAR

`speech diarize` and `speech asr --diarize` run FASTDIAR from one float32 GGUF: the streaming ReDimNet2 encoder of
[microsoft/fastdiar](https://github.com/microsoft/fastdiar) at `f48bff53b9a6a0cce4539e94b6def95f609eaea4`, its VAD's
hysteresis and streaming clustering, and Silero VAD's weights, which the file carries so that one model is all a caller
names. The frontend, the VAD and the clustering run on the CPU; the backbone and projection on the selected backend.

FASTDIAR has no most speakers, which `speech info` shows as any number and `speech_model_info_speakers()` as 0. It gives
one speaker per frame, so overlapped speech counts against it. A live stream gives its turns in pieces and never reports
a speaker under way through `speech_diarization_speaking()`; whole-file diarization gives whole turns.

## Convert and check an encoder

The reference is [microsoft/fastdiar](https://github.com/microsoft/fastdiar) at
`f48bff53b9a6a0cce4539e94b6def95f609eaea4`. Its MIT license and third-party notice are retained here.
Download a checkpoint from that repository's `v0.1.0` release. The scripts verify its full SHA-256 before loading it.
Python and its locked environment are used for conversion and reference generation only.
The official environment's silero-vad 6.2.1 and speech.cpp's 6.2.3 ship the same `silero_vad.jit` (SHA-256 `e1122837...`), so
speech.cpp's Silero GGUF is carried unchanged.

```sh
git clone https://github.com/microsoft/fastdiar /tmp/fastdiar
git -C /tmp/fastdiar checkout f48bff53b9a6a0cce4539e94b6def95f609eaea4
uv run --project reference/fastdiar reference/fastdiar/convert.py small \
  --source /tmp/fastdiar --checkpoint /path/to/fastdiar-small-90be5da5.pt \
  --silero /path/to/silero-vad-309K-v6.2-F32.gguf --out models/FASTDIAR-small-2.4M-F32.gguf
uv run --project reference/fastdiar reference/fastdiar/encoder-dump.py small \
  --source /tmp/fastdiar --checkpoint /path/to/fastdiar-small-90be5da5.pt \
  --out reference/fastdiar/out/encoder-small --audio /path/to/public-16khz-mono.wav
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target fastdiar-encoder-check -j4
build/fastdiar-encoder-check models/FASTDIAR-small-2.4M-F32.gguf reference/fastdiar/out/encoder-small/speech cpu
build/fastdiar-encoder-check models/FASTDIAR-small-2.4M-F32.gguf reference/fastdiar/out/encoder-small/speech gpu
```

Use `large` and `fastdiar-large-f6d71444.pt` for the large checkpoint. The dump also makes `noise`, `silence` and
`tiny` cases. With `--whole`, it saves the entire supplied audio and reference embeddings without intermediate
stages, for long-stream checking. Its `--audio` input must be a public or otherwise authorized mono 16 kHz recording.
The reference environment is locked for Apple Silicon macOS. Native Windows/Vulkan inference is not verified.

The checks compare the frontend, each stage from its own reference inputs, the composed backbone, projection and
per-layer causal histories. Whole-waveform checks cover fixed, irregular, single-sample and whole-buffer delivery
and reset, with the final incomplete frame left unused as in the official streaming wrapper. The same configured
network schedule is kept regardless of how a caller splits the audio. A computation failure requires a reset.
F32 is the only supported storage type; quantization is not advertised before its own checks.

A C++ caller can use `fastdiar::Encoder` from `fastdiar/encoder.h`: construct it with a model path and a live ggml
backend, call `push(samples, count, embeddings)` for contiguous float samples, and `end(embeddings)` to flush.
It appends row-major normalized embeddings. `reset()` starts another recording without reloading weights.
The backend must outlive the encoder; callers serialize access to that backend.

## Check streaming clustering

`OnlineClustering` follows the official `post_process=False` path with full history and exact coordinate medians.
It has no eight-speaker limit. Its output IDs are internal cluster numbers, not the arrival-ordered public speaker
labels assigned by `StreamingDiarizer`.

```sh
uv run --project reference/fastdiar reference/fastdiar/dump.py \
  --source /tmp/fastdiar --out reference/fastdiar/out/clustering
cmake --build build --target fastdiar-clustering-check -j4
build/fastdiar-clustering-check reference/fastdiar/out/clustering/*
```

`dump.py --embeddings result.npz` additionally checks normalized embeddings and VAD decisions recorded by an official
`StreamingDiarizer` (`embeddings` float32, `vad` uint8). IDs and their emission steps compare exactly, including
flush, twelve returning voices, online merges, silence and short tails. Whole-encoder case folders may also carry
`vad.npy` and `labels.npy` (the clustering dump's `expected.npy`): the encoder check then verifies that its native
embeddings preserve the official cluster IDs and emission steps. This supplies a reference VAD mask; it does not
claim that native VAD inference is integrated.

Whole-model performance and DER measurements belong in speech-bench, not these correctness checks.

## Check the VAD and the whole path

```sh
uv run --project reference/fastdiar reference/fastdiar/vad-dump.py \
  --source /tmp/fastdiar --audio /path/to/public-16khz-mono.wav --out reference/fastdiar/out/vad
cmake --build build --target fastdiar-vad-check speech-cli -j4
build/fastdiar-vad-check reference/fastdiar/out/vad
build/speech diarize models/FASTDIAR-large-10.4M-F32.gguf --format rttm /path/to/public-16khz-mono.wav > ours.rttm
python3 checks/fastdiar/compare-rttm.py official.rttm ours.rttm
```

`official.rttm` comes from the official `StreamingDiarizer` with `shift_sec=0.32` on the same audio.
