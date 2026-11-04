"""Dump the official StreamingVAD's probabilities, intervals and decided frontier after each scored frame.

Usage: uv run vad-dump.py --source OFFICIAL --audio 16k-mono.wav --out DIR
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch

p = argparse.ArgumentParser(description=__doc__)
for name in ("source", "audio", "out"):
    p.add_argument("--" + name, type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.source.resolve()))
from fastdiar.vad import StreamingVAD

audio, rate = soundfile.read(a.audio, dtype="float32")
assert rate == 16000 and audio.ndim == 1
vad = StreamingVAD(sr=16000, shift_ms=320)
probs, decided, intervals, original = [], [], [], vad._step


def step(prob):
    out = original(prob)
    intervals.extend((len(probs), s, e) for s, e in out)
    probs.append(prob)
    decided.append(vad.decided_upto())
    return out


vad._step = step
with torch.no_grad():
    for at in range(0, len(audio), 5120):
        vad(torch.from_numpy(audio[at:at + 5120]))
    intervals.extend((len(probs), s, e) for s, e in vad.flush())
a.out.mkdir(parents=True, exist_ok=True)
np.save(a.out / "probs.npy", np.array(probs, dtype=np.float64))
np.save(a.out / "decided.npy", np.array(decided, dtype=np.float64))
np.save(a.out / "intervals.npy", np.array(intervals, dtype=np.float64).reshape(-1, 3))
print(len(probs), "frames", len(intervals), "intervals")
