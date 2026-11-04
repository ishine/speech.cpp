"""Dump official streaming clustering inputs and labels, including the push that finalizes each frame.

Usage: uv run dump.py --source /path/to/microsoft-fastdiar --out out/clustering [--embeddings trial.npz]
The optional npz holds float32 embeddings and uint8 vad from an official StreamingDiarizer run.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

COMMIT = "f48bff53b9a6a0cce4539e94b6def95f609eaea4"


def save(folder, embeddings, vad, options):
    from fastdiar.clustering import OnlineClustering

    folder.mkdir(parents=True, exist_ok=True)
    clustering = OnlineClustering(**options)
    expected = []
    for i, (embedding, speech) in enumerate(zip(embeddings, vad, strict=True)):
        expected.extend((i, frame, -1 if label is None else label) for frame, label in clustering.fit(embedding, bool(speech)))
    expected.extend((len(vad), frame, -1 if label is None else label) for frame, label in clustering.finalize())
    np.save(folder / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    np.save(folder / "vad.npy", np.asarray(vad, dtype=np.uint8))
    np.save(folder / "expected.npy", np.asarray(expected, dtype=np.int32).reshape(-1, 3))
    np.save(folder / "options.npy", np.asarray([
        clustering.delay_frames, clustering.max_delay_frames, clustering.clust_th,
        clustering.merge_th, clustering.confidence, int(clustering.online_merge),
    ], dtype=np.float64))
    (folder / "meta.json").write_text(json.dumps({"officialCommit": COMMIT, "options": options, "frames":len(vad), "dimensions":embeddings.shape[1]}, indent=2) + "\n")
    print(f"{folder.name}: {len(vad)} frames, {len(expected)} finalized labels", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise SystemExit(f"official source is {revision}, not {COMMIT}")
    sys.path.insert(0, str(args.source.resolve()))
    rng = np.random.default_rng(20261103)
    centers = np.eye(16, dtype=np.float32)
    # Twelve voices return in reverse order; silence and short turns exercise forced assignments and flushes.
    embeddings = np.vstack([np.repeat(centers[[s]], 35, axis=0) + rng.normal(0, .01, (35,16)).astype(np.float32)
                            for s in list(range(12)) + list(reversed(range(12)))])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    vad = np.ones(len(embeddings), dtype=np.uint8)
    vad[::37] = 0
    save(args.out / "twelve-speakers", embeddings, vad, {})
    save(args.out / "no-merge", embeddings, vad, {"online_merge":False})
    save(args.out / "silence", embeddings[:67], np.zeros(67, dtype=np.uint8), {})
    save(args.out / "short-tail", embeddings[:7], vad[:7], {})
    save(args.out / "empty", embeddings[:0], vad[:0], {})
    save(args.out / "zero-delay", embeddings[:150], vad[:150], {"delay_frames":0,"max_delay_sec":0})
    # Nearby centers pass the speech-cluster merge threshold after distinct anchor clusters are opened.
    nearby = rng.normal(size=(9,16)).astype(np.float32)
    nearby /= np.linalg.norm(nearby, axis=1, keepdims=True)
    noisy = np.vstack([np.repeat(nearby[[s % 9]], 20, axis=0) + rng.normal(0,.02,(20,16)).astype(np.float32) for s in range(50)])
    noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
    save(args.out / "online-merges", noisy, np.ones(len(noisy), dtype=np.uint8), {"clust_th":.99,"merge_th":.6,"confidence":.2})
    if args.embeddings:
        data = np.load(args.embeddings)
        save(args.out / "recording", data["embeddings"], data["vad"], {})


if __name__ == "__main__":
    main()
