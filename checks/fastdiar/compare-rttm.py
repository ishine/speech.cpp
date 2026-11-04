"""Compare two RTTM files frame by frame at 80 ms, speakers named by their order of first appearance.

usage: python3 compare-rttm.py OFFICIAL.rttm OURS.rttm
"""
import sys


def frames(path):
    turns = sorted((float(p[3]), float(p[4]), p[7]) for p in (line.split() for line in open(path)))
    names, out = {}, {}
    for start, length, name in turns:
        number = names.setdefault(name, len(names))
        for f in range(round(start / 0.08), round((start + length) / 0.08)):
            out[f] = number
    return out, len(names)


official, official_speakers = frames(sys.argv[1])
ours, our_speakers = frames(sys.argv[2])
differ = [f for f in sorted(set(official) | set(ours)) if official.get(f) != ours.get(f)]
print(f"speakers {official_speakers} / {our_speakers}, labelled frames {len(official)} / {len(ours)}, differing {len(differ)}", differ[:10])
sys.exit(1 if differ or official_speakers != our_speakers else 0)
