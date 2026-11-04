"""Check that a stream's utterances list each speaker's unbroken run once, though FASTDIAR gives its turns in pieces.

usage: python3 live-speakers.py SPEECH MODEL.gguf INPUT.wav [seconds]
The audio goes at four times real time, in the small pieces a microphone gives, since one large push shows no pieces.
"""
import json
import subprocess
import sys
import threading
import time
import wave

speech, model, audio = sys.argv[1:4]
seconds = int(sys.argv[4]) if len(sys.argv) > 4 else 40
process = subprocess.Popen([speech, "asr", "haas-ja-v1.0", "--vad", "silero-vad", "--diarize", model, "--rate", "16000", "--format", "json", "-"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
events = []
reader = threading.Thread(target=lambda: events.extend(json.loads(line) for line in process.stdout))
reader.start()
time.sleep(3)
source = wave.open(audio)
for _ in range(seconds * 10):
    process.stdin.write(source.readframes(1600))
    process.stdin.flush()
    time.sleep(0.025)
process.stdin.close()
reader.join()
assert process.wait() == 0
lists = [e["speakers"] for e in events if e.get("type", "").endswith(".completed") and e.get("speakers")]
assert lists, "no utterance has speakers"
broken = sum(1 for speakers in lists for a, b in zip(speakers, speakers[1:]) if a["speaker"] == b["speaker"] and b["start_ms"] - a["end_ms"] <= 1)
print(f"{len(lists)} utterances, {sum(map(len, lists))} entries, {broken} pieces of one run")
sys.exit(1 if broken else 0)
