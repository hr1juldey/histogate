"""Mux the cloned narration onto the silent walkthrough.

Narration scenes (6) map onto video scenes (10) at their xfade-adjuted
start times + 0.4s lead. Builds a delayed-concat audio track, then muxes.
"""
import subprocess

SCE = "/home/riju279/Documents/code/AIRA/histogate/video/scenes"
VOICE = "/home/riju279/Documents/code/AIRA/histogate/video/voice"
OUT = "/home/riju279/Documents/code/AIRA/histogate/video/out"
XFADE = 0.5
PLAN = [("s1", None), ("s2", None), ("s3", None), ("s4", 21.0), ("s5", None),
        ("s6", None), ("s7", None), ("s8", None), ("s9", None), ("s10", None)]


def dur(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", p], capture_output=True, text=True)
    return float(r.stdout.strip())


durs = {n: (min(dur(f"{SCE}/{n}.mp4"), c) if c else dur(f"{SCE}/{n}.mp4")) for n, c in PLAN}
starts = {}
t = 0.0
for i, (n, _) in enumerate(PLAN):
    starts[n] = t
    t += durs[n] - (XFADE if i < len(PLAN) - 1 else 0)

# narration scene -> video scene it starts on
NARR_MAP = {"s1": "s1", "s2": "s2", "s3": "s3", "s4": "s4", "s5": "s5", "s6": "s9"}
LEAD = 0.4

import wave
import numpy as np

sr = 24000
segs = []
for nid, vid in NARR_MAP.items():
    with wave.open(f"{VOICE}/narr_{nid}.wav", "rb") as w:
        assert w.getframerate() == sr
        segs.append((starts[vid] + LEAD, np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)))

total = int((dur(f"{OUT}/histogate_walkthrough_silent.mp4")) * sr) + sr
track = np.zeros(total, dtype=np.int16)
for start, a in segs:
    i0 = int(start * sr)
    track[i0:i0 + len(a)] = a[: max(0, total - i0)]
with wave.open(f"{VOICE}/narration_timed.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(track.tobytes())

for nid, vid in NARR_MAP.items():
    print(nid, "->", vid, "at", round(starts[vid] + LEAD, 2), "s")
print("timed track:", round(len(track) / sr, 1), "s")

subprocess.run([
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
    "-i", f"{OUT}/histogate_walkthrough_silent.mp4",
    "-i", f"{VOICE}/narration_timed.wav",
    "-map", "0:v", "-map", "1:a",
    "-c:v", "copy",
    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
    "-c:a", "aac", "-b:a", "160k",
    "-movflags", "+faststart",
    "-shortest",
    f"{OUT}/histogate_walkthrough_voiced.mp4",
], check=True)
r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size",
                    "-of", "csv=p=0", f"{OUT}/histogate_walkthrough_voiced.mp4"],
                   capture_output=True, text=True)
print("VOICED:", r.stdout.strip())
