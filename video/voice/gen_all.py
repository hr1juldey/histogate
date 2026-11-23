"""Generate all 6 v5 narration scenes in the cloned (natural, denoised)
voice, verify each numerically, concatenate into the full narration track."""
import json
import subprocess

import numpy as np
import wave

from pocket_tts import TTSModel

SCENES = json.load(open("narration_v5.json"))
PROMPT = "bank/prompt_denoised.wav"

m = TTSModel.load_model()
state = m.get_state_for_audio_prompt(PROMPT)

report = []
wavs = []
for s in SCENES:
    audio = m.generate_audio(state, s["text"])
    a = np.array(list(audio), dtype=np.float32)
    if a.max() <= 1.0 + 1e-3:
        a = a * 32767
    a16 = np.clip(np.round(a), -32768, 32767).astype(np.int16)
    rms = float(np.sqrt(((a16.astype(np.float32) / 32768) ** 2).mean()))
    peak = float(np.abs(a16.astype(np.float32) / 32768).max())
    dur = len(a16) / m.sample_rate
    wps = dur / len(s["text"].split())
    ok = rms > 0.03 and not np.isnan(a16).any() and 0.2 <= wps <= 0.6
    path = f"narr_{s['id']}.wav"
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(m.sample_rate)
        w.writeframes(a16.tobytes())
    report.append((s["id"], round(dur, 1), round(rms, 4), round(peak, 3), round(wps, 2), "OK" if ok else "FAIL"))
    wavs.append(path)
    print(report[-1])

# concat with 0.6s gaps between scenes
gap = int(0.6 * m.sample_rate)
sil = np.zeros(gap, dtype=np.int16)
chunks = []
for p in wavs:
    with wave.open(p, "rb") as w:
        assert w.getframerate() == m.sample_rate
        chunks.append(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16))
        chunks.append(sil)
full = np.concatenate(chunks[:-1])
with wave.open("narration_full.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(m.sample_rate)
    w.writeframes(full.tobytes())
print("FULL:", round(len(full) / m.sample_rate, 1), "s")
