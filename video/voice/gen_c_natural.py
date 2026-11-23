from pocket_tts import TTSModel
import numpy as np
import wave

# v4 script line for s4 — assembled from the user's OWN spoken sentences (t4_dial)
TEXT = (
    "So rather than just training on data, I actually trained it on physics also. "
    "The Beer-Lambert law was ingrained into the model, so no matter how much staining is done, "
    "the inconsistencies can be mitigated up to a level. "
    "I cannot say I have done the impossible, but I made it robust enough to work with "
    "different kinds of staining."
)
m = TTSModel.load_model()
state = m.get_state_for_audio_prompt("bank/prompt_natural_t2.wav")
audio = m.generate_audio(state, TEXT)
a = np.array(list(audio), dtype=np.float32)
if a.max() <= 1.0 + 1e-3:
    a = a * 32767
a16 = np.clip(np.round(a), -32768, 32767).astype(np.int16)
peak = float(np.abs(a16.astype(np.float32) / 32768).max())
rms = float(np.sqrt(((a16.astype(np.float32) / 32768) ** 2).mean()))
dur = len(a16) / m.sample_rate
print(f"peak {peak:.3f} rms {rms:.4f} dur {dur:.1f}s s/word {dur/len(TEXT.split()):.2f}")
with wave.open("C_s4_cloned_natural.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(m.sample_rate)
    w.writeframes(a16.tobytes())
print("OK C_s4_cloned_natural.wav")
