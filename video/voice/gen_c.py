from pocket_tts import TTSModel
import numpy as np
import wave

m = TTSModel.load_model()
state = m.get_state_for_audio_prompt("bank/s7_take5.wav")
text = (
    "This is why it broke. Same tissue, different staining machines. "
    "The color changes, the disease doesn't. So I drag this dial. "
    "The first model gives a wrong answer. The shipped one holds. "
    "And when the color change is too big, it doesn't guess. It just says no."
)
audio = m.generate_audio(state, text)
print("type", type(audio))
a = np.array(list(audio), dtype=np.float32)  # safe conversion
if a.max() <= 1.0 + 1e-3:
    a = a * 32767
a16 = np.clip(np.round(a), -32768, 32767).astype(np.int16)
peak = float(np.abs(a16.astype(np.float32) / 32768).max())
rms = float(np.sqrt(((a16.astype(np.float32) / 32768) ** 2).mean()))
dur = len(a16) / m.sample_rate
print(f"peak {peak:.3f} rms {rms:.4f} dur {dur:.1f}s s/word {dur/len(text.split()):.2f}")
with wave.open("C_s4_cloned.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(m.sample_rate)
    w.writeframes(a16.tobytes())
print("OK C_s4_cloned.wav")
