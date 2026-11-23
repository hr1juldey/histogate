from pocket_tts import TTSModel
import numpy as np
import wave

# v5 s4 — biologist-turned-engineer voice, style copied from his corpus
TEXT = (
    "I am a biologist first, so I went back to the staining chemistry. "
    "H and E have known absorbance. The Beer-Lambert law tells you how much light each dye takes in. "
    "I actually baked that physics into the training, so the model sees through the stain "
    "instead of memorizing it. Drag the dial, and the old model gives a wrong answer. "
    "The shipped model holds."
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
with wave.open("C_s4_v5.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(m.sample_rate)
    w.writeframes(a16.tobytes())
print("OK C_s4_v5.wav")
