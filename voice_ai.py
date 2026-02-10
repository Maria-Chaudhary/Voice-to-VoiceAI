# voice_ai.py
import os
import torch
import numpy as np
import soundfile as sf
from transformers import pipeline
from groq import Groq

# --- Initialize STT (Whisper) ---
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# --- Initialize TTS ---
tts_pipeline = pipeline("text-to-speech", model="facebook/mms-tts-eng")

# --- Initialize Groq client ---
GROQ_API_KEY = os.environ.get("GGROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Set GGROQ_API_KEY in your environment!")

groq_client = Groq(api_key=GROQ_API_KEY)

# --- Voice AI function ---
def voice_ai(audio: np.ndarray, sampling_rate: int):
    # Save input audio
    sf.write("input.wav", audio, sampling_rate)

    # 1️⃣ STT
    text = stt_pipeline("input.wav")["text"]

    # 2️⃣ Groq LLM
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": text}],
        model="llama-3.3-70b-versatile"
    )
    ai_text = chat_completion.choices[0].message.content

    # 3️⃣ TTS
    speech = tts_pipeline(ai_text, voice="alloy")  # voice parameter depends on model
    sf.write("output.wav", speech["audio"], speech["sampling_rate"])

    # Load output audio for Gradio
    out_audio, sr = sf.read("output.wav")
    return ai_text, (sr, out_audio)
