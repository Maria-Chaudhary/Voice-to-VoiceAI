import os
import numpy as np
import soundfile as sf
from transformers import pipeline

# ---------- STT (Speech-to-Text) ----------
stt = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# ---------- TTS (Text-to-Speech) ----------
tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")

# ---------- Groq API ----------
from groq import Groq
groq_client = Groq(api_key=os.environ.get("GGROQ_API_KEY"))

def generate_response(transcription: str) -> str:
    """Call Groq LLM to generate AI response"""
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": transcription}],
        model="llama-3.3-70b-versatile"
    )
    return chat_completion.choices[0].message.content

def voice_ai(audio: tuple):
    """
    Input: audio tuple (np_array, sample_rate)
    Output: ai_text, ai_audio (np_array, sample_rate)
    """
    audio_array, sr = audio
    sf.write("temp.wav", audio_array, sr)

    # 1. STT
    transcription = stt("temp.wav")["text"]

    # 2. AI Response via Groq
    ai_text = generate_response(transcription)

    # 3. TTS
    speech = tts(ai_text)
    speech_array = np.array(speech["array"])
    speech_sr = speech["sampling_rate"]

    return ai_text, (speech_array, speech_sr)
