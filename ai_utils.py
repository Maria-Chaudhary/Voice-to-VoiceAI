# ai_utils.py
import os
import torch
import soundfile as sf
from transformers import pipeline
from gtts import gTTS
from groq import Groq

# ---------------------------
# Initialize STT (speech to text)
# ---------------------------
stt = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# ---------------------------
# Initialize Groq client
# ---------------------------
GROQ_API_KEY = os.environ.get("GGROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Groq API key not found in environment variables!")
client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# Retrieval function for RAG
# ---------------------------
def retrieve_context(query: str):
    """
    Simple retrieval from Wikipedia summary using Hugging Face datasets or Wikipedia API.
    Returns a string context.
    """
    try:
        from wikipedia import summary
        context = summary(query, sentences=3)
        return context
    except Exception as e:
        print("RAG retrieval error:", e)
        return ""

# ---------------------------
# Process audio input
# ---------------------------
def process_audio(audio_file_path: str):
    """
    Input: path to audio file (.wav)
    Output: tuple (text_response, audio_response_path)
    """
    try:
        # 1️⃣ STT: Convert speech to text
        print("Converting audio to text...")
        user_text = stt(audio_file_path)["text"]
        print("User said:", user_text)

        # 2️⃣ RAG: Retrieve context
        print("Retrieving context for query...")
        context = retrieve_context(user_text)
        if context:
            prompt = f"You are a helpful AI assistant. Use the following context to answer the user.\nContext: {context}\nUser: {user_text}\nAssistant:"
        else:
            prompt = f"You are a helpful AI assistant. Answer naturally.\nUser: {user_text}\nAssistant:"

        # 3️⃣ Groq AI: Generate response
        print("Generating AI response...")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        ai_text = response.choices[0].message.content
        print("AI Response:", ai_text)

        # 4️⃣ TTS: Convert response text to audio
        print("Generating TTS audio...")
        tts = gTTS(ai_text)
        audio_response_path = "response.mp3"
        tts.save(audio_response_path)
        print("Audio response saved at:", audio_response_path)

        return ai_text, audio_response_path

    except Exception as e:
        print("ERROR in process_audio:", e)
        return f"ERROR: {e}", None
