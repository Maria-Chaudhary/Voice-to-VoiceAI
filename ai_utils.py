# ai_utils.py
import os
import torch
import soundfile as sf
from transformers import pipeline
from gtts import gTTS

# ---------------------------
# Initialize STT (speech-to-text)
# ---------------------------
print("Loading STT model (Whisper-small)...")
stt = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# ---------------------------
# Initialize Groq client
# ---------------------------
from groq import Client as Groq

GROQ_API_KEY = os.environ.get("GGROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: Groq API key not found in environment variables!")
client = Groq(api_key=GROQ_API_KEY)

def get_groq_response(text_input: str):
    """Send text to Groq model and return AI response."""
    try:
        # Replace 'your_model_name' with your actual Groq deployed model name
        response = client.run(model="your_model_name", input_data={"text": text_input})
        groq_response = response[0] if isinstance(response, list) else response
        return groq_response
    except Exception as e:
        print("ERROR running Groq model:", e)
        return f"ERROR: {e}"

# ---------------------------
# Process voice input
# ---------------------------
def process_audio(audio_file_path: str):
    """
    Input: path to audio file (.wav)
    Output: tuple (text_response, audio_response_path)
    """
    try:
        # STT: Convert speech to text
        print("Converting audio to text...")
        text_input = stt(audio_file_path)["text"]
        print("STT output:", text_input)

        # Groq AI inference
        print("Sending text to Groq API...")
        groq_response = get_groq_response(text_input)
        print("Groq response:", groq_response)

        # TTS: Convert response text to audio using gTTS
        print("Generating audio response with gTTS...")
        tts_audio_path = "response.mp3"
        tts = gTTS(text=groq_response, lang="en")
        tts.save(tts_audio_path)
        print("Audio saved at:", tts_audio_path)

        return groq_response, tts_audio_path

    except Exception as e:
        print("ERROR in process_audio:", e)
        return f"ERROR: {e}", None
