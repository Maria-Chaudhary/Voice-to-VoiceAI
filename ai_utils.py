# ai_utils.py
import os
import torch
import soundfile as sf
from transformers import pipeline
from groq.client import Groq

# ---------------------------
# Initialize STT (speech to text)
# ---------------------------
stt = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# ---------------------------
# Initialize TTS (text to speech)
# ---------------------------
tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")

# ---------------------------
# Initialize Groq client
# ---------------------------
GROQ_API_KEY = os.environ.get("GGROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: Groq API key not found in environment variables!")
client = Groq(api_key=GROQ_API_KEY)

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

        # Groq AI inference (echoing text for demo, replace with your model call)
        print("Sending text to Groq API...")
        # Here you would send text_input to your Groq model
        # Example: groq_response = client.predict(model="YOUR_MODEL", input=text_input)
        groq_response = f"AI Response to: {text_input}"  # placeholder
        print("Groq response:", groq_response)

        # TTS: Convert response text to audio
        print("Generating audio response...")
        tts_output = tts(groq_response)
        audio_response_path = "response.wav"
        sf.write(audio_response_path, tts_output["array"], tts_output["sampling_rate"])
        print("Audio saved at:", audio_response_path)

        return groq_response, audio_response_path

    except Exception as e:
        print("ERROR in process_audio:", e)
        return f"ERROR: {e}", None
