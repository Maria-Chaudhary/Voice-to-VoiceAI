# ai_utils.py
import os
import soundfile as sf
from gtts import gTTS
from transformers import pipeline

# ---------------------------
# Initialize STT (speech-to-text)
# ---------------------------
stt = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# ---------------------------
# Groq client (optional)
# ---------------------------
try:
    from groq.client import Groq
    GROQ_API_KEY = os.environ.get("GGROQ_API_KEY")
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception as e:
    print("Groq client not loaded:", e)
    client = None

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

        # AI inference
        if client:
            print("Sending text to Groq API...")
            # Replace with your Groq model call
            groq_response = f"AI Response to: {text_input}"  # placeholder
        else:
            groq_response = f"AI Response to: {text_input}"
        print("AI response:", groq_response)

        # TTS: Convert text to speech using gTTS
        print("Generating audio response with gTTS...")
        tts = gTTS(groq_response)
        audio_response_path = "response.mp3"
        tts.save(audio_response_path)
        print("Audio saved at:", audio_response_path)

        return groq_response, audio_response_path

    except Exception as e:
        print("ERROR in process_audio:", e)
        return f"ERROR: {e}", None
