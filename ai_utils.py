# ai_utils.py
import os
import torch
import soundfile as sf
from transformers import pipeline

# ---------------------------
# Import Groq safely
# ---------------------------
try:
    from groq import Groq
except ImportError as e:
    print("ERROR: Groq module not found. Install it with 'pip install groq'")
    Groq = None

# ---------------------------
# Initialize STT (speech to text)
# ---------------------------
try:
    stt = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    print("STT pipeline loaded successfully.")
except Exception as e:
    print("ERROR loading STT pipeline:", e)
    stt = None

# ---------------------------
# Initialize TTS (text to speech)
# ---------------------------
try:
    tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    print("TTS pipeline loaded successfully.")
except Exception as e:
    print("ERROR loading TTS pipeline:", e)
    tts = None

# ---------------------------
# Initialize Groq client
# ---------------------------
GROQ_API_KEY = os.environ.get("GGROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: Groq API key not found in environment variables!")

if Groq and GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized successfully.")
    except Exception as e:
        print("ERROR initializing Groq client:", e)
        client = None
else:
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
        if stt is None or tts is None:
            raise RuntimeError("STT or TTS pipeline not initialized!")

        # STT: Convert speech to text
        print("Converting audio to text...")
        text_input = stt(audio_file_path)["text"]
        print("STT output:", text_input)

        # Groq AI inference
        if client:
            print("Sending text to Groq API...")
            # Replace with your Groq model call
            # Example: groq_response = client.predict(model="YOUR_MODEL", input=text_input)
            groq_response = f"AI Response to: {text_input}"  # placeholder
        else:
            groq_response = f"AI Response to: {text_input}"  # fallback
        print("Groq response:", groq_response)

        # TTS: Convert response text to audio
        print("Generating audio response...")
        tts_output = tts(groq_response)

        # Handle different return formats
        if isinstance(tts_output, dict):
            audio_array = tts_output.get("array")
            sampling_rate = tts_output.get("sampling_rate", 22050)
        else:
            # sometimes it's just a numpy array
            audio_array = tts_output
            sampling_rate = 22050

        if audio_array is None:
            raise RuntimeError("TTS returned no audio!")

        audio_response_path = "response.wav"
        sf.write(audio_response_path, audio_array, sampling_rate)
        print("Audio saved at:", audio_response_path)

        return groq_response, audio_response_path

    except Exception as e:
        print("ERROR in process_audio:", e)
        return f"ERROR: {e}", None
