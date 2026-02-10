# ai_utils.py
import os
import soundfile as sf
from gtts import gTTS
from transformers import pipeline
from groq import Client as Groq


# ---------------------------
# Initialize STT (speech to text)
# ---------------------------
stt = pipeline("automatic-speech-recognition", model="openai/whisper-small")

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
    Output: tuple (AI_text_response, audio_response_path)
    """
    try:
        # ---------------------------
        # STT: Convert speech to text
        # ---------------------------
        print("Converting audio to text...")
        text_input = stt(audio_file_path)["text"]
        print("STT output:", text_input)

        # ---------------------------
        # Groq AI inference
        # ---------------------------
        print("Sending text to Groq API...")
        groq_response = client.predict(
            model="YOUR_MODEL_NAME",  # replace with your Groq model
            input=text_input
        )
        print("Groq response:", groq_response)

        # ---------------------------
        # TTS: Convert response text to audio using gTTS
        # ---------------------------
        print("Generating audio response...")
        tts = gTTS(groq_response, lang="en")
        audio_response_path = "response.mp3"
        tts.save(audio_response_path)
        print("Audio saved at:", audio_response_path)

        return groq_response, audio_response_path

    except Exception as e:
        print("ERROR in process_audio:", e)
        return f"ERROR: {e}", None
