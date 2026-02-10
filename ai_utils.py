# ai_utils.py
import os
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
from groq import Groq

GROQ_API_KEY = os.environ.get("GGROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: Groq API key not found in environment variables!")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# Function to get AI response from Groq
# ---------------------------
def get_groq_response(text_input: str):
    """Send text to Groq LLM model and return AI response."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # replace with your Groq model
            messages=[{"role": "user", "content": text_input}],
        )
        ai_text = response.choices[0].message.content
        return ai_text
    except Exception as e:
        print("ERROR in Groq API:", e)
        return f"ERROR: {e}"

# ---------------------------
# Process voice input
# ---------------------------
def process_audio(audio_file_path: str):
    """
    Input: path to audio file (.wav)
    Output: tuple (AI response text, TTS audio path)
    """
    try:
        # STT: Convert speech to text
        print("Converting audio to text...")
        text_input = stt(audio_file_path)["text"]
        print("STT output:", text_input)

        # Groq AI inference
        print("Sending text to Groq API...")
        ai_response = get_groq_response(text_input)
        print("AI response:", ai_response)

        # TTS: Convert AI response to audio
        print("Generating audio response with gTTS...")
        tts_audio_path = "response.mp3"
        tts = gTTS(text=ai_response, lang="en")
        tts.save(tts_audio_path)
        print("Audio saved at:", tts_audio_path)

        return ai_response, tts_audio_path

    except Exception as e:
        print("ERROR in process_audio:", e)
        return f"ERROR: {e}", None
