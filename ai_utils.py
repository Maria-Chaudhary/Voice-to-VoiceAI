import os
import torch
import soundfile as sf
from transformers import pipeline

# ---------------------------
# Initialize STT (speech-to-text)
# ---------------------------
stt = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# ---------------------------
# Initialize TTS (text-to-speech) using stable Google TTS
# ---------------------------
tts = pipeline("text-to-speech", model="tts_models/en/ljspeech/tacotron2-DDC")

# ---------------------------
# Initialize Groq client (optional)
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
        # Convert speech to text
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

        # Convert text to speech
        print("Generating audio response...")
        tts_output = tts(groq_response)

        # tts_output from this model is always a NumPy array
        if isinstance(tts_output, dict):
            audio_array = tts_output.get("array")
            sampling_rate = tts_output.get("sampling_rate", 22050)
        else:
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
