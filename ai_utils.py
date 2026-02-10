# ai_utils.py
import os
import soundfile as sf
from transformers import pipeline
from groq import Groq
from gtts import gTTS
import wikipedia

# ---------------------------
# Initialize STT (Whisper)
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
# Fallback knowledge for historical / factual queries
# ---------------------------
FALLBACK_KNOWLEDGE = {
    "major shaitan singh bhati": (
        "Major Shaitan Singh Bhati was a hero in the Battle of Rezang La "
        "during the 1962 India-China war. He led his company with extraordinary courage "
        "and sacrificed his life defending his post against overwhelming odds."
    ),
    "rezang la": (
        "Rezang La was the site of a key battle in the 1962 India-China war, "
        "where Indian soldiers, led by Major Shaitan Singh Bhati, displayed extraordinary valor."
    ),
}

# ---------------------------
# Retrieve context from Wikipedia
# ---------------------------
def retrieve_context(query: str):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except Exception:
        # If Wikipedia fails, check fallback knowledge
        key = query.lower()
        return FALLBACK_KNOWLEDGE.get(key, "")

# ---------------------------
# Convert AI text to speech using gTTS
# ---------------------------
def text_to_speech(text: str, filename="response.mp3"):
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(filename)
        return filename
    except Exception as e:
        print("TTS Error:", e)
        return None

# ---------------------------
# Process audio input and generate AI response
# ---------------------------
def process_audio(audio_file_path: str):
    """
    Input: path to audio file (.wav)
    Output: tuple (AI_text_response, audio_response_path)
    """
    try:
        # STT: Convert speech to text
        print("Converting audio to text...")
        user_text = stt(audio_file_path)["text"]
        print("User said:", user_text)

        # Retrieve context (Wikipedia or fallback)
        context = retrieve_context(user_text)
        if context:
            print("Retrieved context:", context)

        # Groq: Generate AI response
        prompt = f"""
        You are a helpful and factual assistant.
        Always provide accurate answers based on the following context:
        {context}

        User asked: {user_text}
        Assistant:"""

        print("Sending prompt to Groq...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        ai_text = response.choices[0].message.content
        print("AI Response:", ai_text)

        # TTS: Convert AI text to audio
        audio_response_path = text_to_speech(ai_text)
        if not audio_response_path:
            print("Warning: TTS returned no audio!")

        return ai_text, audio_response_path

    except Exception as e:
        print("ERROR in process_audio:", e)
        return f"ERROR: {e}", None
