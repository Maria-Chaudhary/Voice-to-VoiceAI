# ai_utils.py
import os
import torch
import soundfile as sf
from transformers import pipeline
from groq import Groq
from gtts import gTTS
import wikipedia

# ---------------------------
# Initialize STT (Speech to Text)
# ---------------------------
stt = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# ---------------------------
# Initialize Groq client
# ---------------------------
GROQ_API_KEY = os.environ.get("GGROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: Groq API key not found!")
client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# Process audio to text, get AI response, convert to speech
# ---------------------------
def process_audio(audio_file_path: str, lang: str = "en"):
    """
    Input:
        audio_file_path: str (.wav)
        lang: 'en' for English, 'ur' for Urdu
    Output:
        text_response: str
        audio_response_path: str
    """
    try:
        # --- 1. Speech to text ---
        print("Converting audio to text...")
        text_input = stt(audio_file_path)["text"]
        print("STT output:", text_input)

        # --- 2. Retrieve context from Wikipedia ---
        try:
            wiki_summary = wikipedia.summary(text_input, sentences=2)
        except Exception as e:
            print("Wikipedia fetch failed:", e)
            wiki_summary = ""
        
        # --- 3. Prepare prompt for Groq ---
        prompt = f"Answer the following question concisely using the context:\nContext: {wiki_summary}\nQuestion: {text_input}"

        # --- 4. Groq chat completion ---
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            ai_response = chat_completion.choices[0].message.content
        except Exception as e:
            print("Groq API failed:", e)
            ai_response = f"Sorry, AI could not generate response. Error: {e}"

        print("AI response:", ai_response)

        # --- 5. Convert AI response to audio using gTTS ---
        if ai_response:
            tts_lang = "en" if lang.lower() == "en" else "ur"
            tts = gTTS(ai_response, lang=tts_lang)
            audio_response_path = "response.mp3"
            tts.save(audio_response_path)
            print("Audio saved at:", audio_response_path)
        else:
            audio_response_path = None
            print("TTS skipped: no AI response")

        return ai_response, audio_response_path

    except Exception as e:
        print("ERROR in process_audio:", e)
        return f"ERROR: {e}", None

# ---------------------------
# Prepare prompt for Groq
# ---------------------------
def create_prompt(user_text: str, lang: str = "en"):
    """
    lang: "en" for English, "ur" for Urdu
    """
    if lang == "ur":
        # Explicitly instruct Groq to respond in Urdu
        prompt = f"آپ ایک باصلاحیت اسسٹنٹ ہیں۔ صارف نے کہا: {user_text}\nبراہ کرم اردو میں جواب دیں۔"
    else:
        # English
        prompt = f"You are a helpful assistant. The user said: {user_text}\nPlease respond in English."
    return prompt

# ---------------------------
# In process_audio(), replace Groq prompt generation with:
# ---------------------------
prompt = create_prompt(text_input, lang=language)
chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="llama-3.3-70b-versatile",
)
groq_response = chat_completion.choices[0].message.content

