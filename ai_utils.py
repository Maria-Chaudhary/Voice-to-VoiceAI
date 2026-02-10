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
# Helper: Create Groq prompt with strict language rules
# ---------------------------
def create_prompt(user_text: str, wiki_context: str = "", lang: str = "en") -> str:
    """
    Returns a professional human-like prompt for Groq.
    lang: 'en' for English, 'ur' for Urdu
    wiki_context: optional context from Wikipedia (always in same lang)
    """
    if lang.lower() == "ur":
        # Urdu prompt
        if wiki_context:
            wiki_context_ur = f"پس منظر: {wiki_context}"
        else:
            wiki_context_ur = ""
        prompt = (
            f"آپ ایک مہارت رکھنے والے پروفیشنل اسسٹنٹ ہیں جو صارف کے سوالات کے جواب دیتے ہیں۔ "
            f"جواب ہمیشہ اردو میں ہونا چاہیے، صاف، دوستانہ اور انسانی زبان میں۔\n"
            f"صارف نے کہا: {user_text}\n"
            f"{wiki_context_ur}\n"
            "براہ کرم صرف اردو میں جواب دیں، کبھی بھی ہندی یا انگریزی میں جواب نہ دیں۔"
        )
    else:
        # English prompt
        prompt = (
            "You are a professional, helpful assistant who answers questions in a natural, human-like tone. "
            "Do not mention AI limitations or knowledge cutoff. Provide clear, concise, and actionable responses.\n"
            f"User said: {user_text}\n"
        )
        if wiki_context:
            prompt += f"Context: {wiki_context}\n"
        prompt += "Please respond in English."
    return prompt

# ---------------------------
# Main: Process audio input, get AI response, convert to speech
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
        # 1️⃣ Speech to Text
        print("Converting audio to text...")
        text_input = stt(audio_file_path)["text"]
        print("STT output:", text_input)

        # 2️⃣ Retrieve context from Wikipedia (optional)
        wiki_summary = ""
        try:
            if lang.lower() == "ur":
                # Wikipedia Urdu summary
                wiki_summary = wikipedia.summary(text_input, sentences=2, auto_suggest=False, redirect=True)
            else:
                # English summary
                wiki_summary = wikipedia.summary(text_input, sentences=2)
        except Exception as e:
            print("Wikipedia fetch failed:", e)
            wiki_summary = ""

        # 3️⃣ Create Groq prompt
        prompt = create_prompt(text_input, wiki_summary, lang)

        # 4️⃣ Groq chat completion
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

        # 5️⃣ Convert AI response to audio using gTTS
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
