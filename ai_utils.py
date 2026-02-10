# ai_utils.py
import os
import soundfile as sf
from transformers import pipeline
from groq import Groq
from gtts import gTTS
import wikipedia

# ---------------------------
# Initialize STT (Speech to Text)
# ---------------------------
print("Initializing STT (Whisper)...")
stt = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# ---------------------------
# Initialize Groq client
# ---------------------------
GROQ_API_KEY = os.environ.get("GGROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: Groq API key not found in environment variables!")
client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# Prompt creation for human-like responses
# ---------------------------
def create_prompt(user_text: str, wiki_context: str = "", lang: str = "en") -> str:
    """
    Create a prompt for Groq to generate professional, human-like responses.
    lang: "en" for English, "ur" for Urdu
    wiki_context: optional retrieved context
    """
    if lang.lower() == "ur":
        prompt = (
            f"آپ ایک مہارت رکھنے والے پروفیشنل اسسٹنٹ ہیں جو صارف کے سوالات کے جواب دیتے ہیں۔ "
            f"جواب دوستانہ، صاف، اور انسانی زبان میں ہونا چاہیے۔\n"
            f"صارف نے کہا: {user_text}\n"
        )
        if wiki_context:
            prompt += f"پس منظر: {wiki_context}\n"
        prompt += "براہ کرم اردو میں جواب دیں۔"
    else:
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
# Process audio to text, get AI response, convert to speech
# ---------------------------
def process_audio(audio_file_path: str, lang: str = "en"):
    """
    Input:
        audio_file_path: path to .wav file
        lang: "en" or "ur"
    Output:
        tuple(text_response:str, audio_response_path:str)
    """
    try:
        # 1️⃣ Convert audio to text
        print("Converting audio to text...")
        text_input = stt(audio_file_path)["text"]
        print("STT output:", text_input)

        # 2️⃣ Retrieve Wikipedia context
        wiki_summary = ""
        try:
            wiki_summary = wikipedia.summary(text_input, sentences=2)
            print("Wikipedia context retrieved.")
        except Exception as e:
            print("Wikipedia fetch failed:", e)

        # 3️⃣ Create Groq prompt
        prompt = create_prompt(text_input, wiki_summary, lang)

        # 4️⃣ Generate AI response from Groq
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile"
            )
            ai_response = chat_completion.choices[0].message.content.strip()
            print("AI response:", ai_response)
        except Exception as e:
            print("Groq API failed:", e)
            ai_response = f"Sorry, AI could not generate a response. Error: {e}"

        # 5️⃣ Convert AI response to audio using gTTS
        audio_response_path = None
        if ai_response:
            tts_lang = "en" if lang.lower() == "en" else "ur"
            try:
                tts = gTTS(text=ai_response, lang=tts_lang, slow=False)
                audio_response_path = "response.mp3"
                tts.save(audio_response_path)
                print("Audio saved at:", audio_response_path)
            except Exception as e:
                print("TTS generation failed:", e)
                audio_response_path = None

        return ai_response, audio_response_path

    except Exception as e:
        print("ERROR in process_audio:", e)
        return f"ERROR: {e}", None
