---
title: Voice To VoiceAI
emoji: 🏢
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: mit
---

---

# 🎙️ Voice-to-Voice AI Assistant

A real-time **Voice-to-Voice AI assistant** that listens to your voice, understands it intelligently, and responds back with natural speech.
The system supports **English and Urdu**, with strict language control and a clean interactive UI.

---

## 🚀 Live Demo

👉 **Hugging Face Space:**
[https://huggingface.co/spaces/Mariaaa123/Voice-to-VoiceAI](https://huggingface.co/spaces/Mariaaa123/Voice-to-VoiceAI)

---

## ✨ Features

* 🎤 **Speech-to-Text (STT)** using OpenAI Whisper (small, fast, accurate)
* 🤖 **AI Responses** powered by Groq (LLaMA 3)
* 🔊 **Text-to-Speech (TTS)** using Google gTTS (stable & reliable)
* 🌐 **Language Support**

  * English → English responses only
  * Urdu → Urdu responses only 
* 📚 **RAG (Retrieval-Augmented Generation)** using Wikipedia for factual accuracy
* 🧑‍💻 **Professional & interactive UI** (Gradio)
* ☁️ **Fully deployable on Hugging Face Spaces**

---

## 🛠️ Tech Stack

* **Python**
* **Gradio** (UI)
* **Whisper-small** (Speech to Text)
* **Groq API** (LLM)
* **gTTS** (Text to Speech)
* **Wikipedia API** (Retrieval)

---

## 📂 Project Structure

```
.
├── app.py
├── ai_utils.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Local Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/Voice-to-VoiceAI.git
cd Voice-to-VoiceAI
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set Groq API Key

Linux / macOS:

```bash
export GROQ_API_KEY=your_api_key_here
```

Windows (PowerShell):

```powershell
setx GROQ_API_KEY "your_api_key_here"
```

### 4️⃣ Run the app

```bash
python app.py
```

---

## 🌍 Language Behavior (Important)

* If **English** is selected → responses are **only in English**
* If **Urdu** is selected → responses are **only in Urdu**
* No Hindi-Urdu mixing
* Prompting is strictly controlled at model level

---

## 🧠 How It Works

1. User speaks through the microphone
2. Whisper converts speech → text
3. Relevant context is retrieved from Wikipedia (RAG)
4. Groq LLM generates a **natural, professional response**
5. gTTS converts the response → voice
6. Audio response is played back to the user

---

## 🔐 Environment Variables

| Variable       | Description               |
| -------------- | ------------------------- |
| `GROQ_API_KEY` | Required for AI responses |

---

## 📄 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute it.

---

## 🙌 Acknowledgements

* OpenAI Whisper
* Groq
* Hugging Face
* Wikipedia
* Google Text-to-Speech

---

## ⭐ Support

If you find this project useful:

* Give it a ⭐ on GitHub
* Share feedback
* Fork & improve 🚀

---



