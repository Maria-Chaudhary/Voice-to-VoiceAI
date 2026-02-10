# app.py
import gradio as gr
from ai_utils import process_audio

def voice_ai(audio):
    if audio is None:
        return "No audio received!", None
    return process_audio(audio)

with gr.Blocks() as ui:
    gr.Markdown("## 🎙️ Voice-to-Voice AI Assistant")
    audio_input = gr.Audio(source="microphone", type="filepath", label="Record your voice")
    submit_btn = gr.Button("Submit")
    text_output = gr.Textbox(label="AI Response (Text)")
    audio_output = gr.Audio(label="AI Response (Audio)")

    submit_btn.click(fn=voice_ai, inputs=audio_input, outputs=[text_output, audio_output])

# Important: In Colab use share=False, Spaces share=True automatically
ui.launch(share=False)
