# app.py
import gradio as gr
from voice_ai import voice_ai

# Gradio UI
with gr.Blocks() as ui:
    gr.Markdown("## 🎤 Voice-to-Voice AI Assistant")
    with gr.Row():
        inp = gr.Audio(source="microphone", type="numpy", label="Record your voice")
    with gr.Row():
        out_text = gr.Textbox(label="AI Response (Text)")
        out_audio = gr.Audio(label="AI Response (Audio)")

    inp.change(voice_ai, inputs=inp, outputs=[out_text, out_audio])

ui.launch()
