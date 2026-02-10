# app.py
import gradio as gr
from ai_utils import process_audio

def handle_audio(audio):
    if audio is None:
        return "ERROR: No audio received", None
    text_resp, audio_resp = process_audio(audio)
    return text_resp, audio_resp

with gr.Blocks() as ui:
    gr.Markdown("## Voice-to-Voice AI Assistant")
    audio_input = gr.Audio(sources="microphone", type="filepath", label="Record your voice")
    submit_btn = gr.Button("Submit")
    text_output = gr.Textbox(label="AI Response (Text)")
    audio_output = gr.Audio(label="AI Response (Audio)")

    submit_btn.click(
        fn=handle_audio,
        inputs=audio_input,
        outputs=[text_output, audio_output]
    )

ui.launch(share=True)
