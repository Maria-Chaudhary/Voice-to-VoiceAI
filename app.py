# app.py
import gradio as gr
from ai_utils import process_audio

# ---------------------------
# Gradio interface
# ---------------------------
with gr.Blocks() as ui:
    gr.Markdown("## 🎤 Voice-to-Voice AI Assistant (RAG enabled)")
    
    with gr.Row():
        audio_input = gr.Audio(label="Record your voice", type="filepath")
        submit_btn = gr.Button("Submit")
    
    with gr.Row():
        text_output = gr.Textbox(label="AI Response", interactive=False)
        audio_output = gr.Audio(label="AI Voice Response", type="filepath")

    def submit_audio(audio_file):
        ai_text, ai_audio = process_audio(audio_file)
        return ai_text, ai_audio

    submit_btn.click(submit_audio, inputs=audio_input, outputs=[text_output, audio_output])

# Launch UI
ui.launch(share=True)
