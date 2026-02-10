# app.py
import gradio as gr
from ai_utils import process_audio

# ---------------------------
# Gradio UI
# ---------------------------

with gr.Blocks(title="Voice-to-Voice AI Assistant") as ui:
    
    gr.Markdown("## 🎙️ Voice-to-Voice AI Assistant")
    gr.Markdown(
        "Record your voice, submit, and the AI will respond with text and audio.\n\n"
        "**Powered by Groq, Whisper, and gTTS**"
    )

    with gr.Row():
        audio_input = gr.Audio(
            label="Record your voice here", 
            type="filepath",  # Provides a file path to process_audio
            sources="microphone"
        )
        submit_btn = gr.Button("Submit")

    ai_text = gr.Textbox(label="AI Response Text", placeholder="AI response will appear here...", lines=5)
    ai_audio = gr.Audio(label="AI Response Audio")

    def submit_action(audio_file):
        if not audio_file:
            return "ERROR: No audio recorded.", None

        text, audio_path = process_audio(audio_file)
        return text, audio_path

    submit_btn.click(fn=submit_action, inputs=audio_input, outputs=[ai_text, ai_audio])

# Launch Gradio app
ui.launch(debug=True, share=True)
