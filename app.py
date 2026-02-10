import gradio as gr
from ai_utils import voice_ai

with gr.Blocks() as ui:
    gr.Markdown("## 🎙️ Voice-to-Voice AI Assistant")
    gr.Markdown(
        "Record your voice, press Submit, and get AI response in both text and audio."
    )

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"], type="numpy", label="🎤 Record your voice"
        )
        submit_btn = gr.Button("Submit")

    output_text = gr.Textbox(label="📝 AI Response")
    output_audio = gr.Audio(type="numpy", label="🔊 AI Response Audio")

    submit_btn.click(voice_ai, inputs=audio_input, outputs=[output_text, output_audio])

ui.launch(share=True)

