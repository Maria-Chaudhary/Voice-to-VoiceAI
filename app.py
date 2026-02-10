# app.py
import gradio as gr
from ai_utils import process_audio

# ---------------------------
# Function to handle Gradio submission
# ---------------------------
def submit_audio(audio, language):
    if audio is None:
        return "No audio received.", None
    # Process audio through our ai_utils function
    text_resp, audio_path = process_audio(audio, lang=language)
    return text_resp, audio_path

# ---------------------------
# Build Gradio Interface
# ---------------------------
with gr.Blocks(title="Voice-to-Voice AI Assistant") as ui:
    gr.Markdown(
        """
        # 🎤 Voice-to-Voice AI Assistant
        Speak your question, select language (English/Urdu), and get a real-time AI response with audio.
        """
    )

    # Audio input from microphone
    audio_input = gr.Audio(sources="microphone", type="filepath", label="Record your voice")

    # Language selector
    language_select = gr.Radio(choices=["en", "ur"], label="Select Language for Response", value="en")

    # Submit button
    submit_btn = gr.Button("Submit")

    # Outputs: AI Text + Audio
    ai_text = gr.Textbox(label="AI Response Text", lines=5)
    ai_audio = gr.Audio(label="AI Response Audio")

    # Connect the button to processing function
    submit_btn.click(fn=submit_audio, inputs=[audio_input, language_select], outputs=[ai_text, ai_audio])

# Launch the UI
if __name__ == "__main__":
    ui.launch()
