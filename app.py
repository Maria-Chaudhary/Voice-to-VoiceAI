# app.py
import gradio as gr
from ai_utils import process_audio

# ---------------------------
# Function to handle audio submission
# ---------------------------
def submit_audio(audio):
    """
    audio: a tuple (sample_rate, numpy_array) from Gradio Audio component
    Returns: AI text response and audio file path
    """
    try:
        if audio is None:
            return "No audio received!", None

        # Gradio provides a tuple (sample_rate, numpy array)
        sample_rate, audio_array = audio

        # Save incoming audio to file
        input_audio_path = "input.wav"
        import soundfile as sf
        sf.write(input_audio_path, audio_array, sample_rate)
        
        # Process audio using ai_utils
        ai_text, ai_audio_path = process_audio(input_audio_path)
        return ai_text, ai_audio_path

    except Exception as e:
        # Log exact error
        print("ERROR in submit_audio:", e)
        return f"ERROR: {e}", None

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks() as ui:
    gr.Markdown("## 🎤 Voice-to-Voice AI Assistant")
    
    with gr.Row():
        audio_input = gr.Audio(sources="microphone", type="numpy", label="Record your voice")
        submit_btn = gr.Button("Submit")
    
    with gr.Row():
        text_output = gr.Textbox(label="AI Response", lines=4)
        audio_output = gr.Audio(label="AI Voice Response", type="filepath")

    # Link button click
    submit_btn.click(
        fn=submit_audio,
        inputs=[audio_input],
        outputs=[text_output, audio_output]
    )

# Launch Gradio app
ui.launch(share=True)
