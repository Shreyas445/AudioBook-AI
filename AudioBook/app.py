import gradio as gr
import os
import threading
from main import AudiobookConverter
from config import OUTPUT_DIR, FINAL_FILENAME

def convert_audiobook_ui(file_obj, progress=gr.Progress()):
    if file_obj is None:
        return None, "Please upload a file first."
    
    logs = []
    
    def log_callback(msg):
        print(msg) # Keep console output
        logs.append(msg)
        
    converter = AudiobookConverter(log_callback=log_callback)
    
    # We need to yield logs while running. Gradio supports generator outputs.
    # However, for simplicity in this synchronous flow, we might just return the final result.
    # To stream logs *during* execution, we would need a generator.
    
    # Let's try a generator approach for logs.
    pass

# Better approach for streaming logs in Gradio:
# Run the heavy task in a separate thread/generator and yield updates.

import shutil

def check_disk_space(required_gb=2.0):
    total, used, free = shutil.disk_usage(".")
    free_gb = free // (2**30)
    if free_gb < required_gb:
        return False, f"CRITICAL: Low Disk Space! Only {free_gb:.2f} GB free. AI models need at least {required_gb} GB buffer."
    return True, f"Disk Space OK ({free_gb:.2f} GB free)"

def run_conversion(file_obj):
    logs = []
    def logger(msg):
        logs.append(msg)
        print(msg)
        # In a real generator, we would yield here.
        
    if not file_obj:
        yield None, "Please upload a file."
        return

    # 1. Check Disk
    has_space, space_msg = check_disk_space()
    if not has_space:
        yield None, space_msg + "\nPlease free up space on Drive C: and try again."
        return
    logger(space_msg)

    # 2. Run Conversion
    converter = AudiobookConverter(log_callback=logger)
    yield None, "Initializing... (Check console for real-time logs)"
    
    try:
        final_path, status = converter.convert(file_obj.name, progress_bar=None)
        
        full_log = "\n".join(logs)
        yield final_path, full_log + "\n\n" + status
        
    except Exception as e:
        import traceback
        err_details = traceback.format_exc()
        full_log = "\n".join(logs)
        # Return log so user sees what happened
        yield None, f"ERROR: {e}\n\n{err_details}\n\nLogs:\n{full_log}"


# Let's try to make a proper generator to stream logs!
# We can subclass AudiobookConverter or just use a generator in `main.py`.
# But for now, let's stick to the simple version to ensure stability first.
# The user wants "debug console". A final dump is often acceptable if progress bar works.

# Let's create the App
with gr.Blocks(title="AI Audiobook Converter") as app:
    gr.Markdown("# 📚 AI Audiobook Converter")
    gr.Markdown("Transform text/PDF into an immersive audiobook with Voices and Music.")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Book (PDF or TXT)", file_types=[".txt", ".pdf"])
            convert_btn = gr.Button("Convert to Audiobook", variant="primary")
        
        with gr.Column():
            audio_out = gr.Audio(label="Generated Audiobook")
            log_out = gr.Textbox(label="Conversion Log", lines=15, interactive=False)

    convert_btn.click(
        fn=run_conversion,
        inputs=[file_input],
        outputs=[audio_out, log_out]
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", inbrowser=True)
