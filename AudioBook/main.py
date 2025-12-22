import os
import argparse
import time
import shutil
from tqdm import tqdm

from config import OUTPUT_DIR, TEMP_DIR, FINAL_FILENAME, MUSIC_DURATION
from core.script_engine import ScriptGenerator
from core.tts_engine import TTSEngine
from core.music_engine import MusicEngine
from core.studio import Studio
from utils.text_utils import read_text_file, chunk_text, read_pdf_file

class AudiobookConverter:
    def __init__(self, log_callback=print):
        self.log = log_callback
        self.log(f"Initializing Engines (CUDA: {'Enabled' if 'cuda' in os.environ.get('CUDA_VISIBLE_DEVICES', '') else 'Check Config'})...")
        
        self.script_gen = ScriptGenerator()
        try:
            self.tts_engine = TTSEngine()
        except Exception as e:
            self.log(f"Error initializing TTS: {e}")
            self.tts_engine = None

        try:
            self.music_engine = MusicEngine()
        except Exception as e:
            self.log(f"Warning: Music Engine failed to initialize ({e}). Continuing without music.")
            self.music_engine = None

        self.studio = Studio()

    def convert(self, input_file: str, progress_bar=None):
        if not self.tts_engine:
            return None, "TTS Engine failed to load."

        self.log(f"Reading {input_file}...")
        
        if input_file.lower().endswith('.pdf'):
            raw_text = read_pdf_file(input_file)
        else:
            raw_text = read_text_file(input_file)
            
        chunks = chunk_text(raw_text)
        self.log(f"Split into {len(chunks)} chunks.")

        os.makedirs(TEMP_DIR, exist_ok=True)
        all_voice_files = []
        all_music_files = []
        music_cache = {}

        total_steps = len(chunks)
        
        for chunk_idx, chunk in enumerate(chunks):
            self.log(f"Processing Chunk {chunk_idx + 1}/{total_steps}")
            
            # Script
            try:
                script = self.script_gen.generate_script(chunk)
            except Exception as e:
                self.log(f"Failed to generate script: {e}")
                continue

            # Assets
            for block_idx, block in enumerate(script.blocks):
                # Voice
                voice_filename = f"c{chunk_idx}_b{block_idx}_voice.wav"
                voice_path = os.path.join(TEMP_DIR, voice_filename)
                
                try:
                    self.tts_engine.generate(block.text, block.voice_id, voice_path)
                    all_voice_files.append(voice_path)
                except Exception as e:
                     self.log(f"TTS Error: {e}")

                # Music
                music_path = None
                if block.bg_music_prompt and self.music_engine:
                    # Clean prompt for filename
                    safe_prompt = "".join([c for c in block.bg_music_prompt if c.isalnum()]).strip()[:20]
                    if not safe_prompt: safe_prompt = "bg"
                    
                    cached = music_cache.get(safe_prompt)
                    if cached and os.path.exists(cached):
                        music_path = cached
                    else:
                        m_file = f"music_{safe_prompt}.wav"
                        m_path = os.path.join(TEMP_DIR, m_file)
                        if not os.path.exists(m_path):
                            try:
                                self.log(f"Generating music: {block.bg_music_prompt}")
                                self.music_engine.generate(block.bg_music_prompt, MUSIC_DURATION, m_path)
                                music_cache[safe_prompt] = m_path
                                music_path = m_path
                            except Exception as e:
                                self.log(f"Music Error: {e}")
                        else:
                            music_cache[safe_prompt] = m_path
                            music_path = m_path
                
                all_music_files.append(music_path)
            
            if progress_bar:
                progress_bar((chunk_idx + 1) / total_steps)

        # Stitch
        self.log("Stitching audio...")
        output_path = os.path.join(OUTPUT_DIR, FINAL_FILENAME)
        final_file = self.studio.mix(all_voice_files, all_music_files, output_path)
        
        # Cleanup
        self.log("Cleaning up temp files...")
        try:
            shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR, exist_ok=True)
        except Exception as e:
            self.log(f"Cleanup warning: {e}")
            
        self.log("Done!")
        return final_file, "Conversion Complete"

def main():
    parser = argparse.ArgumentParser(description="AI Audiobook Converter")
    parser.add_argument("input_file", help="Path to input file")
    args = parser.parse_args()
    
    converter = AudiobookConverter()
    converter.convert(args.input_file)

if __name__ == "__main__":
    main()
