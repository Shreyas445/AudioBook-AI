import os
import soundfile as sf
try:
    from kokoro import KPipeline
except ImportError:
    print("Warning: kokoro not installed. TTS will fail.")
    KPipeline = None

from config import TTS_LANG_CODE, TEMP_DIR, SAMPLE_RATE, VALID_VOICES, TTS_DEFAULT_VOICE

class TTSEngine:
    def __init__(self):
        if KPipeline:
            # 'a' for American English, 'b' for British English
            self.pipeline = KPipeline(lang_code=TTS_LANG_CODE)
        else:
            self.pipeline = None

    def generate(self, text: str, voice_id: str, output_path: str):
        """
        Generates audio for the given text and saves it to output_path.
        """
        if not self.pipeline:
            raise RuntimeError("Kokoro pipeline not initialized.")
        
        # Validate voice
        if voice_id not in VALID_VOICES:
            print(f"Warning: Unknown voice '{voice_id}'. Falling back to '{TTS_DEFAULT_VOICE}'.")
            voice_id = TTS_DEFAULT_VOICE

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Generate audio
        # pipeline returns a generator of (graphemes, phonemes, audio)
        generator = self.pipeline(
            text, 
            voice=voice_id, 
            speed=1, 
            split_pattern=r'\n+'
        )
        
        # We might get multiple segments. For simplicity, let's stitch them or just take the first if it's one block.
        # But actually, Kokoro might split by sentence. We want to concatenate them for this block.
        
        import numpy as np
        all_audio = []

        for _, _, audio in generator:
            all_audio.append(audio)

        if all_audio:
            final_audio = np.concatenate(all_audio)
            sf.write(output_path, final_audio, SAMPLE_RATE)
            return output_path
        else:
            print(f"Warning: No audio generated for text: {text[:50]}...")
            return None

if __name__ == "__main__":
    # Test stub
    tts = TTSEngine()
    test_file = os.path.join(TEMP_DIR, "test_voice.wav")
    tts.generate("Hello, this is a test of the Kokoro TTS engine.", "af_bella", test_file)
    print(f"Generated: {test_file}")
