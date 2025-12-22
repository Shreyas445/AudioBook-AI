import os
import torch
import soundfile as sf
from diffusers import AudioLDM2Pipeline
from config import MUSIC_MODEL_ID, TEMP_DIR

class MusicEngine:
    def __init__(self):
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        if self.pipeline is None:
            print(f"Loading AudioLDM2 model on {self.device}...")
            self.pipeline = AudioLDM2Pipeline.from_pretrained(
                MUSIC_MODEL_ID, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.pipeline.to(self.device)

    def generate(self, prompt: str, duration: int, output_path: str):
        """
        Generates music based on the prompt.
        """
        self.load_model()
        
        # AudioLDM 2 takes 'prompt' and generates audio
        # num_inference_steps=200 for high quality, lower for speed. 
        # audio_length_in_s sets duration.
        
        print(f"Generating music for prompt: '{prompt}'...")
        audio = self.pipeline(
            prompt, 
            num_inference_steps=50, 
            audio_length_in_s=duration
        ).audios[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, 16000) # AudioLDM usually outputs 16kHz
        return output_path

if __name__ == "__main__":
    # Test stub
    eng = MusicEngine()
    test_file = os.path.join(TEMP_DIR, "test_music.wav")
    eng.generate("Lo-fi hip hop beat, chill, relax", 10, test_file)
    print(f"Generated: {test_file}")
