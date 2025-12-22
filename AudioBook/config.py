import os

# Ollama Settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# TTS Settings
# Kokoro voices: af_bella, af_sarah, am_adam, am_michael, etc.
TTS_DEFAULT_VOICE = "af_bella" 
TTS_LANG_CODE = "a" # 'a' for American English, 'b' for British
VALID_VOICES = ["af_bella", "af_sarah", "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]

# Audio Settings
OUTPUT_DIR = "output"
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
FINAL_FILENAME = "audiobook.mp3"
SAMPLE_RATE = 24000

# Music Settings
MUSIC_MODEL_ID = "cvssp/audioldm2" # HuggingFace model ID
MUSIC_DURATION = 10 # Seconds (will be looped)
