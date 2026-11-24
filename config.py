import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
SCRIPTS_DIR = os.path.join(OUTPUT_DIR, "scripts")
AUDIO_LIB_DIR = os.path.join(OUTPUT_DIR, "audio_library")
BGM_LIB_DIR = os.path.join(AUDIO_LIB_DIR, "bgm")
AMBIENCE_LIB_DIR = os.path.join(AUDIO_LIB_DIR, "ambience")
SFX_LIB_DIR = os.path.join(AUDIO_LIB_DIR, "sfx")
FINAL_FILENAME = "audiobook.mp3"

# LLM Provider Configurations
LLM_PROVIDERS = [
    "NVIDIA NIM API",
    "Google Gemini API",
    "OpenAI API",
    "Ollama (Local - Optional)",
]

PROVIDER_MODELS = {
    "NVIDIA NIM API": [
        "nvidia/nemotron-3.5-bolt-e4b",
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "nvidia/llama-3.3-nemotron-super-49b-v1",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    ],
    "Google Gemini API": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-3-pro-preview",
        "gemini-3.8-flash",
    ],
    "OpenAI API": [
        "gpt-5-mini",
        "gpt-5",
        "gpt-4o",
        "gpt-4o-mini",
        "o4-mini",
        "o3",
    ],
    "Ollama (Local - Optional)": [
        os.getenv("OLLAMA_MODEL", "llama3"),
        "llama3.1",
        "qwen2.5",
        "deepseek-r1",
        "mistral",
    ],
}

DEFAULT_BASE_URLS = {
    "NVIDIA NIM API": os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    "Google Gemini API": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
    "OpenAI API": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "Ollama (Local - Optional)": os.getenv("OLLAMA_URL", "http://localhost:11434"),
}

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Kokoro-82M TTS Settings & Quality-Graded Multi-Accent Catalog
# `quality_bonus` reflects official hexgrad/Kokoro-82M VOICES.md training quality & fidelity grades
TTS_REPO_ID = "hexgrad/Kokoro-82M"
TTS_DEFAULT_VOICE = "af_heart"
TTS_FALLBACK_VOICE = "af_bella"
TTS_LANG_CODE = "a"
SAMPLE_RATE = 24000

VOICE_CATALOG = {
    # American Female ('a') — Highest fidelity & prosodic range
    "af_heart": {"gender": "female", "accent": "american", "age": "adult", "quality": 26, "traits": "warm, expressive, dramatic, narrator, heartfelt"},
    "af_bella": {"gender": "female", "accent": "american", "age": "young_adult", "quality": 22, "traits": "clear, natural, expressive, heroine, knowing"},
    "af_nicole": {"gender": "female", "accent": "american", "age": "young_adult", "quality": 18, "traits": "ethereal, gentle, breathy, soft, whisper, ancient, mysterious"},
    "af_sarah": {"gender": "female", "accent": "american", "age": "adult", "quality": 16, "traits": "confident, steady, mature, commanding, poised"},
    "af_kore": {"gender": "female", "accent": "american", "age": "elderly", "quality": 15, "traits": "ancient, wise, regal, poised, queen, mature"},
    "af_sky": {"gender": "female", "accent": "american", "age": "child", "quality": 14, "traits": "bright, youthful, bubbly, girl, little, sister"},
    "af_nova": {"gender": "female", "accent": "american", "age": "teen", "quality": 14, "traits": "energetic, lively, spirited, young, teen"},
    "af_aoede": {"gender": "female", "accent": "american", "age": "adult", "quality": 15, "traits": "lyrical, theatrical, musical, siren, mermaid"},
    "af_river": {"gender": "female", "accent": "american", "age": "adult", "quality": 13, "traits": "smooth, calm, reflective, mellow"},
    "af_jessica": {"gender": "female", "accent": "american", "age": "adult", "quality": 12, "traits": "friendly, conversational, cheerful"},
    "af_alloy": {"gender": "female", "accent": "american", "age": "adult", "quality": 12, "traits": "balanced, crisp, neutral"},
    # American Male ('a') — Ranked by official VOICES.md fidelity
    "am_fenrir": {"gender": "male", "accent": "american", "age": "adult", "quality": 20, "traits": "gravelly, commanding, pirate, intense, grumpy, dark, determined, warrior"},
    "am_michael": {"gender": "male", "accent": "american", "age": "adult", "quality": 20, "traits": "deep, warm, narrator, confident, resonant, leader"},
    "am_puck": {"gender": "male", "accent": "american", "age": "young_adult", "quality": 18, "traits": "expressive, witty, sarcastic, playful, boy, teen, energetic"},
    "am_echo": {"gender": "male", "accent": "american", "age": "young_adult", "quality": 14, "traits": "gentle, soft-spoken, thoughtful, mysterious"},
    "am_adam": {"gender": "male", "accent": "american", "age": "adult", "quality": 11, "traits": "grounded, steady, protagonist"},
    "am_eric": {"gender": "male", "accent": "american", "age": "adult", "quality": 11, "traits": "firm, serious, analytical"},
    "am_liam": {"gender": "male", "accent": "american", "age": "young_adult", "quality": 10, "traits": "friendly, young man, casual"},
    "am_onyx": {"gender": "male", "accent": "american", "age": "elderly", "quality": 12, "traits": "deep, stoic, heavy, older"},
    "am_santa": {"gender": "male", "accent": "american", "age": "elderly", "quality": 6, "traits": "grandfatherly, aged"},
    # British Female ('b')
    "bf_emma": {"gender": "female", "accent": "british", "age": "adult", "quality": 20, "traits": "refined, articulate, expressive, british narrator, royal"},
    "bf_isabella": {"gender": "female", "accent": "british", "age": "young_adult", "quality": 15, "traits": "warm, graceful, ethereal, princess"},
    "bf_lily": {"gender": "female", "accent": "british", "age": "child", "quality": 14, "traits": "light, young, gentle, british girl"},
    "bf_alice": {"gender": "female", "accent": "british", "age": "elderly", "quality": 12, "traits": "classic, mature, wise, stern"},
    # British Male ('b')
    "bm_fable": {"gender": "male", "accent": "british", "age": "adult", "quality": 17, "traits": "dramatic, theatrical, expressive, pirate, sarcastic, commanding"},
    "bm_george": {"gender": "male", "accent": "british", "age": "elderly", "quality": 16, "traits": "authoritative, grumpy, distinguished, captain, older"},
    "bm_lewis": {"gender": "male", "accent": "british", "age": "adult", "quality": 15, "traits": "natural, steady, determined"},
    "bm_daniel": {"gender": "male", "accent": "british", "age": "adult", "quality": 14, "traits": "crisp, composed, smooth"},
    # Indian / South Asian ('h' voice weights)
    "hf_alpha": {"gender": "female", "accent": "indian", "age": "adult", "quality": 16, "traits": "indian woman, clear, warm, expressive"},
    "hf_beta": {"gender": "female", "accent": "indian", "age": "young_adult", "quality": 15, "traits": "indian young woman, bright, gentle"},
    "hm_omega": {"gender": "male", "accent": "indian", "age": "adult", "quality": 16, "traits": "indian man, resonant, confident, 30s"},
    "hm_psi": {"gender": "male", "accent": "indian", "age": "young_adult", "quality": 15, "traits": "indian young man, conversational, warm"},
}

VALID_VOICES = list(VOICE_CATALOG.keys())

# Music, Ambience & SFX Generation Settings (6 GB GPU Safe + Hi-Fi Mastered)
MUSIC_MODEL_ID = "cvssp/audioldm2"
MUSIC_DURATION = 12       # Seconds per BGM score loop (seamlessly crossfaded)
AMBIENCE_DURATION = 8     # Seconds per ambient bed loop
SFX_DURATION = 3          # Seconds per spot foley sound effect

AUDIO_NEGATIVE_PROMPT = (
    "human speech, talking, vocals, singing, words, choir, muffled, lo-fi, "
    "telephone quality, static hiss, white noise, distortion, clipping, harsh"
)

# High-Definition Orchestral & Cinema Score Prompts
BGM_THEME_PROMPTS = {
    "none": None,
    "peaceful_calm": "warm felt piano melody with lush solo cello and soft ambient pad, studio film score recording, high fidelity, clean instrumental",
    "happy": "uplifting warm acoustic guitar, sparkling harp and joyful lyrical strings, heartwarming cinema soundtrack, studio recording, instrumental",
    "warm_heartfelt": "intimate emotional piano and lyrical warm cello duet, tender moving cinematic score, rich harmonics, clean studio instrumental",
    "sad_emotional": "poignant solo violin and sparse resonant piano in minor key, deep emotional film score, warm concert hall acoustics, instrumental",
    "suspense_tense": "deep pulsing cello ostinato, dark bowed double bass and subtle thriller tension pad, clean cinema underscore, high fidelity instrumental",
    "intense": "urgent dramatic staccato string orchestra, deep cinematic taiko pulse and brass tension, high-stakes film score, studio quality instrumental",
    "epic_action": "epic orchestral adventure score, powerful cinematic drums, heroic brass swells and soaring strings, blockbuster soundtrack, clean instrumental",
    "mystery": "celestial harp harmonics, shimmering glass celesta and deep mysterious underwater pad, enchanting cinema score, studio quality instrumental",
    "dark_ominous": "deep sub-bass orchestral drone, haunting low brass and dark atmospheric cello textures, ominous cinema score, clean instrumental",
    "romantic": "lush sweeping romantic string orchestra, tender piano melody and warm french horn, emotional love theme soundtrack, studio instrumental",
    "playful_whimsical": "charming pizzicato strings, delicate celesta and warm woodwinds, magical storybook film score, crisp studio instrumental",
    "triumphant": "soaring noble french horns, uplifting full string orchestra and warm timpani, victorious emotional finale score, studio instrumental",
}

AMBIENCE_PROMPTS = {
    "none": None,
    "ocean_waves": "calm deep ocean swells gently lapping against wooden ship hull, soothing coastal sea air, clean studio nature recording",
    "gentle_rain": "soft steady rain pattering gently outside window, warm soothing weather atmosphere, clean field recording",
    "thunderstorm": "steady rainstorm with deep distant rolling thunder on horizon, clean atmospheric weather recording",
    "forest_wind": "soft breeze rustling through lush forest leaves, peaceful woodland air, clean nature recording",
    "night_crickets": "quiet evening crickets in calm night air, peaceful nocturnal ambience, clean field recording",
    "crackling_fireplace": "warm cozy hearth fire softly crackling, gentle wood embers, clean indoor recording",
    "busy_cafe": "soft distant warm room murmur with subtle porcelain clinks, quiet indoor atmosphere",
    "city_street": "quiet distant nighttime city horizon hum, subtle urban atmosphere",
    "castle_hall": "deep quiet stone cathedral hall room tone, subtle airy resonance",
    "spaceship_hum": "deep smooth low-frequency spacecraft cabin hum, clean sci-fi room tone",
}

SFX_PROMPTS = {
    "none": None,
    "thunder_clap": "deep resonant distant thunder rumble rolling smoothly, clean cinema foley",
    "door_creak": "heavy old wooden door slowly creaking open, crisp studio foley sound effect",
    "door_slam": "heavy solid wooden door slamming shut with deep thud, crisp studio foley",
    "door_knock": "three clear resonant knocks on a heavy wooden door, crisp studio foley",
    "footsteps_slow": "measured footsteps walking on wooden deck planks, crisp foley",
    "footsteps_running": "urgent fast footsteps running across stone floor, crisp foley",
    "glass_shatter": "crystalline glass shattering cleanly on stone floor, studio foley",
    "sword_clash": "ringing steel sword blade unsheathing and clashing sharply, cinema foley",
    "heartbeat": "deep slow dramatic heartbeat pulse, clean sub-bass cinema sound effect",
    "phone_ring": "classic bell telephone ringing twice clearly, clean indoor foley",
    "car_engine": "smooth engine ignition and low idle purr, clean foley",
    "wind_gust": "deep atmospheric gust of sea wind sweeping past sails, clean cinema foley",
    "sigh_gasp": "soft dramatic breath whoosh swell transition, subtle cinema sound effect",
    "magic_chime": "shimmering ethereal crystal harmonic bell chord, warm magical cinema sound effect",
}

# Mixing & Mastering Constants (Dynamic Voice Expression + Crisp Ducking)
VOICE_TARGET_DBFS = -2.8        # Baseline narrator/dialogue peak target
BGM_OPEN_DBFS = -14.8           # Rich, felt emotional score in pauses & scene transitions
BGM_DUCKED_DBFS = -23.0         # Cleanly ducked score beneath active dialogue
AMBIENCE_OPEN_DBFS = -22.5      # Subtle atmospheric immersion in pauses
AMBIENCE_DUCKED_DBFS = -29.0    # Whisper-quiet room/ocean bed under speech
SFX_MAX_DBFS = -10.5            # Spot SFX ceiling in open beats (never louder than voice!)
SFX_UNDER_SPEECH_DBFS = -19.5   # Spot SFX ceiling when overlapping dialogue
MIN_VOICE_TO_BG_RATIO_DB = 13.5 # Hard guarantee: Voice RMS >= (BGM + Amb + SFX) RMS + 13.5 dB
