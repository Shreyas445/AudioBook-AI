import json
import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


def _strip_markdown_emphasis(text: str) -> str:
    """Strips **bold**, *italic*, __underline__, and # headings from spoken text so TTS never reads symbols."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    s = re.sub(r"__(.+?)__", r"\1", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", s)
    s = re.sub(r"(?m)^#{1,6}\s+", "", s)
    return s.strip()

from config import (
    AMBIENCE_PROMPTS,
    BGM_THEME_PROMPTS,
    SFX_PROMPTS,
    VOICE_CATALOG,
)
from core.llm_client import UnifiedLLMClient, extract_and_parse_json_objects


BGM_KEYS = list(BGM_THEME_PROMPTS.keys())
AMBIENCE_KEYS = list(AMBIENCE_PROMPTS.keys())
SFX_KEYS = list(SFX_PROMPTS.keys())


class CharacterSpec(BaseModel):
    name: str = Field(..., description="Character name, e.g. 'Narrator', 'Aarav', 'Lily'.")
    voice_profile: str = Field(
        default="clear expressive adult storyteller",
        description="Rich persona & vocal description, e.g. 'Indian 30y man, warm & confident', '10y old curious girl', '70y grumpy British man'.",
    )
    gender: str = Field(default="neutral", description="'male', 'female', or 'neutral'.")
    age_group: str = Field(default="adult", description="'child', 'teen', 'young_adult', 'adult', or 'elderly'.")
    accent: str = Field(default="american", description="'american', 'british', or 'indian'.")
    voice_id: Optional[str] = Field(
        default=None,
        description="Optional Kokoro voice ID override (e.g. 'af_heart', 'am_michael', 'hm_omega', 'bf_emma', 'af_sky').",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        out["name"] = str(out.get("name") or out.get("speaker") or out.get("character") or "Narrator").strip()
        out["voice_profile"] = str(
            out.get("voice_profile")
            or out.get("description")
            or out.get("character_profile")
            or out.get("persona")
            or out.get("voice_description")
            or "expressive adult"
        ).strip()
        out["gender"] = str(out.get("gender") or "neutral").strip()
        out["age_group"] = str(out.get("age_group") or out.get("age") or "adult").strip()
        out["accent"] = str(out.get("accent") or "american").strip()
        if not out.get("voice_id") or str(out.get("voice_id")).strip().lower() in ("none", "null", "auto"):
            out["voice_id"] = None
        return out


class DialogueBlock(BaseModel):
    speaker: str = Field(..., description="Speaker name, e.g. 'Narrator', 'Aarav', 'Lily'.")
    text: str = Field(..., description="Exact spoken narration or dialogue line.")
    voice_profile: str = Field(
        default="",
        description="How the character sounds, e.g. 'grumpy 65y man', '10y old happy girl', 'Indian 30y man, calm', 'whispering narrator'.",
    )
    gender: str = Field(default="", description="'male', 'female', or 'neutral'.")
    age_group: str = Field(default="", description="'child', 'teen', 'young_adult', 'adult', or 'elderly'.")
    accent: str = Field(default="", description="'american', 'british', or 'indian'.")
    emotion: str = Field(
        default="neutral",
        description="Performance tone: 'neutral', 'happy', 'sad', 'angry', 'grumpy', 'whisper', 'fearful', 'excited', 'tense', 'tender', 'sarcastic', 'shouting'.",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Optional Kokoro voice_id (e.g. 'af_heart', 'af_sky', 'am_michael', 'am_fenrir', 'bm_george', 'hm_omega', 'hf_alpha').",
    )
    bg_music: str = Field(
        default="peaceful_calm",
        description=f"Scene music theme ({', '.join(BGM_KEYS)}) OR a short instrumental mood phrase. Keep consistent across a scene!",
    )
    music_intensity: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Emotional intensity of the music from 0.0 (silent) to 1.0 (climactic peak; still auto-ducked below voice).",
    )
    ambience: str = Field(
        default="none",
        description=f"Subtle continuous room/environment bed ({', '.join(AMBIENCE_KEYS)}) or 'none'.",
    )
    sfx: str = Field(
        default="none",
        description=f"Optional spot sound effect ({', '.join(SFX_KEYS)}) or short foley phrase, or 'none'. Use sparingly at key moments!",
    )
    sfx_timing: Literal["before", "start", "under", "after"] = Field(
        default="before",
        description="Exact timing for SFX relative to this spoken line: 'before' (dramatic beat before speaking), 'start' (at first word), 'under' (beneath speech), 'after' (right after speaking).",
    )
    pause_after_ms: int = Field(
        default=280,
        ge=80,
        le=2500,
        description="Natural pause after this line in milliseconds (e.g. 160 for fast banter, 280 normal, 650 scene/paragraph shift).",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_block_aliases(cls, data: Any) -> Any:
        """Accepts flexible JSON field names when users paste JSON from ChatGPT, Claude, Gemini, etc."""
        if not isinstance(data, dict):
            return data
        out = dict(data)

        out["speaker"] = str(out.get("speaker") or out.get("character") or out.get("name") or out.get("role") or "Narrator").strip()
        raw_txt = str(out.get("text") or out.get("line") or out.get("dialogue") or out.get("content") or out.get("speech") or "").strip()
        out["text"] = _strip_markdown_emphasis(raw_txt)

        out["voice_profile"] = str(
            out.get("voice_profile")
            or out.get("character_profile")
            or out.get("voice_description")
            or out.get("voice_style")
            or out.get("character_voice")
            or out.get("persona")
            or ""
        ).strip()
        out["gender"] = str(out.get("gender") or "").strip()
        out["age_group"] = str(out.get("age_group") or out.get("age") or "").strip()
        out["accent"] = str(out.get("accent") or "").strip()
        out["emotion"] = str(out.get("emotion") or out.get("tone") or "neutral").strip()

        if not out.get("voice_id") or str(out.get("voice_id")).strip().lower() in ("none", "null", "auto"):
            out["voice_id"] = None

        out["bg_music"] = str(
            out.get("bg_music")
            or out.get("bg_music_prompt")
            or out.get("bgm")
            or out.get("music")
            or out.get("background_music")
            or "peaceful_calm"
        ).strip()

        out["ambience"] = str(
            out.get("ambience") or out.get("ambient") or out.get("environment") or out.get("room_tone") or "none"
        ).strip()

        out["sfx"] = str(
            out.get("sfx") or out.get("sound_effect") or out.get("foley") or "none"
        ).strip()

        raw_timing = str(out.get("sfx_timing") or "before").strip().lower()
        timing_map = {
            "pre": "before",
            "before": "before",
            "intro": "before",
            "start": "start",
            "with": "start",
            "during": "under",
            "under": "under",
            "background": "under",
            "end": "after",
            "after": "after",
            "post": "after",
        }
        out["sfx_timing"] = timing_map.get(raw_timing, "before")

        # Clamp numeric fields safely if LLM outputs strings or out-of-range numbers
        try:
            mi = float(out.get("music_intensity", 0.55))
            if mi > 1.0 and mi <= 100.0:
                mi = mi / 100.0
            out["music_intensity"] = max(0.0, min(1.0, mi))
        except Exception:
            out["music_intensity"] = 0.55

        try:
            pa = int(float(out.get("pause_after_ms", out.get("pause_ms", 280))))
            out["pause_after_ms"] = max(80, min(2500, pa))
        except Exception:
            out["pause_after_ms"] = 280

        return out


def _flatten_items(seq: Any) -> List[Any]:
    """Recursively flattens nested lists while preserving both raw dicts and validated Pydantic BaseModel instances."""
    out: List[Any] = []
    if isinstance(seq, (dict, BaseModel)):
        out.append(seq)
    elif isinstance(seq, (list, tuple)):
        for item in seq:
            out.extend(_flatten_items(item))
    return out


def _is_dialogue_item(item: Any) -> bool:
    if isinstance(item, DialogueBlock):
        return bool(item.text and item.text.strip())
    if isinstance(item, dict):
        return any(bool(item.get(k)) for k in ("text", "line", "dialogue", "content", "speech"))
    return False


def _is_character_item(item: Any) -> bool:
    if isinstance(item, CharacterSpec):
        return True
    if isinstance(item, dict) and not _is_dialogue_item(item):
        return bool(item.get("name") or item.get("voice_profile") or item.get("voice_id"))
    return False


class ProductionScript(BaseModel):
    characters: List[CharacterSpec] = Field(
        default_factory=list,
        description="Cast list defining each speaker's consistent vocal persona, age, gender, and accent.",
    )
    blocks: List[DialogueBlock] = Field(
        default_factory=list,
        description="Ordered sequence of narration and dialogue blocks with synchronized BGM, ambience, and SFX.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_root_aliases(cls, data: Any) -> Any:
        if isinstance(data, list):
            flat = _flatten_items(data)
            chars = [x for x in flat if _is_character_item(x)]
            blks = [x for x in flat if _is_dialogue_item(x)]
            return {"characters": chars, "blocks": blks}
        if not isinstance(data, dict):
            return data
        out = dict(data)
        raw_chars = out.get("characters") or out.get("cast") or out.get("speakers") or []
        out["characters"] = [x for x in _flatten_items(raw_chars) if isinstance(x, (dict, CharacterSpec))]

        if "blocks" in out:
            out["blocks"] = [x for x in _flatten_items(out["blocks"]) if _is_dialogue_item(x)]
        elif "scenes" in out and isinstance(out["scenes"], list):
            flat_blocks = []
            for sc in _flatten_items(out["scenes"]):
                if isinstance(sc, dict):
                    scene_bgm = sc.get("bg_music") or sc.get("bgm") or sc.get("music")
                    scene_amb = sc.get("ambience") or sc.get("ambient")
                    sub = sc.get("blocks") or sc.get("dialogue") or sc.get("lines") or []
                    for b in _flatten_items(sub):
                        if isinstance(b, dict) and _is_dialogue_item(b):
                            b_copy = dict(b)
                            if scene_bgm and "bg_music" not in b_copy and "bgm" not in b_copy:
                                b_copy["bg_music"] = scene_bgm
                            if scene_amb and "ambience" not in b_copy and "ambient" not in b_copy:
                                b_copy["ambience"] = scene_amb
                            flat_blocks.append(b_copy)
                        elif isinstance(b, DialogueBlock):
                            flat_blocks.append(b)
            out["blocks"] = flat_blocks
        else:
            alt_blocks = (
                out.get("dialogue")
                or out.get("lines")
                or out.get("script")
                or out.get("segments")
                or ([] if not _is_dialogue_item(out) else [out])
            )
            out["blocks"] = [x for x in _flatten_items(alt_blocks) if _is_dialogue_item(x)]
        return out


def parse_script_objects(objects: List[Any]) -> ProductionScript:
    """
    Merges one or more parsed JSON objects/lists into a single validated ProductionScript,
    gracefully handling separate character lists, nested block lists, or multi-chunk pastes.
    """
    merged_chars: Dict[str, CharacterSpec] = {}
    merged_blocks: List[DialogueBlock] = []

    for obj in objects:
        part = ProductionScript.model_validate(obj)
        for ch in part.characters:
            key = ch.name.strip().lower()
            if key not in merged_chars:
                merged_chars[key] = ch
        for blk in part.blocks:
            if blk.text and blk.text.strip():
                merged_blocks.append(blk)

    if not merged_blocks:
        raise ValueError("Parsed JSON succeeded, but no non-empty spoken dialogue 'blocks' were found.")

    # Propagate cast traits onto blocks that omitted inline voice_profile / gender / accent / age_group
    for blk in merged_blocks:
        spec = merged_chars.get(blk.speaker.strip().lower())
        if spec:
            if not blk.voice_profile:
                blk.voice_profile = spec.voice_profile
            if not blk.gender:
                blk.gender = spec.gender
            if not blk.age_group:
                blk.age_group = spec.age_group
            if not blk.accent:
                blk.accent = spec.accent
            if not blk.voice_id and spec.voice_id:
                blk.voice_id = spec.voice_id

    return ProductionScript(characters=list(merged_chars.values()), blocks=merged_blocks)


def parse_pasted_script_json(raw_json_text: str) -> ProductionScript:
    """
    Parses user-pasted JSON from ANY external AI (ChatGPT, Claude, Gemini, Grok, Nemotron).
    Automatically merges multiple pasted chunk JSONs into one unified ProductionScript!
    """
    objects = extract_and_parse_json_objects(raw_json_text)
    return parse_script_objects(objects)


DIRECTOR_SYSTEM_PROMPT = f"""You are a Master Cinema Audiobook Director & Sound Designer.
Convert the provided book passage into a richly expressive, perfectly timed JSON production script.

CRITICAL DIRECTION RULES:
1. COMPLETE & FAITHFUL TEXT: Preserve every story event and spoken line faithfully. Split narration vs. spoken character dialogue into separate sequential `blocks` so each character speaks in their own voice! Do NOT include "he whispered" inside the character's spoken quote—put narration tags in a 'Narrator' block or omit trivial attribution if obvious, keeping the flow natural.
2. RICH CHARACTER VOICES (`characters` & `voice_profile`):
   - Give every speaker a vivid `voice_profile` describing age, nationality/accent, gender, and personality (examples: "Narrator, warm cinematic storyteller", "Indian 30y man, calm & confident", "10y old girl, bubbly & curious", "70y grumpy British man, raspy", "teenage boy, nervous & fast").
   - Set `gender` ("male" | "female" | "neutral"), `age_group` ("child" | "teen" | "young_adult" | "adult" | "elderly"), and `accent` ("american" | "british" | "indian").
   - Optional `voice_id` from Kokoro catalog: {', '.join(VOICE_CATALOG.keys())}.
   - Set `emotion` per line ("neutral", "happy", "sad", "angry", "grumpy", "whisper", "fearful", "excited", "tense", "tender", "sarcastic", "shouting").
3. EMOTIONAL SCENE MUSIC (`bg_music` & `music_intensity`):
   - Choose `bg_music` from: [{', '.join(BGM_KEYS)}].
   - Keep the SAME `bg_music` across all blocks of a scene so the music flows continuously! Switch ONLY when the emotional mood genuinely shifts (e.g. `peaceful_calm` -> `suspense_tense` -> `intense` -> `warm_heartfelt`).
   - Set `music_intensity` (0.0 to 1.0): e.g. 0.35 for quiet dialogue, 0.55 for normal mood, 0.85 for intense/emotional climax.
4. SUBTLE AMBIENCE (`ambience`):
   - Choose a continuous environmental layer from: [{', '.join(AMBIENCE_KEYS)}]. Keep consistent within the same physical location!
5. PERFECTLY TIMED SPOT SFX (`sfx` & `sfx_timing`):
   - Use `sfx` ONLY on moments where a physical sound truly enhances immersion (roughly 1 in 4–8 blocks; set `"none"` otherwise!).
   - Choose from [{', '.join(SFX_KEYS)}] (or a concise 2-3 word foley tag like `"wooden_chair_scrape"`).
   - Set `sfx_timing`:
     * `"before"` -> sound plays in a dramatic beat right BEFORE the character speaks (e.g. `door_knock` or `thunder_clap` before "Who's there?").
     * `"start"` -> sound triggers as the line begins.
     * `"under"` -> continuous action foley softly underneath the line (e.g. `footsteps_running`).
     * `"after"` -> sound lands right AFTER the final word (e.g. `door_slam` after "Get out!").
6. OUTPUT STRICTLY VALID JSON ONLY (no markdown prose outside JSON). Never put unescaped double quotes inside `"text"`—use single quotes `'...'` or escape `\"` if quoting inside `"text"`. Match this structure:
{{
  "characters": [
    {{
      "name": "Narrator",
      "voice_profile": "Warm cinematic adult storyteller",
      "gender": "female",
      "age_group": "adult",
      "accent": "american",
      "voice_id": "af_heart"
    }}
  ],
  "blocks": [
    {{
      "speaker": "Narrator",
      "text": "Rain lashed against the Mumbai apartment window as the clock struck midnight.",
      "voice_profile": "Warm cinematic adult storyteller",
      "gender": "female",
      "age_group": "adult",
      "accent": "american",
      "emotion": "tense",
      "voice_id": "af_heart",
      "bg_music": "suspense_tense",
      "music_intensity": 0.6,
      "ambience": "gentle_rain",
      "sfx": "thunder_clap",
      "sfx_timing": "before",
      "pause_after_ms": 400
    }}
  ]
}}"""


def build_standalone_ai_prompt(custom_direction: str = "") -> str:
    """
    Generates a universal, story-free Cinema Audiobook Director Prompt!
    Users upload/attach their own story file (PDF/TXT/DOCX) directly inside ChatGPT, Claude, Gemini, or Grok,
    paste this prompt alongside their uploaded story, and get clean production JSON back.
    """
    extra = (
        f"\nADDITIONAL CREATIVE DIRECTION FROM USER:\n- {custom_direction.strip()}\n"
        if custom_direction and custom_direction.strip()
        else ""
    )
    return (
        f"{DIRECTOR_SYSTEM_PROMPT}\n"
        f"{extra}\n"
        "=== YOUR TASK ===\n"
        "Read the story / book file or text I have uploaded or attached with this message.\n"
        "Convert the entire passage faithfully into the exact JSON format above with expressive character "
        "`voice_profile` descriptions (age, accent, gender, mood), continuous scene `bg_music`, subtle `ambience`, "
        "and well-timed `sfx`.\n"
        "Output ONLY raw valid JSON inside a single ```json code block—no extra commentary."
    )


def build_copyable_ai_prompt(
    chunk_text: str = "",
    chunk_index: int = 1,
    total_chunks: int = 1,
    continuity_note: str = "",
    custom_direction: str = "",
) -> str:
    """
    Creates a ready-to-copy Master Prompt. If chunk_text is omitted, returns the standalone
    prompt for users who attach their story directly in their own AI chat!
    """
    if not chunk_text or not chunk_text.strip():
        return build_standalone_ai_prompt(custom_direction=custom_direction)

    part_header = (
        f"### PART {chunk_index} OF {total_chunks}\n"
        "IMPORTANT: Keep exact `speaker` names, `voice_profile`, `accent`, `age_group`, and `voice_id` "
        "100% consistent across all parts! Output ONLY raw valid JSON inside a single ```json code block.\n"
        if total_chunks > 1
        else "Output ONLY raw valid JSON inside a single ```json code block.\n"
    )
    extra = f"\nCUSTOM DIRECTION: {custom_direction.strip()}\n" if custom_direction and custom_direction.strip() else ""
    ctx_block = f"\nPREVIOUS SCENE / CAST CONTINUITY:\n{continuity_note}\n" if continuity_note else ""

    return (
        f"{DIRECTOR_SYSTEM_PROMPT}\n\n"
        f"{part_header}"
        f"{extra}"
        f"{ctx_block}\n"
        f"=== BOOK PASSAGE TO CONVERT (PART {chunk_index}/{total_chunks}) ===\n"
        f"{chunk_text.strip()}\n"
        f"=== END OF PASSAGE ==="
    )


class ScriptGenerator:
    """Drives multi-chunk script generation with persistent character & scene continuity across chunks."""

    def __init__(self, llm_client: Optional[UnifiedLLMClient] = None):
        self.llm_client = llm_client or UnifiedLLMClient(provider="Ollama (Local - Optional)")

    def generate_script(
        self,
        text_chunk: str,
        known_cast: Optional[Dict[str, CharacterSpec]] = None,
        last_scene_state: Optional[Dict[str, str]] = None,
        unload_after: bool = False,
    ) -> ProductionScript:
        continuity_lines = []
        if known_cast:
            cast_summary = [
                f"- {c.name}: profile='{c.voice_profile}', gender={c.gender}, age={c.age_group}, accent={c.accent}, voice_id={c.voice_id or 'auto'}"
                for c in known_cast.values()
            ]
            continuity_lines.append("Established Character Cast (MUST reuse exact names & voice traits):\n" + "\n".join(cast_summary))

        if last_scene_state:
            continuity_lines.append(
                f"Previous chunk ended with bg_music='{last_scene_state.get('bg_music', 'peaceful_calm')}', "
                f"ambience='{last_scene_state.get('ambience', 'none')}', "
                f"last_speaker='{last_scene_state.get('speaker', 'Narrator')}'. Maintain continuity unless the scene changes."
            )

        context_str = "\n\n".join(continuity_lines) if continuity_lines else "Start of book."
        user_prompt = f"CONTINUITY CONTEXT:\n{context_str}\n\nBOOK TEXT TO DIRECT:\n{text_chunk}"

        parsed_list = self.llm_client.generate_json(
            system_prompt=DIRECTOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=ProductionScript.model_json_schema(),
            unload_after=unload_after,
        )

        return parse_script_objects(parsed_list)
