from pydantic import BaseModel, Field
from typing import List, Optional
import ollama
import json
from config import OLLAMA_MODEL

class DialogueBlock(BaseModel):
    speaker: str = Field(..., description="Name of the speaker, e.g., 'Narrator', 'Alice'.")
    voice_id: str = Field(..., description="Voice ID to use. Options: 'af_bella' (female), 'am_adam' (male), 'bf_emma' (brit female), 'bm_george' (brit male). Choose based on character gender/vibe.")
    text: str = Field(..., description="The spoken text.")
    emotion: str = Field(..., description="Tone of voice, e.g., 'neutral', 'whisper', 'shouting', 'sad'.")
    bg_music_prompt: str = Field(..., description="Prompt for background music, e.g., 'quiet ambient rain', 'tense cinematic strings', 'joyful acoustic guitar'. Keep consistent for a scene.")

class ProductionScript(BaseModel):
    blocks: List[DialogueBlock]

class ScriptGenerator:
    def __init__(self, model_name=OLLAMA_MODEL):
        self.model_name = model_name

    def generate_script(self, text_chunk: str, context: str = "") -> ProductionScript:
        """
        Converts a raw text chunk into a structured ProductionScript using Ollama.
        """
        music_themes = "Calm, Sad, Tense, Mystery, Action, Happy, Romance, Dark_Ambient"
        
        system_prompt = (
            "You are an expert audiobook director. Convert the text into a structured JSON script.\n"
            "Rules:\n"
            "1. Assign `voice_id` based on character (af_bella=female, am_adam=male, etc).\n"
            "2. `bg_music_prompt`: Choose ONE general theme from this list: "
            f"[{music_themes}]. Use 'None' only for absolute silence. "
            "Reuse the SAME theme tag effectively for an entire scene to ensure continuity.\n"
            "3. Output MUST be valid JSON matching the schema."
        )
        
        user_prompt = f"Context: {context}\n\nText: {text_chunk}"

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            format=ProductionScript.model_json_schema(),
            options={'temperature': 0.2} # Lower temp for consistent JSON and tags
        )

        try:
            # The response content should be valid JSON matching the schema
            content = response['message']['content']
            # Basic cleanup if model adds markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.replace("```", "").strip()
                
            script = ProductionScript.model_validate_json(content)
            return script
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            raise e

if __name__ == "__main__":
    # Test stub
    gen = ScriptGenerator()
    sample_text = "The rain beat against the window. 'I can't believe it,' whispered Sarah."
    script = gen.generate_script(sample_text)
    print(script.model_dump_json(indent=2))
