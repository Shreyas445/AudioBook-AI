import json
import mimetypes
import os
import queue
import re
import shutil
import threading
import time
import traceback
import urllib.parse
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from config import (
    AMBIENCE_PROMPTS,
    BASE_DIR,
    BGM_THEME_PROMPTS,
    DEFAULT_BASE_URLS,
    LLM_PROVIDERS,
    OUTPUT_DIR,
    PROVIDER_MODELS,
    SFX_PROMPTS,
    TTS_DEFAULT_VOICE,
    VOICE_CATALOG,
)
from core.llm_client import fetch_provider_models
from core.music_engine import normalize_cue_slug
from core.script_engine import (
    ProductionScript,
    build_copyable_ai_prompt,
    build_standalone_ai_prompt,
    parse_pasted_script_json,
)
from core.tts_engine import CharacterVoiceResolver
from main import AudiobookConverter, get_gpu_status_string, slugify_title
from utils.text_utils import chunk_text


WEB_DIR = os.path.join(BASE_DIR, "web")
UPLOADS_DIR = os.path.join(OUTPUT_DIR, "uploads")
ENV_LOCAL_PATH = os.path.join(BASE_DIR, ".env.local")
HOST = "127.0.0.1"
PORT = int(os.getenv("PORT", "7860"))

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_local_env() -> Dict[str, str]:
    stored: Dict[str, str] = {}
    if os.path.exists(ENV_LOCAL_PATH):
        try:
            with open(ENV_LOCAL_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and v:
                            stored[k] = v
                            os.environ.setdefault(k, v)
        except Exception:
            pass
    return stored


def _save_provider_key_locally(provider: str, api_key: str):
    key_map = {
        "NVIDIA NIM API": "NVIDIA_API_KEY",
        "Google Gemini API": "GEMINI_API_KEY",
        "OpenAI API": "OPENAI_API_KEY",
    }
    env_name = key_map.get(provider)
    if not env_name or not api_key.strip():
        return
    existing = _load_local_env()
    existing[env_name] = api_key.strip()
    os.environ[env_name] = api_key.strip()
    with open(ENV_LOCAL_PATH, "w", encoding="utf-8") as f:
        for k, v in existing.items():
            f.write(f"{k}={v}\n")


def _get_saved_key_status() -> Dict[str, str]:
    """Returns masked or saved keys for UI convenience on localhost."""
    return {
        "NVIDIA NIM API": os.getenv("NVIDIA_API_KEY", ""),
        "Google Gemini API": os.getenv("GEMINI_API_KEY", ""),
        "OpenAI API": os.getenv("OPENAI_API_KEY", ""),
        "Ollama (Local - Optional)": "",
    }


_load_local_env()
CONVERTER = AudiobookConverter()


def serialize_script_overview(script: ProductionScript, narrator_voice: str = TTS_DEFAULT_VOICE) -> Dict[str, Any]:
    """Builds structured JSON metadata for the Monochrome UI Cast Cards & Soundscape Badges."""
    resolver = CharacterVoiceResolver(narrator_default_voice=narrator_voice)
    for ch in script.characters:
        resolver.resolve(ch.name, ch.voice_profile, ch.gender, ch.age_group, ch.accent, ch.voice_id)

    counts: Dict[str, int] = {}
    profiles: Dict[str, str] = {}
    emotions_by_spk: Dict[str, List[str]] = {}
    bgm_cues: List[str] = []
    amb_cues: List[str] = []
    sfx_cues: List[Dict[str, str]] = []

    for b in script.blocks:
        resolver.resolve(b.speaker, b.voice_profile, b.gender, b.age_group, b.accent, b.voice_id)
        counts[b.speaker] = counts.get(b.speaker, 0) + 1
        if b.voice_profile and b.speaker not in profiles:
            profiles[b.speaker] = b.voice_profile
        emo_list = emotions_by_spk.setdefault(b.speaker, [])
        if b.emotion and b.emotion not in emo_list:
            emo_list.append(b.emotion)

        bgm = normalize_cue_slug(b.bg_music)
        if bgm != "none" and bgm not in bgm_cues:
            bgm_cues.append(bgm)
        amb = normalize_cue_slug(b.ambience)
        if amb != "none" and amb not in amb_cues:
            amb_cues.append(amb)
        sfx = normalize_cue_slug(b.sfx)
        if sfx != "none":
            sfx_cues.append({"cue": sfx, "timing": b.sfx_timing, "speaker": b.speaker})

    cast_cards = []
    for spk, line_count in counts.items():
        assign = resolver.assignments.get(spk.strip().lower())
        cast_cards.append({
            "speaker": spk,
            "initials": "".join(part[0].upper() for part in spk.split()[:2] if part) or "S",
            "profile": profiles.get(spk) or "Expressive character voice",
            "voice_id": assign.primary_voice if assign else narrator_voice,
            "secondary_voice": assign.secondary_voice if assign else None,
            "summary": assign.summary if assign else narrator_voice,
            "pitch_st": round(assign.base_pitch_semitones, 1) if assign else 0.0,
            "lines": line_count,
            "emotions": emotions_by_spk.get(spk, ["neutral"])[:4],
        })

    return {
        "total_blocks": len(script.blocks),
        "cast": cast_cards,
        "bgm_themes": bgm_cues,
        "ambiences": amb_cues,
        "sfx_cues": sfx_cues[:24],
        "sfx_total": len(sfx_cues),
    }


def _infer_stage_from_log(msg: str, prog: float) -> int:
    if "[Stage 1/4]" in msg or prog < 0.39:
        return 1
    if "[Stage 2/4]" in msg or "Creating BGM" in msg or "Creating AMBIENCE" in msg or "Creating SFX" in msg or "Synthesizing" in msg and "cue" in msg:
        return 2
    if "[Stage 3/4]" in msg or "Rendered [" in msg:
        return 3
    if "[Stage 4/4]" in msg or prog >= 0.88:
        return 4
    return 2


class JobState:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.events: List[Dict[str, Any]] = []
        self.done = False
        self.cond = threading.Condition()

    def push(self, event_type: str, payload: Dict[str, Any]):
        with self.cond:
            self.events.append({"type": event_type, "data": payload, "ts": time.time()})
            if event_type in ("complete", "job_error"):
                self.done = True
            self.cond.notify_all()


JOBS: Dict[str, JobState] = {}
JOBS_LOCK = threading.Lock()


def _start_conversion_job(mode: str, params: Dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = JobState(job_id)
    with JOBS_LOCK:
        # Keep memory tidy: retain last 10 jobs
        if len(JOBS) > 10:
            oldest = next(iter(JOBS))
            JOBS.pop(oldest, None)
        JOBS[job_id] = job

    def worker():
        try:
            soundscape_mode = params.get("soundscape_mode", "neural")
            _, _, free_bytes = shutil.disk_usage(BASE_DIR)
            free_gb = free_bytes / (1024**3)
            min_gb = 1.5 if soundscape_mode == "neural" else 0.4
            if free_gb < min_gb:
                raise RuntimeError(
                    f"Low disk space ({free_gb:.2f} GB free). At least {min_gb:.1f} GB required "
                    "(or switch Soundscape Engine to 'Procedural Synth')."
                )

            narrator_voice = params.get("narrator_voice", TTS_DEFAULT_VOICE)
            bgm_vol_db = float(params.get("bgm_volume_db", 0.0))
            enable_sfx = bool(params.get("enable_sfx", True))

            if mode == "auto":
                provider = params.get("provider", "NVIDIA NIM API")
                api_key = params.get("api_key", "")
                if params.get("save_key") and api_key:
                    _save_provider_key_locally(provider, api_key)

                stream = CONVERTER.stream_convert_from_text(
                    input_file=params.get("file_path") or None,
                    raw_text=params.get("raw_text") or None,
                    provider=provider,
                    model_name=params.get("model_name", ""),
                    api_key=api_key,
                    base_url=params.get("base_url", ""),
                    soundscape_mode=soundscape_mode,
                    narrator_voice=narrator_voice,
                    bgm_volume_db=bgm_vol_db,
                    enable_sfx=enable_sfx,
                    chunk_chars=int(params.get("chunk_chars", 2800)),
                )
            else:
                stream = CONVERTER.stream_convert_from_json(
                    raw_json_text=params.get("pasted_json", ""),
                    project_title=params.get("project_title") or "cinema_audiobook",
                    soundscape_mode=soundscape_mode,
                    narrator_voice=narrator_voice,
                    bgm_volume_db=bgm_vol_db,
                    enable_sfx=enable_sfx,
                )

            sent_overview = False
            for prog, msg, audio_file, script_json_str, _ in stream:
                stage = _infer_stage_from_log(msg, prog) if mode == "auto" else (2 if prog < 0.36 else (3 if prog < 0.88 else 4))
                overview = None
                if script_json_str and not sent_overview:
                    try:
                        parsed = parse_pasted_script_json(script_json_str)
                        overview = serialize_script_overview(parsed, narrator_voice=narrator_voice)
                        sent_overview = True
                    except Exception:
                        pass

                audio_url = None
                audio_filename = None
                if audio_file and os.path.exists(audio_file):
                    audio_filename = os.path.basename(audio_file)
                    audio_url = f"/api/audio/{urllib.parse.quote(audio_filename)}?t={int(time.time())}"

                job.push("progress", {
                    "progress": round(float(prog), 4),
                    "stage": stage,
                    "log": msg,
                    "overview": overview,
                    "script_json": script_json_str if overview else None,
                    "audio_url": audio_url,
                    "audio_filename": audio_filename,
                })

                if audio_url:
                    job.push("complete", {
                        "progress": 1.0,
                        "stage": 4,
                        "audio_url": audio_url,
                        "audio_filename": audio_filename,
                        "log": msg,
                    })
        except Exception as e:
            print(traceback.format_exc())
            job.push("job_error", {"message": str(e)})

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return job_id


class StudioRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args):
        # Keep terminal clean—suppress noisy static asset access logs
        if args and isinstance(args[0], str) and ("/api/jobs/" in args[0] or "/web/" in args[0] or args[0].startswith("GET / ")):
            return
        super().log_message(fmt, *args)

    def _send_json(self, payload: Any, status: int = 200):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}

    def _serve_static_file(self, file_path: str, download_name: Optional[str] = None):
        if not os.path.isfile(file_path):
            self._send_json({"error": "File not found"}, status=404)
            return

        ctype, _ = mimetypes.guess_type(file_path)
        if file_path.endswith(".js"):
            ctype = "application/javascript; charset=utf-8"
        elif file_path.endswith(".css"):
            ctype = "text/css; charset=utf-8"
        elif file_path.endswith(".html"):
            ctype = "text/html; charset=utf-8"
        elif not ctype:
            ctype = "application/octet-stream"

        file_size = os.path.getsize(file_path)
        range_header = self.headers.get("Range")

        # Full HTTP 206 Partial Content support for instant HTML5 <audio> scrubbing!
        if range_header and range_header.startswith("bytes="):
            m = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if m:
                start_s, end_s = m.groups()
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else file_size - 1
                start = max(0, min(start, file_size - 1))
                end = max(start, min(end, file_size - 1))
                chunk_len = end - start + 1

                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(chunk_len))
                self.end_headers()
                with open(file_path, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(chunk_len))
                return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(file_size))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        else:
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(file_path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            return self._serve_static_file(os.path.join(WEB_DIR, "index.html"))

        if path.startswith("/web/"):
            rel = path[len("/web/"):]
            safe_path = os.path.normpath(os.path.join(WEB_DIR, rel))
            if safe_path.startswith(os.path.normpath(WEB_DIR)):
                return self._serve_static_file(safe_path)
            return self._send_json({"error": "Forbidden"}, status=403)

        if path == "/api/system":
            _, _, free = shutil.disk_usage(BASE_DIR)
            voices = [
                {
                    "id": vid,
                    "gender": meta["gender"],
                    "accent": meta["accent"],
                    "age": meta["age"],
                    "traits": meta["traits"],
                }
                for vid, meta in VOICE_CATALOG.items()
            ]
            return self._send_json({
                "gpu": get_gpu_status_string(),
                "free_gb": round(free / (1024**3), 2),
                "providers": LLM_PROVIDERS,
                "provider_models": PROVIDER_MODELS,
                "base_urls": DEFAULT_BASE_URLS,
                "saved_keys": _get_saved_key_status(),
                "default_voice": TTS_DEFAULT_VOICE,
                "voices": voices,
                "bgm_Presets": [k for k in BGM_THEME_PROMPTS.keys() if k != "none"],
                "ambience_presets": [k for k in AMBIENCE_PROMPTS.keys() if k != "none"],
                "sfx_presets": [k for k in SFX_PROMPTS.keys() if k != "none"],
                "standalone_prompt": build_standalone_ai_prompt(),
            })

        if path.startswith("/api/audio/"):
            fname = os.path.basename(urllib.parse.unquote(path[len("/api/audio/"):]))
            target = os.path.join(OUTPUT_DIR, fname)
            qs = urllib.parse.parse_qs(parsed.query)
            dl = fname if qs.get("download") else None
            return self._serve_static_file(target, download_name=dl)

        if path.startswith("/api/jobs/") and path.endswith("/stream"):
            parts = path.strip("/").split("/")
            job_id = parts[2] if len(parts) >= 4 else ""
            return self._handle_sse_stream(job_id)

        self._send_json({"error": "Not found"}, status=404)

    def _handle_sse_stream(self, job_id: str):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job:
            return self._send_json({"error": "Job not found"}, status=404)

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        idx = 0
        try:
            while True:
                with job.cond:
                    while idx >= len(job.events) and not job.done:
                        job.cond.wait(timeout=12.0)
                        if idx >= len(job.events) and not job.done:
                            # Send SSE heartbeat comment so browser never times out during long GPU steps
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                    pending = job.events[idx:]
                    idx = len(job.events)
                    is_done = job.done

                for ev in pending:
                    frame = f"event: {ev['type']}\ndata: {json.dumps(ev['data'])}\n\n".encode("utf-8")
                    self.wfile.write(frame)
                    self.wfile.flush()

                if is_done and idx >= len(job.events):
                    break
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            if path == "/api/upload":
                raw_name = urllib.parse.unquote(self.headers.get("X-Filename", "book.txt"))
                ext = os.path.splitext(raw_name)[1].lower()
                if ext not in (".txt", ".pdf", ".json"):
                    return self._send_json({"error": "Only .txt, .pdf, or .json files are supported."}, status=400)
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    return self._send_json({"error": "Uploaded file is empty."}, status=400)

                safe_base = slugify_title(raw_name)
                dest_path = os.path.join(UPLOADS_DIR, f"{safe_base}_{int(time.time())}{ext}")
                with open(dest_path, "wb") as f:
                    f.write(self.rfile.read(length))

                text, title = CONVERTER.load_book_text(input_file=dest_path)
                return self._send_json({
                    "file_path": dest_path,
                    "filename": raw_name,
                    "title": title,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "preview": text[:420],
                })

            body = self._read_json_body()

            if path == "/api/models":
                models = fetch_provider_models(
                    provider=body.get("provider", "NVIDIA NIM API"),
                    api_key=body.get("api_key", ""),
                    base_url=body.get("base_url", ""),
                )
                return self._send_json({"models": models})

            if path == "/api/prompts":
                custom_dir = (body.get("custom_direction") or "").strip()
                has_file = bool(body.get("file_path"))
                has_text = bool((body.get("raw_text") or "").strip())

                # Story-free Standalone Prompt mode: user uploads their own story inside ChatGPT/Claude/Gemini!
                if not has_file and not has_text:
                    standalone = build_standalone_ai_prompt(custom_direction=custom_dir)
                    return self._send_json({
                        "title": "standalone_director_prompt",
                        "total_chars": len(standalone),
                        "parts": [{"part": 1, "total": 1, "chars": len(standalone), "prompt": standalone}],
                    })

                text, title = CONVERTER.load_book_text(
                    input_file=body.get("file_path") or None,
                    raw_text=body.get("raw_text") or None,
                )
                chunk_chars = int(body.get("chunk_chars", 4200))
                chunks = chunk_text(text, max_chars=chunk_chars)
                prompts = [
                    {
                        "part": i + 1,
                        "total": len(chunks),
                        "chars": len(ch),
                        "prompt": build_copyable_ai_prompt(
                            ch,
                            chunk_index=i + 1,
                            total_chunks=len(chunks),
                            custom_direction=custom_dir,
                        ),
                    }
                    for i, ch in enumerate(chunks)
                ]
                return self._send_json({"title": title, "total_chars": len(text), "parts": prompts})

            if path == "/api/validate-json":
                pasted = body.get("pasted_json", "")
                script = parse_pasted_script_json(pasted)
                overview = serialize_script_overview(
                    script,
                    narrator_voice=body.get("narrator_voice", TTS_DEFAULT_VOICE),
                )
                return self._send_json({
                    "valid": True,
                    "overview": overview,
                    "normalized_json": script.model_dump_json(indent=2),
                })

            if path == "/api/convert/auto":
                job_id = _start_conversion_job("auto", body)
                return self._send_json({"job_id": job_id})

            if path == "/api/convert/json":
                job_id = _start_conversion_job("json", body)
                return self._send_json({"job_id": job_id})

            if path == "/api/jobs/stop":
                CONVERTER.request_cancel()
                return self._send_json({"status": "stopping"})

            if path == "/api/vram/free":
                CONVERTER.free_gpu_memory()
                return self._send_json({"status": "cleared", "gpu": get_gpu_status_string()})

            self._send_json({"error": "Endpoint not found"}, status=404)
        except Exception as e:
            self._send_json({"error": str(e)}, status=400)


def main():
    url = f"http://{HOST}:{PORT}"
    server = ThreadingHTTPServer((HOST, PORT), StudioRequestHandler)
    print(f"\n  ● NEXGEN MONOCHROME AUDIOBOOK STUDIO")
    print(f"  ● Local Studio URL : {url}")
    print(f"  ● Hardware Status  : {get_gpu_status_string()}\n")

    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Studio server...")
        server.server_close()


if __name__ == "__main__":
    main()
