import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from config import DEFAULT_BASE_URLS, PROVIDER_MODELS


# Reasoning tag patterns emitted by DeepSeek, Nemotron, Qwen, Ollama, etc.
_CLOSED_THINK_RE = re.compile(
    r"(?:<think>.*?</think>|◁think▷.*?◁/think▷|\[THINK\].*?\[/THINK\]|<thought>.*?</thought>)",
    re.DOTALL | re.IGNORECASE,
)
_OPEN_THINK_TOKENS = ("<think>", "◁think▷", "[think]", "<thought>")


def strip_reasoning_traces(text: str) -> str:
    """
    Removes chain-of-thought / reasoning blocks (<think>...</think>, plain 'Here's a thinking process:' preambles, etc.)
    and isolates the final structured output.
    """
    if not text:
        return ""
    cleaned = _CLOSED_THINK_RE.sub("", text).strip()

    lower = cleaned.lower()
    for tok in _OPEN_THINK_TOKENS:
        idx = lower.find(tok)
        if idx != -1:
            after = cleaned[idx + len(tok):]
            # Look for actual JSON start after an unclosed <think> tag
            m = re.search(r'\{\s*"(?:characters|cast|blocks|scenes|dialogue|lines)"', after)
            if m:
                cleaned = (cleaned[:idx] + after[m.start():]).strip()
            else:
                cleaned = cleaned[:idx].strip()

    # Also strip untagged plain-text thinking preambles ("Here's a thinking process: ... { "characters": ...")
    m_root = re.search(r'\{\s*"(?:characters|cast|blocks|scenes|dialogue|lines|speaker)"', cleaned)
    if m_root and m_root.start() > 0:
        prefix = cleaned[:m_root.start()]
        if "thinking process" in prefix.lower() or "analyze user input" in prefix.lower() or "```" in prefix:
            cleaned = cleaned[m_root.start():]

    return cleaned.strip()


def _extract_balanced_json_snippets(text: str) -> List[str]:
    """
    Scans text with a quote-aware state machine and extracts top-level {...} or [...] JSON blocks.
    Supports users pasting MULTIPLE chunk JSON responses sequentially into one textbox!
    """
    snippets: List[str] = []
    n = len(text)
    i = 0

    while i < n:
        ch = text[i]
        if ch in "{[":
            stack = [ch]
            start = i
            in_str = False
            esc = False
            i += 1
            while i < n and stack:
                c = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c in "{[":
                        stack.append(c)
                    elif c == "}":
                        if stack and stack[-1] == "{":
                            stack.pop()
                        else:
                            break
                    elif c == "]":
                        if stack and stack[-1] == "[":
                            stack.pop()
                        else:
                            break
                i += 1
            if not stack:
                snippets.append(text[start:i])
                continue
        i += 1
    return snippets


def _escape_unescaped_inner_quotes(s: str) -> str:
    """
    Repairs unescaped double-quotes inside JSON string values—e.g.:
      "text": "Seraphina screamed. "IT'S NOT HER!""
    by checking whether a '"' inside a string is genuinely followed by a valid JSON structural token.
    """
    out: List[str] = []
    n = len(s)
    in_str = False
    is_key = False
    esc = False
    i = 0

    while i < n:
        ch = s[i]
        if not in_str:
            if ch == '"':
                in_str = True
                # Determine if this string is likely an object key by checking previous non-ws char
                j = len(out) - 1
                while j >= 0 and out[j] in " \t\r\n":
                    j -= 1
                prev_ch = out[j] if j >= 0 else "{"
                is_key = prev_ch in ("{", ",")
            out.append(ch)
        else:
            if esc:
                esc = False
                out.append(ch)
            elif ch == "\\":
                esc = True
                out.append(ch)
            elif ch == '"':
                # Peek ahead past whitespace to check if this quote legally closes the JSON string
                k = i + 1
                while k < n and s[k] in " \t\r\n":
                    k += 1
                next_ch = s[k] if k < n else ""

                valid_close = False
                if is_key:
                    valid_close = next_ch == ":"
                else:
                    if next_ch in ("}", "]", ""):
                        valid_close = True
                    elif next_ch == ",":
                        # After a comma in an object/array, the next non-ws token must start a valid JSON key/value
                        m = k + 1
                        while m < n and s[m] in " \t\r\n":
                            m += 1
                        after_comma = s[m] if m < n else ""
                        if after_comma in ('"', "{", "[", "}", "]", "-", "t", "f", "n") or after_comma.isdigit():
                            valid_close = True

                if valid_close:
                    in_str = False
                    out.append(ch)
                else:
                    out.append('\\"')
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                pass
            else:
                out.append(ch)
        i += 1

    return "".join(out)


def _repair_json_string(raw: str) -> str:
    """Fixes common LLM / copy-paste JSON quirks: fences, smart quotes, comments, trailing commas, inner quotes."""
    s = raw.strip()
    s = re.sub(r"^```(?:json|javascript|js)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")

    s = re.sub(r"(?m)^\s*//.*$", "", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return _escape_unescaped_inner_quotes(s.strip())


def _salvage_individual_objects(text: str) -> List[Dict[str, Any]]:
    """
    Last-resort recovery if an outer JSON object was truncated mid-stream or malformed:
    scans every balanced {...} dict inside the text and recovers all valid character & dialogue dicts!
    """
    recovered_chars: List[Dict[str, Any]] = []
    recovered_blocks: List[Dict[str, Any]] = []
    n = len(text)

    for m in re.finditer(r"\{", text):
        start = m.start()
        # Only inspect leaf-level or modest-sized dicts (< 4000 chars) to avoid O(N^2) on huge roots
        end_limit = min(n, start + 4000)
        sub = text[start:end_limit]
        snippets = _extract_balanced_json_snippets(sub)
        if not snippets:
            continue
        cand = snippets[0]
        for attempt_str in (cand, _repair_json_string(cand)):
            try:
                obj = json.loads(attempt_str)
                if isinstance(obj, dict):
                    has_text = bool(obj.get("text") or obj.get("line") or obj.get("dialogue") or obj.get("content"))
                    if has_text and (obj.get("speaker") or obj.get("character") or obj.get("name")):
                        recovered_blocks.append(obj)
                    elif (obj.get("name") or obj.get("character")) and obj.get("voice_profile") and not has_text:
                        recovered_chars.append(obj)
                break
            except Exception:
                continue

    if recovered_blocks:
        return [{"characters": recovered_chars, "blocks": recovered_blocks}]
    return []


def _looks_like_script_payload(obj: Any) -> bool:
    if isinstance(obj, list):
        return len(obj) > 0 and isinstance(obj[0], dict)
    if isinstance(obj, dict):
        script_keys = {"blocks", "scenes", "dialogue", "lines", "script", "segments", "characters", "cast", "speaker", "text"}
        return bool(script_keys.intersection(obj.keys()))
    return False


def extract_and_parse_json_objects(raw_text: str) -> List[Any]:
    """
    Extracts and parses one or more JSON payloads from ANY LLM output or user-pasted text:
    - Strips <think>...</think> and plain-text thinking preambles
    - Handles single or multiple markdown ```json blocks
    - Skips invalid pseudo-braces inside reasoning notes and extracts the real script JSON
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response received from AI model (no JSON content found).")

    cleaned = strip_reasoning_traces(raw_text)
    if not cleaned:
        raise ValueError(
            "Model spent its entire output budget on internal reasoning without writing JSON. "
            "Try increasing max tokens or switching to a faster non-thinking preset."
        )

    for candidate_whole in (cleaned, _repair_json_string(cleaned)):
        try:
            parsed = json.loads(candidate_whole)
            if _looks_like_script_payload(parsed):
                return [parsed] if not (isinstance(parsed, dict) and "speaker" in parsed and "blocks" not in parsed) else [{"blocks": [parsed]}]
        except Exception:
            pass

    snippets = _extract_balanced_json_snippets(cleaned)
    parsed_objects: List[Any] = []
    fallback_objects: List[Any] = []
    last_err: Optional[Exception] = None

    for snip in snippets:
        for candidate in (snip, _repair_json_string(snip)):
            try:
                obj = json.loads(candidate)
                if _looks_like_script_payload(obj):
                    if isinstance(obj, dict) and "speaker" in obj and "text" in obj and "blocks" not in obj:
                        parsed_objects.append({"blocks": [obj]})
                    else:
                        parsed_objects.append(obj)
                elif isinstance(obj, (dict, list)):
                    fallback_objects.append(obj)
                break
            except Exception as e:
                last_err = e

    if parsed_objects:
        return parsed_objects
    if fallback_objects:
        return fallback_objects

    # Last-resort salvage for truncated or outer-broken JSON streams
    salvaged = _salvage_individual_objects(cleaned)
    if salvaged:
        return salvaged

    preview = cleaned[:240].replace("\n", " ")
    raise ValueError(f"Model did not return valid script JSON ({last_err}). Output preview: {preview}...")


def is_reasoning_model(model_name: str) -> bool:
    """
    Detects models with internal chain-of-thought / reasoning capabilities.
    NOTE: ALL NVIDIA Nemotron 3 / 3.5 variants (Bolt, Lightning, Super, Ultra, Nano) use reasoning templates!
    """
    m = (model_name or "").lower()
    reasoning_markers = (
        "nemotron",
        "bolt",
        "lightning",
        "ultra",
        "super",
        "reasoning",
        "deepseek-r1",
        "qwq",
        "qwen3",
        "o1",
        "o3",
        "o4",
        "gpt-5",
        "gemini-2.5-pro",
        "gemini-3",
        "thinking",
    )
    return any(tok in m for tok in reasoning_markers)


def is_openai_strict_reasoning_model(model_name: str) -> bool:
    """OpenAI o-series and gpt-5 models reject custom temperature != 1 and require max_completion_tokens."""
    m = (model_name or "").lower().split("/")[-1]
    return m.startswith(("o1", "o3", "o4", "gpt-5"))


class UnifiedLLMClient:
    """
    Multi-Provider LLM Client supporting:
    - NVIDIA NIM API (Nemotron 3.5 Bolt, Lightning, Ultra, Super, Llama-Nemotron, etc.)
    - Google Gemini API (Native REST v1beta with Gemini 2.5 / 3 thought-part filtering)
    - OpenAI API & OpenAI-Compatible endpoints (GPT-5, GPT-4o, o3, o4-mini, OpenRouter, etc.)
    - Ollama Local (100% Optional)
    """

    def __init__(
        self,
        provider: str = "NVIDIA NIM API",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 210,
    ):
        self.provider = provider
        default_models = PROVIDER_MODELS.get(provider, ["llama3"])
        self.model_name = (model_name or default_models[0]).strip()
        self.api_key = self._resolve_api_key(provider, api_key)
        self.base_url = (base_url or DEFAULT_BASE_URLS.get(provider, "")).strip().rstrip("/")
        self.timeout = timeout
        self._ssl_ctx = ssl.create_default_context()

    @staticmethod
    def _resolve_api_key(provider: str, explicit_key: Optional[str]) -> str:
        if explicit_key and explicit_key.strip():
            return explicit_key.strip()
        env_map = {
            "NVIDIA NIM API": ("NVIDIA_API_KEY", "NIM_API_KEY", "NGC_API_KEY"),
            "Google Gemini API": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            "OpenAI API": ("OPENAI_API_KEY",),
            "Ollama (Local - Optional)": (),
        }
        for env_var in env_map.get(provider, ()):
            val = os.getenv(env_var, "").strip()
            if val:
                return val
        return ""

    def _http_json(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: int = 2,
    ) -> Tuple[int, Dict[str, Any]]:
        req_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Noir-Audiobook-Studio/2.5",
        }
        if headers:
            req_headers.update(headers)

        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        last_exc: Optional[Exception] = None

        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl_ctx) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                    return resp.status, json.loads(raw) if raw.strip() else {}
            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(2.0 * (attempt + 1))
                    last_exc = RuntimeError(f"HTTP {e.code}: {err_body[:300]}")
                    continue
                try:
                    parsed_err = json.loads(err_body)
                except Exception:
                    parsed_err = {"raw_error": err_body}
                return e.code, parsed_err
            except (urllib.error.URLError, TimeoutError) as e:
                last_exc = e
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise ConnectionError(f"Network error connecting to {self.provider} ({url}): {e}") from e

        raise RuntimeError(f"API request failed after retries: {last_exc}")

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[Dict[str, Any]] = None,
        unload_after: bool = False,
    ) -> List[Any]:
        if self.provider == "Google Gemini API":
            raw_text = self._call_gemini(system_prompt, user_prompt)
        elif self.provider == "Ollama (Local - Optional)":
            raw_text = self._call_ollama(system_prompt, user_prompt, json_schema, unload_after=unload_after)
        elif self.provider in ("NVIDIA NIM API", "OpenAI API"):
            raw_text = self._call_openai_compatible(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        return extract_and_parse_json_objects(raw_text)

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise ValueError("Google Gemini API Key is missing. Enter your Gemini API key in the UI.")

        model_slug = self.model_name
        if model_slug.startswith("models/"):
            model_slug = model_slug[len("models/"):]

        url = f"{self.base_url}/models/{urllib.parse.quote(model_slug)}:generateContent?key={urllib.parse.quote(self.api_key)}"
        payload: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 16384 if is_reasoning_model(model_slug) else 8192,
                "responseMimeType": "application/json",
            },
        }

        status, data = self._http_json("POST", url, payload)
        if status >= 400:
            fallback_payload = {
                "contents": [{"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 16384},
            }
            status2, data2 = self._http_json("POST", url, fallback_payload)
            if status2 >= 400:
                err_msg = data2.get("error", {}).get("message") or data.get("error", {}).get("message") or str(data2)
                raise RuntimeError(f"Gemini API error ({status2}): {err_msg}")
            data = data2

        candidates = data.get("candidates") or []
        if not candidates:
            block_reason = data.get("promptFeedback", {}).get("blockReason", "No candidates returned")
            raise RuntimeError(f"Gemini returned no output ({block_reason}).")

        parts = candidates[0].get("content", {}).get("parts") or []
        answer_chunks = [p.get("text", "") for p in parts if not p.get("thought") and p.get("text")]
        if not answer_chunks:
            answer_chunks = [p.get("text", "") for p in parts if p.get("text")]
        return "\n".join(answer_chunks)

    def _call_openai_compatible(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise ValueError(f"{self.provider} Key is missing. Please enter your API key in the UI.")

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        is_nim = self.provider == "NVIDIA NIM API" or "nvidia.com" in self.base_url
        strict_o_model = (not is_nim) and is_openai_strict_reasoning_model(self.model_name)

        attempts: List[Dict[str, Any]] = []

        if is_nim:
            # Nemotron 3 / 3.5 & Llama-Nemotron reasoning handling:
            # 1) Pass BOTH "detailed thinking off" (for Llama-Nemotron) AND chat_template_kwargs.enable_thinking=False
            #    (for Nemotron 3 / 3.5 Bolt, Lightning, Super, Ultra) so the model writes pure JSON immediately!
            nim_fast_sys = f"detailed thinking off\n{system_prompt}"
            attempts.append({
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": nim_fast_sys},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 8192,
                "chat_template_kwargs": {"enable_thinking": False},
            })

            # 2) If endpoint forces thinking ON or rejects chat_template_kwargs, give a large 16,384 token budget
            #    so internal reasoning_content completes fully AND leaves plenty of room for final msg.content JSON!
            attempts.append({
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": nim_fast_sys},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 16384,
            })

            # 3) Fallback for strict smaller-window NIM endpoints (e.g. 4096 max_tokens limit)
            attempts.append({
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": nim_fast_sys},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 4096,
            })
        elif strict_o_model:
            attempts.append({
                "model": self.model_name,
                "messages": [
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_completion_tokens": 16384,
                "response_format": {"type": "json_object"},
            })
            attempts.append({
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
                ],
                "max_completion_tokens": 16384,
            })
        else:
            attempts.append({
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 8192,
                "response_format": {"type": "json_object"},
            })
            attempts.append({
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 8192,
            })

        last_err_msg = ""
        for payload in attempts:
            status, data = self._http_json("POST", url, payload, headers=headers)
            if status in (401, 403):
                err = data.get("error", {}).get("message") or data.get("detail") or str(data)
                raise PermissionError(f"{self.provider} authentication failed ({status}): {err}")
            if status >= 400:
                last_err_msg = data.get("error", {}).get("message") or data.get("detail") or str(data)
                continue

            choices = data.get("choices") or []
            if not choices:
                last_err_msg = "API returned empty choices list."
                continue

            msg = choices[0].get("message") or {}
            finish_reason = choices[0].get("finish_reason")
            content = msg.get("content")
            reasoning_content = msg.get("reasoning_content")

            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") in ("text", "output_text", None)
                )

            # Validate that content actually contains parsable JSON before accepting this attempt!
            if content and str(content).strip():
                try:
                    extract_and_parse_json_objects(str(content))
                    return str(content)
                except Exception as parse_e:
                    last_err_msg = f"Invalid JSON in content (finish_reason={finish_reason}): {parse_e}"
                    continue

            # Only fall back to reasoning_content if a model accidentally placed full valid JSON with "blocks" inside it
            if reasoning_content and '"blocks"' in str(reasoning_content):
                try:
                    extract_and_parse_json_objects(str(reasoning_content))
                    return str(reasoning_content)
                except Exception:
                    pass

            last_err_msg = (
                f"Model '{self.model_name}' exhausted tokens in thinking mode before writing JSON "
                f"(finish_reason={finish_reason}). Retrying with larger token budget..."
            )

        raise RuntimeError(f"{self.provider} failed for '{self.model_name}': {last_err_msg}")

    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[Dict[str, Any]] = None,
        unload_after: bool = False,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        reasoning = is_reasoning_model(self.model_name)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 8192 if reasoning else 4096,
            },
        }
        if not reasoning:
            payload["format"] = json_schema if json_schema else "json"
        if unload_after:
            payload["keep_alive"] = 0

        try:
            status, data = self._http_json("POST", url, payload, retries=1)
        except ConnectionError as e:
            raise ConnectionError(
                f"Could not connect to local Ollama at {self.base_url}. "
                "Start Ollama ('ollama serve'), switch Provider to NVIDIA NIM / Gemini / OpenAI, "
                "or use the 'Prompt & JSON' tab!"
            ) from e

        if status >= 400:
            err = data.get("error") or str(data)
            raise RuntimeError(f"Ollama error ({status}): {err}")

        return (data.get("message") or {}).get("content", "")

    def unload_local_model(self):
        if self.provider != "Ollama (Local - Optional)":
            return
        try:
            url = f"{self.base_url}/api/generate"
            self._http_json("POST", url, {"model": self.model_name, "keep_alive": 0}, retries=0)
        except Exception:
            pass


def fetch_provider_models(provider: str, api_key: str = "", base_url: str = "") -> List[str]:
    presets = list(PROVIDER_MODELS.get(provider, []))
    client = UnifiedLLMClient(provider=provider, api_key=api_key, base_url=base_url, timeout=12)

    try:
        if provider == "Ollama (Local - Optional)":
            status, data = client._http_json("GET", f"{client.base_url}/api/tags", retries=0)
            if status == 200 and "models" in data:
                live = [m["name"] for m in data["models"] if m.get("name")]
                return live if live else presets

        elif provider == "Google Gemini API" and client.api_key:
            url = f"{client.base_url}/models?key={urllib.parse.quote(client.api_key)}"
            status, data = client._http_json("GET", url, retries=0)
            if status == 200 and "models" in data:
                live = []
                for m in data["models"]:
                    methods = m.get("supportedGenerationMethods") or []
                    name = (m.get("name") or "").replace("models/", "")
                    if "generateContent" in methods and name.startswith("gemini"):
                        live.append(name)
                if live:
                    return presets + [x for x in sorted(live, reverse=True) if x not in presets]

        elif provider in ("NVIDIA NIM API", "OpenAI API") and client.api_key:
            headers = {"Authorization": f"Bearer {client.api_key}"}
            status, data = client._http_json("GET", f"{client.base_url}/models", headers=headers, retries=0)
            if status == 200 and "data" in data:
                live = [m["id"] for m in data["data"] if isinstance(m, dict) and m.get("id")]
                if live:
                    if provider == "NVIDIA NIM API":
                        nemotron_first = [x for x in live if "nemotron" in x.lower()]
                        others = [x for x in live if "nemotron" not in x.lower()]
                        return presets + [x for x in (nemotron_first + sorted(others)) if x not in presets]
                    return presets + [x for x in sorted(live) if x not in presets]
    except Exception:
        pass

    return presets
