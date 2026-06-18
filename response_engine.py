# response_engine.py
"""
Multi-model response engine supporting Anthropic, Mistral, Gemini, DeepSeek, and Ollama APIs
"""
from __future__ import annotations
import time
import os
import re
import json
import base64
from pathlib import Path
from typing import Any, Dict
import requests


def _load_dotenv():
    """Load .env from the project root into os.environ (stdlib only, won't override existing exports)."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())

_load_dotenv()


# ── Backend selector ──────────────────────────────────────────────────────────
# Set exactly one of these to True; the rest False.
USE_ANTHROPIC = False
USE_MISTRAL   = False
USE_GEMINI    = False
USE_DEEPSEEK  = False
USE_OPENAI    = False   # Falls through to Ollama if all above are False

# ── Anthropic config ──────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = "claude-haiku-4-5-20251001"  # or claude-opus-4-20250514 or claude-sonnet-4-20250514 or claude-opus-4-6

# ── Mistral config ────────────────────────────────────────────────────────────
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_MODEL   = "mistral-medium-latest"       # or mistral-small-latest, etc.
MISTRAL_URL     = "https://api.mistral.ai/v1/chat/completions"

# ── Gemini config ─────────────────────────────────────────────────────────────
# Uses Vertex AI (cloud-platform scope, bills to your GCP project credits).
# Fallback: GEMINI_API_KEY env var → AI Studio endpoint (different billing).
GEMINI_ADC_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL    = "gemini-2.5-pro"
GEMINI_REGION          = os.environ.get("GEMINI_REGION", "us-central1")
# Gemini thinking models burn internal tokens before writing — needs a much higher
# output cap than NUM_PREDICT (which is tuned for local/Ollama models).
GEMINI_MAX_OUTPUT_TOKENS = 8192

# Cached OAuth token so we don't hit the refresh endpoint every call
_gemini_access_token: str = ""
_gemini_token_expiry: float = 0.0


def _get_gemini_access_token() -> str:
    """Refresh and cache an OAuth access token from ADC credentials."""
    global _gemini_access_token, _gemini_token_expiry
    now = time.time()
    # Return cached token if it still has >60s left
    if _gemini_access_token and now < _gemini_token_expiry - 60:
        return _gemini_access_token
    try:
        with open(GEMINI_ADC_PATH) as f:
            creds = json.load(f)
        resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id":     creds["client_id"],
                "client_secret": creds["client_secret"],
                "refresh_token": creds["refresh_token"],
                "grant_type":    "refresh_token",
            },
            timeout=(5, 15),
        )
        resp.raise_for_status()
        token_data = resp.json()
        _gemini_access_token = token_data["access_token"]
        _gemini_token_expiry = now + token_data.get("expires_in", 3600)
        return _gemini_access_token
    except Exception as e:
        print(f"[LLM] Gemini ADC token refresh failed: {e}")
        return ""

# ── OpenAI config ─────────────────────────────────────────────────────────────
# Key is OPENAI_KEY in .env — verify model IDs at platform.openai.com/docs/models
OPENAI_API_KEY = os.environ.get("OPENAI_KEY", "")
OPENAI_MODEL   = "gpt-5.5"
OPENAI_URL     = "https://api.openai.com/v1/chat/completions"

# ── DeepSeek config ───────────────────────────────────────────────────────────
# Needs DEEPSEEK_API_KEY in .env (separate from GEMINI_API_KEY)
# Verify model IDs at platform.deepseek.com/api-docs
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL   = "deepseek-reasoner"   # DeepSeek R1 reasoning model
DEEPSEEK_URL     = "https://api.deepseek.com/v1/chat/completions"

# ── Ollama config (local fallback) ────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma2:2b"

# ── Shared knobs ──────────────────────────────────────────────────────────────
CONNECT_TIMEOUT    = 30
API_READ_TIMEOUT   = 30    # cloud APIs answer in seconds — fail loud, don't sit for minutes
LOCAL_READ_TIMEOUT = 180   # local Ollama on the 4060 can be slow; give it room (probing what it can do)
NUM_CTX         = int(os.environ.get("SIGNALBOT_NUM_CTX", "16384"))
# Per-model ctx overrides — some large models need a cap to fit in VRAM
MODEL_CTX_OVERRIDES = {
    # 20k ctx — measured 2026-06-15: ctx size barely affects the GPU split (8k→32k
    # all spill ~23-25% to CPU) because the 7.1GB Q4_K_M weights + KV + compute buffer
    # don't fit in the 8GB 4060 at ANY ctx. So don't cramp it: 20k gives headroom for
    # code-reads at ~no cost. The real speed lever is a smaller quant (fits fully → 100%
    # GPU), NOT lowering ctx. Was 32768 (more spill, no benefit).
    "gemma4:12b": 20480,
}
NUM_PREDICT     = 1200
TEMPERATURE     = 0.7


# Populated after each call — read with get_last_call_meta()
# Lets the caller log tokens + model name without changing generate_response's return type
_last_meta: dict = {"model": None, "tokens_in": 0, "tokens_out": 0}


def get_last_call_meta() -> dict:
    """Return a copy of metadata from the most recent generate_response call."""
    return dict(_last_meta)


def generate_response(prompt: str, images=None) -> str:
    """Generate response using the configured backend.

    images: optional list of base64 strings (no data: prefix). Wired for vision on
    Anthropic, OpenAI, Mistral, Gemini, and vision-capable Ollama models. NOTE: the
    local gemma4:12b GGUF (unsloth Q4) ships WITHOUT the vision projector, so it's
    text-only for now — `ollama show` lists no `vision` capability. Text-only Ollama
    models 400 on images (handled gracefully); DeepSeek is text-only too.
    """
    try:
        if USE_ANTHROPIC:
            return _generate_anthropic(prompt, images)
        elif USE_MISTRAL:
            return _generate_mistral(prompt, images)
        elif USE_GEMINI:
            return _generate_gemini(prompt, images)
        elif USE_DEEPSEEK:
            return _generate_deepseek(prompt)
        elif USE_OPENAI:
            return _generate_openai(prompt, images)
        else:
            return _generate_ollama(prompt, images)
    except Exception as e:
        print(f"[LLM] UNCAUGHT ERROR in generate_response: {type(e).__name__}: {e}")
        return f"[GROUND] Response generation failed unexpectedly: {type(e).__name__}: {e}"


def generate_response_stream(prompt: str, images=None):
    """Streaming sibling of generate_response — yields visible text CHUNKS.

    Why this exists: gemma4:12b prefill on the 8GB 4060 can take ~80s before the
    first token, and the web app sits behind Cloudflare's fixed ~100s edge cap.
    Non-streaming means Cloudflare sees silence for the whole generation → 524.
    Streaming gets the first byte out fast and keeps the connection alive.

    Cloud backends answer in seconds (no 524 risk) so they just yield ONE chunk —
    we reuse the blocking generate_response for them. Only Ollama (the slow local
    path) truly streams. Either way _last_meta is set, same as generate_response.
    """
    if USE_ANTHROPIC or USE_MISTRAL or USE_GEMINI or USE_DEEPSEEK or USE_OPENAI:
        yield generate_response(prompt, images)   # fast cloud path: one shot
        return
    yield from _generate_ollama_stream(prompt, images)


def _sniff_image_mime(b64_str: str) -> str:
    """Recover an image's MIME type from its magic bytes (first bytes of the file).

    The frontend strips the `data:image/...;base64,` label for Ollama, so APIs
    that REQUIRE a media_type (Claude, Gemini) would otherwise be flying blind.
    We decode the first few bytes and match the format's signature. Defaults to
    jpeg if unknown. Note: HEIC (some iPhones) isn't matched here and Claude
    doesn't accept it anyway — resize/convert client-side if that bites.
    """
    try:
        head = base64.b64decode(b64_str[:24])  # 24 b64 chars -> 18 bytes, enough for any signature
    except Exception:
        return "image/jpeg"
    if head[:3] == b"\xff\xd8\xff":                       return "image/jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":                  return "image/png"
    if head[:4] == b"GIF8":                               return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":     return "image/webp"
    return "image/jpeg"


def _generate_anthropic(prompt: str, images=None) -> str:
    """Call Anthropic API. With images, builds a vision content array."""
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=ANTHROPIC_API_KEY, timeout=API_READ_TIMEOUT)
        t0 = time.perf_counter()

        # Vision: content becomes [text, image, image…]; text-only stays a string.
        if images:
            content: Any = [{"type": "text", "text": prompt}]
            for b64 in images:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _sniff_image_mime(b64),
                        "data": b64,
                    },
                })
        else:
            content = prompt

        # Claude 4 series (opus-4, sonnet-4) deprecated the temperature param
        NO_TEMP_MODELS = ("claude-opus-4", "claude-sonnet-4", "claude-fable")
        params = dict(
            model=ANTHROPIC_MODEL,
            max_tokens=NUM_PREDICT,
            messages=[{"role": "user", "content": content}],
        )
        if not any(ANTHROPIC_MODEL.startswith(m) for m in NO_TEMP_MODELS):
            params["temperature"] = TEMPERATURE

        message = client.messages.create(**params)

        dt_ms = (time.perf_counter() - t0) * 1000
        text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")  # skip ThinkingBlocks (Fable)
        print(f"[LLM] Anthropic ok in {dt_ms:.1f} ms | model={ANTHROPIC_MODEL}")
        _last_meta["model"] = ANTHROPIC_MODEL
        _last_meta["tokens_in"] = message.usage.input_tokens if message.usage else 0
        _last_meta["tokens_out"] = message.usage.output_tokens if message.usage else 0

        return text or "[GROUND] Anthropic returned empty response."

    except ImportError:
        return "[GROUND] Anthropic SDK not installed. Run: pip install anthropic --break-system-packages"
    except Exception as e:
        return f"[GROUND] Anthropic API error: {e}"


def _generate_mistral(prompt: str, images=None) -> str:
    """Call Mistral API. With images, builds a vision content array.

    Vision needs a multimodal model (mistral-medium-latest / pixtral-*); the API
    ignores images on a text-only model. Same image_url + data-URL shape as OpenAI.
    """
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    if images:
        content: Any = [{"type": "text", "text": prompt}]
        for b64 in images:
            mime = _sniff_image_mime(b64)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
    else:
        content = prompt
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": NUM_PREDICT,
        "temperature": TEMPERATURE,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            MISTRAL_URL,
            headers=headers,
            json=payload,
            timeout=(CONNECT_TIMEOUT, API_READ_TIMEOUT),
        )
        resp.raise_for_status()
        data = resp.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"[LLM] Mistral ok in {dt_ms:.1f} ms | model={MISTRAL_MODEL}")
        usage = data.get("usage", {})
        _last_meta["model"] = MISTRAL_MODEL
        _last_meta["tokens_in"] = usage.get("prompt_tokens", 0)
        _last_meta["tokens_out"] = usage.get("completion_tokens", 0)

        return text or "[GROUND] Mistral returned empty response."

    except requests.exceptions.ConnectionError:
        return "[GROUND] Can't connect to Mistral API."
    except requests.exceptions.Timeout:
        dt_ms = (time.perf_counter() - t0) * 1000
        return f"[GROUND] Mistral call timed out after {dt_ms:.0f} ms."
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            pass
        return f"[GROUND] Mistral HTTP error: {e}\n{body}"
    except Exception as e:
        return f"[GROUND] Mistral error: {e}"


# Gemma 4 (unsloth GGUF) wraps its chain-of-thought in harmony-style channel
# markers: "<|channel>thought\n...reasoning...<channel|>actual answer". Strip the
# reasoning block so only the user-facing answer reaches the chat/two-lane output.
_CHANNEL_BLOCK = re.compile(r'<\|channel>.*?<channel\|>', re.DOTALL)

def _strip_reasoning(text: str) -> str:
    if '<|channel>' not in text and '<channel|>' not in text:
        return text.strip()  # no markers — normal model, untouched
    cleaned = _CHANNEL_BLOCK.sub('', text)            # drop paired reasoning blocks
    cleaned = cleaned.replace('<|channel>', '').replace('<channel|>', '')  # orphan markers (truncated gen)
    return cleaned.strip()


def _ollama_call(prompt: str, num_ctx: int, images=None):
    """Single Ollama HTTP call. Returns (text, data) or raises.

    images: optional list of base64 strings (no data: prefix). Only vision models
    accept them; text-only Ollama models 400 ("does not support multimodal") —
    _generate_ollama catches that and retries text-only with a heads-up.
    """
    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
        },
    }
    if images:
        payload["images"] = images   # /api/generate vision field
    resp = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=(CONNECT_TIMEOUT, LOCAL_READ_TIMEOUT),
    )
    resp.raise_for_status()
    data = resp.json()
    return _strip_reasoning(data.get("response") or ""), data


def _generate_ollama(prompt: str, images=None) -> str:
    """Call Ollama. Falls back to 8k ctx if the model rejects the larger window."""
    ctx = MODEL_CTX_OVERRIDES.get(OLLAMA_MODEL, NUM_CTX)
    t0 = time.perf_counter()
    try:
        text, data = _ollama_call(prompt, ctx, images)
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"[LLM] Ollama ok in {dt_ms:.1f} ms | model={OLLAMA_MODEL} | ctx={ctx}")
        _last_meta["model"] = OLLAMA_MODEL
        _last_meta["tokens_in"] = data.get("prompt_eval_count", 0)
        _last_meta["tokens_out"] = data.get("eval_count", 0)
        return text or "[GROUND] Ollama returned an empty response."

    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:600]
        except Exception:
            pass
        # Text-only model got an image — Ollama 400s ("does not support multimodal").
        # Don't dump a raw error: retry text-only and tell the user it's blind so
        # they can switch to a vision model or a cloud backend.
        if images and "multimodal" in body.lower():
            print(f"[LLM] {OLLAMA_MODEL} is text-only; retrying without the image")
            try:
                text, data = _ollama_call(prompt, ctx, None)
                _last_meta["model"] = OLLAMA_MODEL
                _last_meta["tokens_in"] = data.get("prompt_eval_count", 0)
                _last_meta["tokens_out"] = data.get("eval_count", 0)
                note = (f"[GROUND] Heads-up: {OLLAMA_MODEL} is text-only and can't see "
                        f"images — answering your text alone. Pick a vision model or a "
                        f"cloud backend (Claude/Gemini/Mistral) to use the photo.\n\n")
                return note + (text or "")
            except Exception as e2:
                return f"[GROUND] {OLLAMA_MODEL} can't process images, and the text-only retry failed: {e2}"

        # Retry at 8k if the error looks like an OOM / ctx-too-large failure
        oom_signals = ("out of memory", "context length", "kv cache", "failed to allocate")
        if any(s in body.lower() for s in oom_signals) and ctx > 8192:
            print(f"[LLM] ctx={ctx} OOM for {OLLAMA_MODEL}, retrying at 8192")
            try:
                text, data = _ollama_call(prompt, 8192, images)
                dt_ms = (time.perf_counter() - t0) * 1000
                print(f"[LLM] Ollama retry ok in {dt_ms:.1f} ms | ctx=8192")
                _last_meta["model"] = OLLAMA_MODEL
                _last_meta["tokens_in"] = data.get("prompt_eval_count", 0)
                _last_meta["tokens_out"] = data.get("eval_count", 0)
                return text or "[GROUND] Ollama returned an empty response."
            except Exception as e2:
                return f"[GROUND] Ollama failed at 8k fallback: {e2}"
        return f"[GROUND] Ollama HTTP error: {e}\n{body}"

    except requests.exceptions.ConnectionError:
        return "[GROUND] Can't connect to Ollama at localhost:11434."
    except requests.exceptions.Timeout:
        dt_ms = (time.perf_counter() - t0) * 1000
        return f"[GROUND] Ollama call timed out after {dt_ms:.0f} ms."
    except ValueError as e:
        return f"[GROUND] Ollama returned non-JSON output: {e}"
    except Exception as e:
        return f"[GROUND] LLM error: {e}"


# Streaming counterpart of _strip_reasoning. gemma4 wraps its chain-of-thought in
# "<|channel>…reasoning…<channel|>answer". When streaming token-by-token we can't
# regex the whole blob, so this state machine suppresses the thought block live and
# only emits the user-facing answer. Markers can straddle chunk boundaries, so we
# hold back the last few chars (a marker is 10 wide) until they're proven safe.
_CH_OPEN  = "<|channel>"
_CH_CLOSE = "<channel|>"

class _StreamReasoningFilter:
    def __init__(self):
        self.buf = ""
        self.in_thought = False

    def feed(self, text: str) -> str:
        """Add a token, return whatever visible text is now safe to emit."""
        self.buf += text
        out = []
        while True:
            if not self.in_thought:
                i = self.buf.find(_CH_OPEN)
                if i == -1:
                    # no opener in buffer — emit all but a possible partial opener tail
                    hold = len(_CH_OPEN) - 1
                    if len(self.buf) > hold:
                        out.append(self.buf[:-hold])
                        self.buf = self.buf[-hold:]
                    break
                out.append(self.buf[:i])                 # text before the thought block
                self.buf = self.buf[i + len(_CH_OPEN):]
                self.in_thought = True
            else:
                j = self.buf.find(_CH_CLOSE)
                if j == -1:
                    # still inside the thought — discard, keep a possible partial closer
                    hold = len(_CH_CLOSE) - 1
                    if len(self.buf) > hold:
                        self.buf = self.buf[-hold:]
                    break
                self.buf = self.buf[j + len(_CH_CLOSE):]  # drop the whole thought block
                self.in_thought = False
        return "".join(out)

    def flush(self) -> str:
        """End of stream — emit any held-back tail (unless we died mid-thought)."""
        if self.in_thought:
            self.buf = ""
            return ""
        out, self.buf = self.buf, ""
        return out


def _generate_ollama_stream(prompt: str, images=None, ctx=None):
    """Stream tokens from Ollama (stream:True), stripping reasoning blocks live.

    Yields visible text chunks; sets _last_meta from the final done-frame. On an
    OOM/ctx error before any tokens flow, retries once at 8k (mirrors _generate_ollama).
    """
    if ctx is None:
        ctx = MODEL_CTX_OVERRIDES.get(OLLAMA_MODEL, NUM_CTX)
    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"num_ctx": ctx, "num_predict": NUM_PREDICT, "temperature": TEMPERATURE},
    }
    if images:
        payload["images"] = images
    t0 = time.perf_counter()
    rfilter = _StreamReasoningFilter()
    final: Dict[str, Any] = {}
    emitted = False
    try:
        resp = requests.post(
            OLLAMA_URL, json=payload, stream=True,
            timeout=(CONNECT_TIMEOUT, LOCAL_READ_TIMEOUT),
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue  # skip a malformed frame rather than killing the stream
            tok = obj.get("response", "")
            if tok:
                vis = rfilter.feed(tok)
                if vis:
                    emitted = True
                    yield vis
            if obj.get("done"):
                final = obj
                break
        tail = rfilter.flush()
        if tail:
            emitted = True
            yield tail
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"[LLM] Ollama STREAM ok in {dt_ms:.1f} ms | model={OLLAMA_MODEL} | ctx={ctx}")
        _last_meta["model"] = OLLAMA_MODEL
        _last_meta["tokens_in"] = final.get("prompt_eval_count", 0)
        _last_meta["tokens_out"] = final.get("eval_count", 0)
        if not emitted:
            yield "[GROUND] Ollama returned an empty response."

    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:600]
        except Exception:
            pass
        # Text-only model got an image — fall back to the non-stream text-only retry.
        if images and "multimodal" in body.lower():
            print(f"[LLM] {OLLAMA_MODEL} is text-only; retrying without the image (non-stream)")
            yield _generate_ollama(prompt, None)   # sets _last_meta itself
            return
        # OOM / ctx-too-large before tokens flowed — retry once at 8k, streamed.
        oom_signals = ("out of memory", "context length", "kv cache", "failed to allocate")
        if not emitted and any(s in body.lower() for s in oom_signals) and ctx > 8192:
            print(f"[LLM] ctx={ctx} OOM for {OLLAMA_MODEL}, retrying stream at 8192")
            yield from _generate_ollama_stream(prompt, images, ctx=8192)
            return
        yield f"[GROUND] Ollama HTTP error: {e}\n{body}"
    except requests.exceptions.ConnectionError:
        yield "[GROUND] Can't connect to Ollama at localhost:11434."
    except requests.exceptions.Timeout:
        dt_ms = (time.perf_counter() - t0) * 1000
        yield f"[GROUND] Ollama stream timed out after {dt_ms:.0f} ms."
    except Exception as e:
        yield f"[GROUND] Ollama stream error: {e}"


def _generate_gemini(prompt: str, images=None) -> str:
    """Call Gemini via Vertex AI (bills to GCP credits) or AI Studio fallback.

    With images, appends inline_data parts (Gemini's own schema — not image_url).
    """
    token = _get_gemini_access_token()
    if token:
        # Vertex AI endpoint — uses cloud-platform scope, bills to GCP project
        try:
            project = json.load(open(GEMINI_ADC_PATH)).get("quota_project_id", "")
        except Exception:
            project = ""
        if not project:
            return "[GROUND] Gemini: ADC file has no quota_project_id — set one with gcloud config set project <id>."
        # gemini-3.x is served on the GLOBAL endpoint only, not regional (probed 2026-05-30)
        _region = "global" if GEMINI_MODEL.startswith("gemini-3") else GEMINI_REGION
        _host = "aiplatform.googleapis.com" if _region == "global" else f"{_region}-aiplatform.googleapis.com"
        url = (
            f"https://{_host}/v1/projects/{project}"
            f"/locations/{_region}/publishers/google/models/{GEMINI_MODEL}:generateContent"
        )
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    elif GEMINI_API_KEY:
        # AI Studio fallback — different billing/quota, no GCP credits
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        )
        headers = {"Content-Type": "application/json"}
    else:
        return "[GROUND] Gemini: no ADC credentials and no GEMINI_API_KEY set."

    # Vision: parts is text + inline_data blocks (Gemini's schema, not image_url)
    parts: list = [{"text": prompt}]
    if images:
        for b64 in images:
            parts.append({"inline_data": {"mime_type": _sniff_image_mime(b64), "data": b64}})
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
        },
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=(CONNECT_TIMEOUT, API_READ_TIMEOUT),
        )
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            or ""
        ).strip()
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"[LLM] Gemini ok in {dt_ms:.1f} ms | model={GEMINI_MODEL}")
        usage = data.get("usageMetadata", {})
        _last_meta["model"]      = GEMINI_MODEL
        _last_meta["tokens_in"]  = usage.get("promptTokenCount", 0)
        _last_meta["tokens_out"] = usage.get("candidatesTokenCount", 0)
        return text or "[GROUND] Gemini returned empty response."
    except requests.exceptions.ConnectionError:
        return "[GROUND] Can't connect to Gemini API."
    except requests.exceptions.Timeout:
        dt_ms = (time.perf_counter() - t0) * 1000
        return f"[GROUND] Gemini call timed out after {dt_ms:.0f} ms."
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            pass
        return f"[GROUND] Gemini HTTP error: {e}\n{body}"
    except Exception as e:
        return f"[GROUND] Gemini error: {e}"


def _generate_deepseek(prompt: str) -> str:
    """Call DeepSeek API (OpenAI-compatible endpoint)."""
    # Needs DEEPSEEK_API_KEY in .env — separate from GEMINI_API_KEY
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": NUM_PREDICT,
        "temperature": TEMPERATURE,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            DEEPSEEK_URL,
            headers=headers,
            json=payload,
            timeout=(CONNECT_TIMEOUT, API_READ_TIMEOUT),
        )
        resp.raise_for_status()
        data = resp.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"[LLM] DeepSeek ok in {dt_ms:.1f} ms | model={DEEPSEEK_MODEL}")
        usage = data.get("usage", {})
        _last_meta["model"] = DEEPSEEK_MODEL
        _last_meta["tokens_in"]  = usage.get("prompt_tokens", 0)
        _last_meta["tokens_out"] = usage.get("completion_tokens", 0)
        return text or "[GROUND] DeepSeek returned empty response."
    except requests.exceptions.ConnectionError:
        return "[GROUND] Can't connect to DeepSeek API."
    except requests.exceptions.Timeout:
        dt_ms = (time.perf_counter() - t0) * 1000
        return f"[GROUND] DeepSeek call timed out after {dt_ms:.0f} ms."
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            pass
        return f"[GROUND] DeepSeek HTTP error: {e}\n{body}"
    except Exception as e:
        return f"[GROUND] DeepSeek error: {e}"


def _generate_openai(prompt: str, images=None) -> str:
    """Call OpenAI chat completions API. With images, builds a vision content array."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    # Vision: OpenAI wants a full data URL (mime baked in), unlike Claude's split fields.
    if images:
        content: Any = [{"type": "text", "text": prompt}]
        for b64 in images:
            mime = _sniff_image_mime(b64)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
    else:
        content = prompt
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": NUM_PREDICT,
        "temperature": TEMPERATURE,
    }
    # o-series and the gpt-5 family (gpt-5, gpt-5.5) don't support temperature (only default 1.0)
    if OPENAI_MODEL.startswith("o") or OPENAI_MODEL.startswith("gpt-5"):
        del payload["temperature"]
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            OPENAI_URL,
            headers=headers,
            json=payload,
            timeout=(CONNECT_TIMEOUT, API_READ_TIMEOUT),
        )
        resp.raise_for_status()
        data = resp.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"[LLM] OpenAI ok in {dt_ms:.1f} ms | model={OPENAI_MODEL}")
        usage = data.get("usage", {})
        _last_meta["model"]      = OPENAI_MODEL
        _last_meta["tokens_in"]  = usage.get("prompt_tokens", 0)
        _last_meta["tokens_out"] = usage.get("completion_tokens", 0)
        return text or "[GROUND] OpenAI returned empty response."
    except requests.exceptions.ConnectionError:
        return "[GROUND] Can't connect to OpenAI API."
    except requests.exceptions.Timeout:
        dt_ms = (time.perf_counter() - t0) * 1000
        return f"[GROUND] OpenAI call timed out after {dt_ms:.0f} ms."
    except requests.exceptions.HTTPError as e:
        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            pass
        return f"[GROUND] OpenAI HTTP error: {e}\n{body}"
    except Exception as e:
        return f"[GROUND] OpenAI error: {e}"
