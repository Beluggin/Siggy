# adapters.py
import anthropic

claude_client = anthropic.Anthropic()

CLAUDE_SYSTEM = """You are Claude, participating in a multi-LLM meeting room \
hosted by Adam, an independent AI researcher. Other participants may include \
Gemini (Google) and ChatGPT (OpenAI). Adam is the human host and moderator.

The transcript shows each speaker tagged in brackets like [Adam]: or [Gemini]:. \
Speak only as yourself. Do not impersonate other participants or narrate their turns. \
When you respond, do not prefix your own message with [Claude]: — just write the content. \
Keep responses focused and substantive; this is a working session, not a chatroom hangout."""

def claude_adapter(transcript, my_name="Claude", model="claude-opus-4-7"):
    """Format the canonical transcript for Claude and call the API."""
    messages = []
    pending_user_chunks = []

    for entry in transcript:
        speaker = entry["speaker"]
        content = entry["content"]

        if speaker == my_name:
            # Flush any pending user chunks first
            if pending_user_chunks:
                messages.append({
                    "role": "user",
                    "content": "\n\n".join(pending_user_chunks)
                })
                pending_user_chunks = []
            messages.append({"role": "assistant", "content": content})
        else:
            pending_user_chunks.append(f"[{speaker}]: {content}")

    # Flush trailing user chunks
    if pending_user_chunks:
        messages.append({
            "role": "user",
            "content": "\n\n".join(pending_user_chunks)
        })

    # Edge case: if Claude is being invoked but the last message ended up as
    # assistant (Claude was the most recent speaker), the API will reject.
    # Add a nudge prompt.
    if messages and messages[-1]["role"] == "assistant":
        messages.append({
            "role": "user",
            "content": "[Adam]: Continue."
        })

    response = claude_client.messages.create(
        model=model,
        max_tokens=1024,
        system=CLAUDE_SYSTEM,
        messages=messages,
    )
    return response.content[0].text

# Gemini Adapter (See later to do tag)
import json
import os
import time
import requests
from pathlib import Path

# --- Gemini via Vertex AI (raw requests, mirrors SignalBot v8 pattern) ---

GEMINI_ADC_PATH = Path.home() / ".config/gcloud/application_default_credentials.json"
GEMINI_REGION = os.environ.get("GEMINI_REGION", "us-central1")
# TODO: swap to "gemini-3.5-flash" once it lands on Vertex AI (currently 404)
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MAX_OUTPUT_TOKENS = 8192  # thinking models burn tokens internally — keep high

_gemini_token_cache = {"token": None, "expires_at": 0}

def _get_gemini_access_token():
    """Refresh OAuth token from ADC and cache until ~expiry."""
    now = time.time()
    if _gemini_token_cache["token"] and _gemini_token_cache["expires_at"] > now + 60:
        return _gemini_token_cache["token"]

    with open(GEMINI_ADC_PATH) as f:
        adc = json.load(f)

    r = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": adc["client_id"],
            "client_secret": adc["client_secret"],
            "refresh_token": adc["refresh_token"],
            "grant_type": "refresh_token",
        },
        timeout=15,
    )
    r.raise_for_status()
    payload = r.json()
    _gemini_token_cache["token"] = payload["access_token"]
    _gemini_token_cache["expires_at"] = now + payload.get("expires_in", 3600)
    return _gemini_token_cache["token"]

def _get_gemini_project():
    with open(GEMINI_ADC_PATH) as f:
        return json.load(f)["quota_project_id"]


GEMINI_SYSTEM = """You are Gemini, participating in a multi-LLM meeting room \
hosted by Adam, an independent AI researcher. Other participants include \
Claude (Anthropic) and may include ChatGPT (OpenAI). Adam is the human host.

The transcript shows each speaker tagged in brackets like [Adam]: or [Claude]:. \
Speak only as yourself. Do not impersonate other participants. Do not prefix \
your own message with [Gemini]: — just write the content. Keep responses \
focused and substantive; this is a working session."""


def gemini_adapter(transcript, my_name="Gemini", model=GEMINI_MODEL):
    """Format the canonical transcript and call Vertex AI Gemini."""
    contents = []
    pending_user_chunks = []

    for entry in transcript:
        speaker = entry["speaker"]
        content = entry["content"]

        if speaker == my_name:
            if pending_user_chunks:
                contents.append({
                    "role": "user",
                    "parts": [{"text": "\n\n".join(pending_user_chunks)}]
                })
                pending_user_chunks = []
            contents.append({
                "role": "model",
                "parts": [{"text": content}]
            })
        else:
            pending_user_chunks.append(f"[{speaker}]: {content}")

    if pending_user_chunks:
        contents.append({
            "role": "user",
            "parts": [{"text": "\n\n".join(pending_user_chunks)}]
        })

    if contents and contents[-1]["role"] == "model":
        contents.append({
            "role": "user",
            "parts": [{"text": "[Adam]: Continue."}]
        })

    token = _get_gemini_access_token()
    project = _get_gemini_project()
    url = (
        f"https://{GEMINI_REGION}-aiplatform.googleapis.com/v1/projects/"
        f"{project}/locations/{GEMINI_REGION}/publishers/google/models/"
        f"{model}:generateContent"
    )
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": GEMINI_SYSTEM}]},
        "generationConfig": {
            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
            "temperature": 0.7,
        },
    }

    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()

    # Defensive parsing — thinking models can return multiple/empty parts
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {data}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text_chunks = [p.get("text", "") for p in parts if p.get("text")]
    if not text_chunks:
        raise RuntimeError(f"Gemini returned no text content: {data}")
    return "\n".join(text_chunks).strip()


# --- ChatGPT via OpenAI REST API (no SDK needed — same pattern as Gemini) ---
# Model: gpt-5.5 — update CHATGPT_MODEL if your account uses a different model ID
# Needs OPENAI_API_KEY in environment.

CHATGPT_MODEL = "gpt-5.5"

CHATGPT_SYSTEM = """You are ChatGPT (OpenAI), participating in a multi-LLM meeting room \
hosted by Adam, an independent AI researcher. Other participants include \
Claude (Anthropic) and Gemini (Google). Adam is the human host and moderator.

The transcript shows each speaker tagged in brackets like [Adam]: or [Claude]:. \
Speak only as yourself. Do not impersonate other participants. Do not prefix \
your own message with [ChatGPT]: — just write the content. Keep responses \
focused and substantive; this is a working session."""


def chatgpt_adapter(transcript, my_name="ChatGPT", model=CHATGPT_MODEL):
    """Format the canonical transcript for ChatGPT and call the OpenAI API."""
    api_key = os.environ.get("OPENAI_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_KEY not set in .env")

    messages = [{"role": "system", "content": CHATGPT_SYSTEM}]
    for entry in transcript:
        speaker = entry["speaker"]
        content = entry["content"]
        if speaker == my_name:
            messages.append({"role": "assistant", "content": content})
        else:
            messages.append({"role": "user", "content": f"[{speaker}]: {content}"})

    # OpenAI rejects a request that ends on an assistant turn — add a nudge
    if messages and messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "[Adam]: Continue."})

    payload = {"model": model, "messages": messages, "max_completion_tokens": 1024}
    # gpt-5 family doesn't accept a temperature parameter
    if not model.startswith("gpt-5") and not model.startswith("o"):
        payload["temperature"] = 0.7

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()
