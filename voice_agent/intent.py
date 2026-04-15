"""
intent.py — Intent classification via LLM
Supports: ollama, lm-studio (OpenAI-compatible), openai-api
"""

from __future__ import annotations
import json
import re
import os
from typing import Tuple

SYSTEM_PROMPT = """You are an intent classification engine for a voice-controlled AI agent.

Given a user's transcribed command, analyse it and return a JSON object with:
{
  "primary_intent": one of ["create_file", "write_code", "summarize", "general_chat", "list_files", "read_file", "delete_file"],
  "secondary_intents": [],           // additional intents if compound command
  "confidence": "high|medium|low",
  "target_file": null or "filename.ext",
  "target_folder": null or "foldername",
  "language": null or "python|javascript|etc",
  "content_hint": null or "brief description of what to generate/do",
  "summary_target": null or "the text or topic to summarize"
}

Rules:
- "write_code" means the user wants code generated and saved.
- "create_file" means create a new file (possibly empty, or with content).
- "summarize" means the user wants some text or topic summarised.
- "general_chat" is a fallback for anything else.
- For compound commands (e.g. "summarise this and save to summary.txt"), set primary_intent to the main action and list others in secondary_intents.
- ALWAYS return only valid JSON. No explanation, no markdown fences.
"""

def classify_intent(
    text: str,
    context: str,
    cfg: dict,
) -> Tuple[dict, str]:
    """
    Returns (intent_dict, raw_llm_response_string).
    """
    backend = cfg.get("llm_backend", "ollama")

    if backend == "ollama":
        raw = _ollama(text, context, cfg)
    elif backend == "lm-studio":
        raw = _lmstudio(text, context, cfg)
    elif backend == "openai-api":
        raw = _openai_llm(text, context, cfg)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")

    intent_data = _parse_json(raw)
    return intent_data, raw


# ─── Ollama ────────────────────────────────────────────────────────────────────

def _ollama(text: str, context: str, cfg: dict) -> str:
    try:
        import requests
    except ImportError:
        raise ImportError("requests not installed. Run: pip install requests")

    host  = cfg.get("ollama_host", "http://localhost:11434")
    model = cfg.get("ollama_model", "llama3.2")

    messages = _build_messages(text, context)
    payload  = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1},
    }

    resp = requests.post(f"{host}/api/chat", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


# ─── LM Studio ────────────────────────────────────────────────────────────────

def _lmstudio(text: str, context: str, cfg: dict) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai SDK not installed. Run: pip install openai")

    base_url = cfg.get("lmstudio_url", "http://localhost:1234/v1")
    client   = OpenAI(base_url=base_url, api_key="lm-studio")

    messages = _build_messages(text, context)
    resp = client.chat.completions.create(
        model="local-model",
        messages=messages,
        temperature=0.1,
    )
    return resp.choices[0].message.content


# ─── OpenAI ───────────────────────────────────────────────────────────────────

def _openai_llm(text: str, context: str, cfg: dict) -> str:
    api_key = cfg.get("openai_key") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OpenAI API key required.")

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai SDK not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)
    messages = _build_messages(text, context)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
    )
    return resp.choices[0].message.content


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_messages(text: str, context: str) -> list:
    user_content = f"User command: {text}"
    if context:
        user_content = f"Session context:\n{context}\n\n{user_content}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def _parse_json(raw: str) -> dict:
    """Safely parse LLM JSON output, stripping markdown fences if present."""
    cleaned = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt to extract first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        # Fallback: return general_chat intent
        return {
            "primary_intent": "general_chat",
            "secondary_intents": [],
            "confidence": "low",
            "target_file": None,
            "target_folder": None,
            "language": None,
            "content_hint": raw,
            "summary_target": None,
        }
