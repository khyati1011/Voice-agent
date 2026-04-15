"""
tools.py — Tool execution for all supported intents.

All file operations are sandboxed to the output/ directory.
"""

from __future__ import annotations
import os
import re
import json
import datetime
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path("output")


def execute_tool(intent_data: dict, original_text: str, cfg: dict) -> dict:
    """
    Dispatch to the correct tool handler based on primary_intent.
    Returns a dict with keys: action, message, output, files_created (list), error (optional)
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    intent = intent_data.get("primary_intent", "general_chat")

    handlers = {
        "create_file":  _handle_create_file,
        "write_code":   _handle_write_code,
        "summarize":    _handle_summarize,
        "general_chat": _handle_general_chat,
        "list_files":   _handle_list_files,
        "read_file":    _handle_read_file,
        "delete_file":  _handle_delete_file,
    }

    handler = handlers.get(intent, _handle_general_chat)
    result  = handler(intent_data, original_text, cfg)

    # Handle compound commands — secondary intents
    secondary = intent_data.get("secondary_intents", [])
    for sec_intent in secondary:
        if sec_intent in handlers and sec_intent != intent:
            sec_result = handlers[sec_intent](intent_data, original_text, cfg)
            # Merge secondary results
            result["output"] += f"\n\n--- Secondary: {sec_intent} ---\n" + sec_result.get("output", "")
            result["files_created"].extend(sec_result.get("files_created", []))
            result["message"] += f" + {sec_intent}"

    return result


# ─── Create File ──────────────────────────────────────────────────────────────

def _handle_create_file(intent_data: dict, text: str, cfg: dict) -> dict:
    target = intent_data.get("target_file") or _infer_filename(text, "txt")
    target = _safe_filename(target)
    path   = OUTPUT_DIR / target

    # If a folder was mentioned, create it
    folder = intent_data.get("target_folder")
    if folder:
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / folder / target

    # Generate initial content if hinted
    hint = intent_data.get("content_hint", "")
    content = f"# {target}\nCreated by VoiceAgent on {_now()}\n"
    if hint:
        content += f"\n# Purpose: {hint}\n"

    path.write_text(content, encoding="utf-8")
    return {
        "action": "create_file",
        "message": f"Created {path}",
        "output": content,
        "files_created": [str(path)],
    }


# ─── Write Code ───────────────────────────────────────────────────────────────

def _handle_write_code(intent_data: dict, text: str, cfg: dict) -> dict:
    lang   = intent_data.get("language") or _infer_language(text)
    ext    = _lang_to_ext(lang)
    target = intent_data.get("target_file") or _infer_filename(text, ext)
    target = _safe_filename(target)
    path   = OUTPUT_DIR / target

    hint = intent_data.get("content_hint") or text
    code = _generate_code(lang, hint, cfg)

    path.write_text(code, encoding="utf-8")
    return {
        "action": "write_code",
        "message": f"Code written to {path}",
        "output": code,
        "files_created": [str(path)],
    }


# ─── Summarize ────────────────────────────────────────────────────────────────

def _handle_summarize(intent_data: dict, text: str, cfg: dict) -> dict:
    topic   = intent_data.get("summary_target") or intent_data.get("content_hint") or text
    summary = _call_llm(
        f"Please summarise the following in 3-5 concise bullet points:\n\n{topic}",
        cfg,
    )

    # Check if we should also save to file (compound)
    target = intent_data.get("target_file")
    files_created = []
    if target:
        target = _safe_filename(target)
        path   = OUTPUT_DIR / target
        path.write_text(summary, encoding="utf-8")
        files_created.append(str(path))

    return {
        "action": "summarize",
        "message": "Summary generated" + (f" and saved to {target}" if target else ""),
        "output": summary,
        "files_created": files_created,
    }


# ─── General Chat ─────────────────────────────────────────────────────────────

def _handle_general_chat(intent_data: dict, text: str, cfg: dict) -> dict:
    response = _call_llm(text, cfg)
    return {
        "action": "general_chat",
        "message": "Responded to general query",
        "output": response,
        "files_created": [],
    }


# ─── List Files ───────────────────────────────────────────────────────────────

def _handle_list_files(intent_data: dict, text: str, cfg: dict) -> dict:
    files = [str(f.relative_to(OUTPUT_DIR)) for f in OUTPUT_DIR.rglob("*") if f.is_file()]
    output = "Files in output/:\n" + ("\n".join(f"  • {f}" for f in files) if files else "  (empty)")
    return {"action": "list_files", "message": "Listed output files", "output": output, "files_created": []}


# ─── Read File ────────────────────────────────────────────────────────────────

def _handle_read_file(intent_data: dict, text: str, cfg: dict) -> dict:
    target = intent_data.get("target_file") or _infer_filename(text, "txt")
    target = _safe_filename(target)
    path   = OUTPUT_DIR / target

    if not path.exists():
        return {"action": "read_file", "message": f"{target} not found", "output": f"File not found: {target}", "files_created": [], "error": "not found"}

    content = path.read_text(encoding="utf-8")
    return {"action": "read_file", "message": f"Read {target}", "output": content, "files_created": []}


# ─── Delete File ─────────────────────────────────────────────────────────────

def _handle_delete_file(intent_data: dict, text: str, cfg: dict) -> dict:
    target = intent_data.get("target_file") or _infer_filename(text, "txt")
    target = _safe_filename(target)
    path   = OUTPUT_DIR / target

    if path.exists() and path.is_file():
        path.unlink()
        return {"action": "delete_file", "message": f"Deleted {target}", "output": f"✓ Deleted {target}", "files_created": []}
    return {"action": "delete_file", "message": f"{target} not found", "output": f"File not found: {target}", "files_created": [], "error": "not found"}


# ─── LLM helpers ─────────────────────────────────────────────────────────────

def _call_llm(prompt: str, cfg: dict) -> str:
    """Generic LLM call for code generation / chat / summarisation."""
    backend = cfg.get("llm_backend", "ollama")

    if backend == "ollama":
        import requests
        host  = cfg.get("ollama_host", "http://localhost:11434")
        model = cfg.get("ollama_model", "llama3.2")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.3},
        }
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    elif backend == "lm-studio":
        from openai import OpenAI
        client = OpenAI(base_url=cfg.get("lmstudio_url", "http://localhost:1234/v1"), api_key="lm-studio")
        resp = client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content

    elif backend == "openai-api":
        from openai import OpenAI
        client = OpenAI(api_key=cfg.get("openai_key") or os.getenv("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content

    return f"[No LLM backend available — prompt was: {prompt[:100]}]"


def _generate_code(lang: str, description: str, cfg: dict) -> str:
    prompt = (
        f"Write clean, well-commented {lang} code for the following task:\n{description}\n\n"
        f"Return ONLY the code, no explanation, no markdown fences."
    )
    return _call_llm(prompt, cfg)


# ─── Utility ──────────────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    """Strip path traversal attempts and sanitise."""
    name = Path(name).name  # strip directory parts
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "output.txt"


def _infer_filename(text: str, default_ext: str) -> str:
    """Try to pull a filename from the text, else generate one."""
    match = re.search(r'[\w\-]+\.\w{1,6}', text)
    if match:
        return match.group()
    slug = re.sub(r'\W+', '_', text[:30]).strip('_').lower()
    return f"{slug}.{default_ext}"


def _infer_language(text: str) -> str:
    text_lower = text.lower()
    for lang in ["python", "javascript", "typescript", "rust", "go", "java", "c++", "c", "bash", "sql", "html", "css"]:
        if lang in text_lower:
            return lang
    return "python"


def _lang_to_ext(lang: str) -> str:
    return {
        "python": "py", "javascript": "js", "typescript": "ts",
        "rust": "rs", "go": "go", "java": "java", "c++": "cpp",
        "c": "c", "bash": "sh", "sql": "sql", "html": "html",
        "css": "css", "markdown": "md",
    }.get(lang.lower(), "txt")


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
