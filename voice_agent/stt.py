"""
stt.py — Speech-to-Text module
Supports:
  • whisper-local  : faster-whisper (runs on CPU/GPU locally)
  • groq-api       : Groq Whisper API (fast, free tier available)
  • openai-api     : OpenAI Whisper API
"""

from __future__ import annotations
import io
import os
import tempfile
from typing import Tuple


def transcribe_audio(audio_bytes: bytes, cfg: dict) -> Tuple[str, str]:
    """
    Returns (transcribed_text, backend_info_string).
    Raises on failure so caller can surface the error.
    """
    backend = cfg.get("stt_backend", "whisper-local")

    if backend == "whisper-local":
        return _whisper_local(audio_bytes)
    elif backend == "groq-api":
        return _groq(audio_bytes, cfg)
    elif backend == "openai-api":
        return _openai_stt(audio_bytes, cfg)
    else:
        raise ValueError(f"Unknown STT backend: {backend}")


# ─── Local Whisper ────────────────────────────────────────────────────────────

def _whisper_local(audio_bytes: bytes) -> Tuple[str, str]:
    """Uses faster-whisper (CPU-friendly CTranslate2 backend)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper is not installed.\n"
            "Run: pip install faster-whisper\n"
            "Or switch to groq-api / openai-api in the sidebar."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # "base" is fast enough on CPU; change to "small" or "medium" for more accuracy
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(tmp_path, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments)
        lang = info.language
        return text.strip(), f"faster-whisper/base ({lang})"
    finally:
        os.unlink(tmp_path)


# ─── Groq API ─────────────────────────────────────────────────────────────────

def _groq(audio_bytes: bytes, cfg: dict) -> Tuple[str, str]:
    """Uses Groq's Whisper endpoint — very fast, generous free tier."""
    api_key = cfg.get("groq_key") or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("Groq API key is required. Set it in the sidebar or GROQ_API_KEY env var.")

    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq SDK not installed. Run: pip install groq")

    client = Groq(api_key=api_key)
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"

    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3-turbo",
        response_format="text",
    )
    return str(transcription).strip(), "Groq / whisper-large-v3-turbo"


# ─── OpenAI STT ───────────────────────────────────────────────────────────────

def _openai_stt(audio_bytes: bytes, cfg: dict) -> Tuple[str, str]:
    api_key = cfg.get("openai_key") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OpenAI API key is required.")

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai SDK not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"

    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text",
    )
    return str(transcription).strip(), "OpenAI / whisper-1"
