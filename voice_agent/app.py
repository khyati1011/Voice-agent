import streamlit as st
import os
from pathlib import Path
from datetime import datetime

from stt import transcribe_audio
from intent import classify_intent
from tools import execute_tool
from memory import SessionMemory

st.set_page_config(
    page_title="VoiceAgent",
    page_icon="🎙️",
    layout="wide",
)

st.markdown("""
<style>


/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0d0d12, #1a1a22);
}

/* ── Global text ── */
html, body, [class*="css"] {
    color: #e8e8ec !important;
    font-family: 'Inter', sans-serif;
}

/* ── LOGO (single color, like you asked for once) ── */
.logo {
    font-size: 36px;
    font-weight: 700;
    color: #7c8cff;   /* change this if you want another vibe */
    letter-spacing: 0.5px;
}

/* ── Subtitle ── */
.subtitle {
    color: #9aa0aa;
    font-size: 14px;
    margin-bottom: 15px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111118;
    border-right: 1px solid rgba(255,255,255,0.05);
}
[data-testid="stSidebar"] * {
    color: #e8e8ec !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.stTabs [aria-selected="true"] {
    color: #7c8cff !important;
    border-bottom: 2px solid #7c8cff;
}

/* ── Inputs ── */
textarea, input {
    background: rgba(255,255,255,0.05) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 10px;
}

.stButton button {
     background-color: #1E90FF;  
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 12em;
        font-size: 16px;
    border: none;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    transition: 0.2s;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
}
.stButton button:hover {
    transform: scale(1.05);
}

/* ── Cards ── */
.card {
    padding: 18px;
    border-radius: 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    margin-bottom: 12px;
}

/* ── Tag ── */
.tag {
    background: rgba(124,140,255,0.15);
    color: #7c8cff;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 12px;
}

/* ── Divider ── */
hr {
    border-top: 1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# session state
if "memory"      not in st.session_state: st.session_state.memory      = SessionMemory()
if "history"     not in st.session_state: st.session_state.history     = []
if "last_result" not in st.session_state: st.session_state.last_result = None
memory: SessionMemory = st.session_state.memory

# sidebar
with st.sidebar:
    st.markdown("## 🎙️ VoiceAgent")
    st.caption("Local AI · Voice Controlled")
    st.divider()
    st.markdown("**Settings**")
    stt_backend = st.selectbox("STT Backend", ["whisper-local", "groq-api", "openai-api"])
    llm_backend = st.selectbox("LLM Backend", ["ollama", "lm-studio", "openai-api"])
    if llm_backend == "ollama":
        ollama_model = st.text_input("Ollama Model", value="llama3.2")
        ollama_host  = st.text_input("Ollama Host",  value="http://localhost:11434")
    elif llm_backend == "lm-studio":
        lmstudio_url = st.text_input("LM Studio URL", value="http://localhost:1234/v1")
    else:
        openai_key = st.text_input("OpenAI API Key", type="password")
    if stt_backend == "groq-api":
        groq_key = st.text_input("Groq API Key", type="password")
    elif stt_backend == "openai-api":
        openai_stt_key = st.text_input("OpenAI STT Key", type="password")
    st.divider()
    st.markdown("**Options**")
    human_in_loop = st.checkbox("Confirm before file operations", value=True)
    show_raw      = st.checkbox("Show raw LLM response", value=False)
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Queries", len(st.session_state.history))
    c2.metric("Files",   memory.files_created)
    if st.button("Clear session"):
        st.session_state.history     = []
        st.session_state.last_result = None
        st.session_state.memory      = SessionMemory()
        st.rerun()

cfg = {
    "stt_backend":   stt_backend,
    "llm_backend":   llm_backend,
    "ollama_model":  locals().get("ollama_model",  "llama3.2"),
    "ollama_host":   locals().get("ollama_host",   "http://localhost:11434"),
    "lmstudio_url":  locals().get("lmstudio_url",  "http://localhost:1234/v1"),
    "openai_key":    locals().get("openai_key",    os.getenv("OPENAI_API_KEY", "")),
    "groq_key":      locals().get("groq_key",      os.getenv("GROQ_API_KEY",  "")),
    "human_in_loop": human_in_loop,
}

# ── Header ─────────────────────────────

st.markdown("""
<div class="logo">🎙️ VoiceAgent</div>

<div style="font-size:24px; font-weight:600; margin-top:5px;">
🎤 Voice-Controlled Local AI Agent
</div>

<div style="color:#9aa0aa; font-size:13px; margin-top:5px;">
Speak a command → Transcribed → Intent classified → Tool executed
</div>
""", unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["Agent", "History", "Output Files"])

# TAB 1
with tab1:
    input_method = st.radio("", ["Upload audio file", "Type text directly"], horizontal=True)
    audio_bytes = None
    text_override = None

    if input_method == "Upload audio file":
        uploaded = st.file_uploader("Upload .wav or .mp3", type=["wav","mp3","m4a","ogg"])
        if uploaded:
            st.audio(uploaded)
            audio_bytes = uploaded.read()
    else:
        text_override = st.text_area("", placeholder="e.g. Write a Python file with a retry function", height=90, label_visibility="collapsed")

    st.write("")
    run = st.button("Run Agent")

    if run:
        if not audio_bytes and not text_override:
            st.error("Please upload a file or type a command.")
        else:
            result   = {}
            progress = st.progress(0, text="Starting…")

            if text_override:
                result["transcription"] = text_override.strip()
                result["stt_method"]    = "typed input"
                progress.progress(33, text="Transcription ready")
            else:
                with st.spinner("Transcribing…"):
                    try:
                        t, info = transcribe_audio(audio_bytes, cfg)
                        result["transcription"] = t
                        result["stt_method"]    = info
                        progress.progress(33, text="Transcription done")
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                        st.stop()

            with st.spinner("Detecting intent…"):
                try:
                    intent_data, raw = classify_intent(result["transcription"], memory.get_context(), cfg)
                    result["intent_data"] = intent_data
                    result["raw_llm"]     = raw
                    progress.progress(66, text="Intent detected")
                except Exception as e:
                    st.error(f"Intent detection failed: {e}")
                    st.stop()

            confirmed  = True
            needs_file = intent_data.get("primary_intent") in ["create_file","write_code"]
            if cfg["human_in_loop"] and needs_file:
                st.warning(f"About to **{intent_data.get('primary_intent')}**. Proceed?")
                b1, b2 = st.columns([1,5])
                if b1.button("Yes"): confirmed = True
                if b2.button("No"):
                    confirmed = False
                    st.info("Cancelled.")

            if confirmed:
                with st.spinner("Running…"):
                    try:
                        tool_result = execute_tool(intent_data, result["transcription"], cfg)
                        result["tool_result"] = tool_result
                        progress.progress(100, text="Done")
                        for f in tool_result.get("files_created", []):
                            memory.record_file(f)
                    except Exception as e:
                        st.error(f"Tool failed: {e}")
                        result["tool_result"] = {"error": str(e), "output": "", "action": "error", "message": str(e), "files_created": []}
                        progress.progress(100)

            result["timestamp"] = datetime.now().strftime("%H:%M:%S")
            st.session_state.history.append(result)
            st.session_state.last_result = result
            memory.add_turn(result["transcription"], intent_data.get("primary_intent","unknown"))

    if st.session_state.last_result:
        r  = st.session_state.last_result
        tr = r.get("tool_result", {})
        intent = r.get("intent_data", {})

        st.divider()
        st.markdown("**Results**")

        st.markdown(f"""
        <div class="rbox">
            <div class="rlabel">Transcription &nbsp;·&nbsp; {r.get('stt_method','')}</div>
            <div class="rvalue">"{r.get('transcription','')}"</div>
        </div>""", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        primary    = intent.get("primary_intent","unknown").replace("_"," ").title()
        confidence = intent.get("confidence","")
        target     = intent.get("target_file","")
        ok         = not tr.get("error")

        with col_a:
            sec = "".join(f'<span class="tag">{s.replace("_"," ").title()}</span>' for s in intent.get("secondary_intents",[]))
            st.markdown(f"""
            <div class="rbox">
                <div class="rlabel">Detected Intent</div>
                <div class="rvalue">
                    <span class="tag">{primary}</span>{sec}
                    <br><br>
                    <span style="font-size:12px;color:#aaa">Confidence: {confidence}</span>
                    {"<br><span style='font-size:12px;color:#aaa'>File: "+target+"</span>" if target else ""}
                </div>
            </div>""", unsafe_allow_html=True)

        with col_b:
            status = f'<span class="tag tag-{"green" if ok else "red"}">{"Success" if ok else "Failed"}</span>'
            files  = "".join(f'<div style="font-size:12px;color:#888;margin-top:4px">📄 {f}</div>' for f in tr.get("files_created",[]))
            st.markdown(f"""
            <div class="rbox">
                <div class="rlabel">Action Taken</div>
                <div class="rvalue">
                    {status}
                    <span style="font-size:13px;color:#666;margin-left:8px">{tr.get('message','')}</span>
                    {files}
                </div>
            </div>""", unsafe_allow_html=True)

        if tr.get("output"):
            st.markdown(f"""
            <div class="rbox">
                <div class="rlabel">Output</div>
                <div class="codeout">{tr['output']}</div>
            </div>""", unsafe_allow_html=True)

        if show_raw and r.get("raw_llm"):
            with st.expander("Raw LLM response"):
                st.code(r["raw_llm"])

# TAB 2
with tab2:
    st.markdown("**Session History**")
    if not st.session_state.history:
        st.caption("Nothing yet.")
    else:
        for h in reversed(st.session_state.history):
            intent  = h.get("intent_data",{}).get("primary_intent","unknown")
            ts      = h.get("timestamp","")
            ok      = not h.get("tool_result",{}).get("error")
            preview = h.get("transcription","")[:60]
            with st.expander(f"{'✓' if ok else '✗'}  [{ts}]  {preview}  ·  {intent}"):
                st.write("**Command:**", h.get("transcription",""))
                st.write("**Intent:**",  intent)
                tr2 = h.get("tool_result",{})
                st.write("**Action:**",  tr2.get("action",""), "—", tr2.get("message",""))
                if tr2.get("output"):
                    st.code(tr2["output"])

# TAB 3
with tab3:
    st.markdown("**Files in `output/`**")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    files = sorted([f for f in output_dir.rglob("*") if f.is_file()])
    if not files:
        st.caption("No files yet.")
    else:
        for f in files:
            with st.expander(f"📄  {f.name}  ·  {f.stat().st_size} bytes"):
                try:
                    content = f.read_text(encoding="utf-8")
                    ext  = f.suffix.lower().lstrip(".")
                    lang = {"py":"python","js":"javascript","ts":"typescript","html":"html",
                            "css":"css","json":"json","md":"markdown","sh":"bash"}.get(ext,"text")
                    st.code(content, language=lang)
                    st.download_button(f"Download {f.name}", content, file_name=f.name, key=f"dl_{f}")
                except Exception:
                    st.warning("Cannot preview this file.")