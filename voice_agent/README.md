# 🎙️ VoiceAgent — Voice-Controlled Local AI Agent

A full-stack, privacy-first AI agent that accepts audio or text commands, classifies intent, and executes actions on your local machine — all inside a polished Streamlit UI.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Audio input** | Upload `.wav`, `.mp3`, `.m4a`, `.ogg` or type directly |
| **STT backends** | faster-whisper (local) · Groq API · OpenAI API |
| **LLM backends** | Ollama (local) · LM Studio · OpenAI API |
| **Intents** | create_file · write_code · summarize · general_chat · list_files · read_file · delete_file |
| **Compound commands** | "Summarise this and save to summary.txt" |
| **Human-in-the-loop** | Confirmation prompt before any file operation |
| **Session memory** | Rolling context window injected into LLM prompts |
| **Safety sandbox** | All file ops restricted to `output/` folder |
| **Graceful errors** | Every stage wrapped with informative error messages |

---

## 🏗️ Architecture

```
Audio/Text Input
      │
      ▼
┌─────────────┐
│   stt.py    │  faster-whisper / Groq / OpenAI  → transcribed text
└─────────────┘
      │
      ▼
┌─────────────┐
│  intent.py  │  Ollama / LM Studio / OpenAI     → structured JSON intent
└─────────────┘
      │
      ▼
┌─────────────┐
│   tools.py  │  file ops / code gen / chat       → result + files
└─────────────┘
      │
      ▼
┌─────────────┐
│   app.py    │  Streamlit UI                     → display pipeline
└─────────────┘
      │
┌─────────────┐
│  memory.py  │  Session context store
└─────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/voice-agent.git
cd voice-agent
pip install -r requirements.txt
```

### 2. Start an LLM (Ollama recommended)

```bash
# Install Ollama: https://ollama.com
ollama serve
ollama pull llama3.2   # or mistral, llama3.1, etc.
```

### 3. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## ⚙️ Configuration

All settings are in the **sidebar** — no `.env` file needed. However, you can pre-set API keys via environment variables:

```bash
export GROQ_API_KEY=your_groq_key
export OPENAI_API_KEY=your_openai_key
```

### STT Backend Options

| Backend | Requirement | Notes |
|---|---|---|
| `whisper-local` | `faster-whisper` pip package | Runs on CPU, ~1-3 s for short clips |
| `groq-api` | `groq` pip package + API key | Very fast, free tier available |
| `openai-api` | `openai` pip package + API key | Reliable, costs per request |

**Hardware note:** If your machine cannot run `faster-whisper` efficiently (no AVX2 support or <4 GB RAM), use Groq API — it's free and faster than local inference.

### LLM Backend Options

| Backend | Requirement | Notes |
|---|---|---|
| `ollama` | Ollama running locally | Best for privacy, free |
| `lm-studio` | LM Studio running locally | OpenAI-compatible API at port 1234 |
| `openai-api` | OpenAI API key | Best quality, costs per request |

---

## 🗂️ Project Structure

```
voice-agent/
├── app.py              # Streamlit UI + pipeline orchestration
├── stt.py              # Speech-to-text module
├── intent.py           # LLM-powered intent classification
├── tools.py            # Tool execution (file ops, code gen, chat)
├── memory.py           # In-session memory
├── requirements.txt    # Python dependencies
├── README.md
└── output/             # ← ALL generated files land here (auto-created)
```

---

## 🎯 Supported Intents

### `create_file`
> "Create a new markdown file called notes.md"

Creates a file in `output/` with an optional content hint.

### `write_code`
> "Write a Python script that implements a binary search tree"
> "Create a JavaScript file with a debounce function"

Generates code via LLM and saves to `output/`.

### `summarize`
> "Summarise the concept of transformer attention mechanisms"
> "Summarise this text and save it to summary.txt"

Returns bullet-point summary; optionally saves to file (compound command).

### `general_chat`
> "What is the difference between TCP and UDP?"

Returns a conversational LLM response.

### `list_files`
> "What files have been created?"

Lists all files in `output/`.

### `read_file`
> "Read the contents of hello.py"

Displays the file content.

### `delete_file`
> "Delete notes.md"

Removes the file from `output/`.

---

## 🔒 Safety

- **Sandbox:** All file creation and writing is restricted to the `output/` directory. Path traversal attempts are sanitised.
- **Human-in-loop:** Enable "Confirm before file ops" in the sidebar to require manual confirmation before any write/create action.
- **No system access:** The agent cannot modify files outside `output/`.

---

## 🌟 Bonus Features Implemented

- ✅ **Compound commands** — secondary intents are detected and executed
- ✅ **Human-in-the-loop** — confirmation prompt for file operations
- ✅ **Graceful degradation** — all errors surfaced in UI, never crash
- ✅ **Session memory** — rolling context window for coherent multi-turn use
- ✅ **Output file browser** — view, preview, and download all created files

---

## 🛠️ Troubleshooting

**`faster-whisper` fails on CPU:**
Switch to Groq API (sidebar → STT Backend → groq-api).

**Ollama connection refused:**
Make sure `ollama serve` is running. Check host URL in sidebar.

**LLM returns non-JSON:**
The intent parser has a fallback to extract JSON from partial responses. If it still fails, the agent falls back to `general_chat`.

---

## 📄 License

MIT
