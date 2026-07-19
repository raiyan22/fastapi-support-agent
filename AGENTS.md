# AGENTS.md

## What This Is

Local AI Customer Support Agent — FastAPI + llama.cpp (Phi-4-mini) + ChromaDB RAG. Zero cloud cost. All inference runs locally.

## Prerequisites (must be running)

1. **llama-server** on port 8080:
   ```
   llama-server -m models/Phi-4-mini-instruct-Q4_K_M.gguf --port 8080 --host 0.0.0.0 -c 8192
   ```
2. **ChromaDB populated** — run ingestion before first use (see below).

## Run Commands

```powershell
# 1. Start llama-server (separate terminal)
llama-server -m models/Phi-4-mini-instruct-Q4_K_M.gguf --port 8080 --host 0.0.0.0 -c 8192

# 2. Ingest knowledge base (only needed once, or after changing data/)
python -m app.rag.ingest

# 3. Start FastAPI
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Architecture

```
app/
├── main.py            ← FastAPI entrypoint, /chat endpoint, tool parsing loop
├── db.py              ← Async SQLAlchemy engine + session (SQLite via aiosqlite)
├── core/config.py     ← Settings from .env (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY)
├── models/schemas.py  ← SQLAlchemy models (TicketDB, ConversationDB) + Pydantic schemas
├── models/documents.py← DocumentDB model for uploaded files
├── routers/documents.py← Upload/List/Delete document endpoints
├── agent/tools.py     ← Tool implementations (tickets, escalation, RAG search) — async, uses DB
└── rag/
    ├── ingest.py      ← Loads data/*.txt + *.pdf → chunks → ChromaDB
    ├── retriever.py   ← similarity_search wrapper over ChromaDB
    └── embed.py       ← Single-file embed/delete for upload API
data/
└── knowledge/         ← Place .txt/.pdf files here (or upload via API)
models/                ← GGUF model files (gitignored)
chroma_db/             ← ChromaDB persistent store (gitignored, created by ingest)
support_agent.db       ← SQLite database (gitignored, auto-created on startup)
```

**Request flow:** `/chat` → OpenAI client → llama-server → if response contains `TOOL:` regex → parse & execute tool (async DB call) → second LLM call with result → final reply.

## Gotchas

- **ChromaDB path is absolute.** All three files (`ingest.py`, `retriever.py`, `embed.py`) resolve `chroma_db/` from `__file__` to project root. Do not change these paths independently.
- **Tool calling is prompt-based, not native function calling.** The LLM outputs `TOOL: name | param: value` strings, parsed by regex in `main.py:parse_tool_call()`. Phi-4-mini doesn't support native tool calling. If you swap models, you may need to change this.
- **SQLite path is absolute.** `db.py` computes `DATABASE_DIR` from `__file__` so `support_agent.db` always lands at project root regardless of CWD.
- **Tools are async.** `create_support_ticket`, `get_ticket`, `escalate_to_human` now take an `AsyncSession` and must be `await`ed.
- **Conversations are still in-memory for LLM context.** The Python dict (`conversations`) holds the full chat history sent to the LLM. Each user/assistant pair is also saved to `ConversationDB` in SQLite. The in-memory dict is separate from the DB — deleting from DB does not clear the in-memory LLM context.
- **No tests, no lint, no typecheck.** There is no test suite, no linter config, and no type checker configured.
- **No requirements.txt or pyproject.toml.** Dependencies installed via `pip` in the venv: `sqlalchemy`, `aiosqlite`, `python-multipart` (for file uploads), plus the existing packages.
- **To reset the database:** delete `support_agent.db` at project root. Tables auto-recreate on next startup.
- **To reset ChromaDB:** `Remove-Item -Recurse -Force chroma_db` then re-run ingest.

## API Endpoints

- `POST /chat` — Send a message. Saves user/assistant pair to `conversations` table.
- `GET /chat/{session_id}` — Retrieve all messages for a session.
- `DELETE /chat/{session_id}` — Delete conversation from DB + clear in-memory LLM context.
- `POST /documents` — Upload a .txt or .pdf file. Auto-ingests into ChromaDB.
- `GET /documents` — List all uploaded documents.
- `DELETE /documents/{id}` — Delete document from DB + disk + ChromaDB.
- `GET /` — Health/info.
- `GET /health` — Simple healthcheck.
- `GET /docs` — Swagger UI.
