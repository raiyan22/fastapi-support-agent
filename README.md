# Local AI Customer Support Agent

A fully local ($0 cost) AI customer support agent powered by FastAPI, llama.cpp, and ChromaDB RAG. No cloud APIs. All inference runs on your machine.

## Features

- **Chat UI** — Dark-themed web interface at `http://localhost:8000`
- **Local LLM** — Phi-4-mini running via llama.cpp (no API keys, no internet required)
- **Tool Calling** — Agent can create support tickets, search knowledge base, escalate to humans, check ticket status
- **RAG** — Upload .txt/.pdf documents, auto-indexed into ChromaDB, searchable by the agent
- **Persistent Storage** — Tickets, conversations, and documents saved in SQLite
- **Customer History** — Past conversations loaded as context for returning customers
- **Logging** — All requests, tool calls, and errors logged with timestamps

## Quick Start

### Prerequisites

1. **Install llama.cpp** (Windows):
   ```powershell
   winget install --id ggml.llamacpp --exact
   ```

2. **Download the model** (~2.5 GB):
   ```powershell
   pip install huggingface_hub
   huggingface-cli download unsloth/Phi-4-mini-instruct-GGUF Phi-4-mini-instruct-Q4_K_M.gguf --local-dir models
   ```

3. **Create virtual environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install fastapi uvicorn[standard] pydantic pydantic-settings python-dotenv openai sqlalchemy aiosqlite python-multipart langchain langchain-community langchain-huggingface langchain-chroma chromadb sentence-transformers pypdf
   ```

4. **Create `.env`** (copy from `.env.example`):
   ```
   LLM_BASE_URL=http://localhost:8080/v1
   LLM_MODEL=phi-4-mini
   LLM_API_KEY=sk-no-key-required
   ```

### Running

Open **three terminals**:

```powershell
# Terminal 1 — Start the LLM server
llama-server -m models/Phi-4-mini-instruct-Q4_K_M.gguf --port 8080 --host 0.0.0.0 -c 8192

# Terminal 2 — Ingest knowledge base (only needed once, or after changing data/)
.venv\Scripts\activate
python -m app.rag.ingest

# Terminal 3 — Start the web server
.venv\Scripts\activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access

| URL | What |
|-----|------|
| `http://localhost:8000` | Chat UI |
| `http://localhost:8000/docs` | Swagger API docs |
| `http://localhost:8000/health` | Health check |

## Project Structure

```
fastapiapp/
├── app/
│   ├── main.py              ← FastAPI app, /chat endpoint, tool parsing
│   ├── db.py                ← Async SQLAlchemy + SQLite
│   ├── core/config.py       ← Settings from .env
│   ├── models/
│   │   ├── schemas.py       ← TicketDB, ConversationDB, Pydantic schemas
│   │   └── documents.py     ← DocumentDB model
│   ├── agent/tools.py       ← Ticket, escalation, RAG search tools
│   ├── routers/documents.py ← Upload/List/Delete document endpoints
│   └── rag/
│       ├── ingest.py        ← Batch ingestion from data/ to ChromaDB
│       ├── retriever.py     ← Similarity search wrapper
│       └── embed.py         ← Single-file embed/delete for upload API
├── static/index.html        ← Chat web UI
├── data/knowledge/          ← Place .txt/.pdf files here
├── models/                  ← GGUF model files (gitignored)
├── chroma_db/               ← Vector store (gitignored)
├── support_agent.db         ← SQLite database (gitignored)
└── .env                     ← Config (gitignored)
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a message. Body: `{"message": "...", "conversation_id": "...", "customer_id": "..."}` |
| `GET` | `/chat/{session_id}` | Get conversation history |
| `DELETE` | `/chat/{session_id}` | Delete conversation |
| `POST` | `/documents` | Upload a file (multipart form) |
| `GET` | `/documents` | List uploaded documents |
| `DELETE` | `/documents/{id}` | Delete a document |
| `GET` | `/health` | Health check |

## How It Works

1. User sends a message via the chat UI or API
2. The message goes to the local Phi-4-mini model via llama.cpp
3. If the model outputs a `TOOL: ...` line, it's parsed and executed (create ticket, search docs, etc.)
4. The tool result is sent back to the model for a final response
5. The user/assistant pair is saved to SQLite with the customer ID
6. For returning customers, past conversations are loaded as context

## Resetting

- **Database** — delete `support_agent.db`, tables auto-recreate on next startup
- **ChromaDB** — delete the `chroma_db/` folder, re-run `python -m app.rag.ingest`
- **Both** — delete both files above

## Tech Stack

| Component | Choice |
|-----------|--------|
| Backend | FastAPI (async) |
| LLM | Phi-4-mini via llama.cpp |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB (persistent) |
| Database | SQLite via SQLAlchemy + aiosqlite |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Frontend | Vanilla HTML/CSS/JS (single file) |
