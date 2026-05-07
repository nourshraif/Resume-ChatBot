# Nour CV ChatBot (RAG)

A personalized resume chatbot built with:
- **FastAPI** backend (`/api/chat`)
- **React + Vite** frontend (`frontend/`)
- **ChromaDB** for vector storage/retrieval
- **Hugging Face Inference API** for final answer generation

The app answers questions about the resume using a RAG pipeline:
1. chunk resume text
2. embed + store in Chroma
3. retrieve top relevant chunks per question
4. generate an answer grounded in retrieved context

---

## Project Structure

```text
Nour_CV_ChatBot/
├─ data/
│  └─ resume.txt              # Your source-of-truth CV text
├─ src/
│  ├─ chunking.py             # Chunking logic
│  ├─ ingest.py               # Build/rebuild vector index
│  ├─ rag.py                  # Retrieve + generate answer
│  ├─ api.py                  # FastAPI backend (/api/chat)
│  └─ io_util.py              # Console UTF-8 helpers
├─ chroma_db/                 # Generated vector DB (after ingest)
├─ frontend/                  # React UI (editorial design)
├─ .env.example
├─ requirements.txt
└─ README.md
```

---

## Requirements

- Python **3.9+**
- Node.js **18+**
- npm

---

## 1) Python Setup

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2) Configure Environment

Create `.env` in project root (copy from `.env.example`) and set:

```env
HF_TOKEN=hf_your_token_here
# Optional model override:
# HF_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Get token from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 3) Add/Update Resume Content

Edit:

`data/resume.txt`

---

## 4) Build the Vector Index (IMPORTANT)

Run every time `resume.txt` or chunking logic changes:

```powershell
python -m src.ingest
```

This creates/refreshes `chroma_db/`.

---

## 5) Run Backend API

In terminal 1 (project root, venv active):

```powershell
uvicorn src.api:app --reload --port 8000
```

Health check:
- [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 6) Run React Frontend

In terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open:
- [http://localhost:5173](http://localhost:5173)

`frontend/vite.config.js` proxies `/api/*` to `http://127.0.0.1:8000`.

---

## API Contract

### `POST /api/chat`

Request:

```json
{
  "message": "What machine learning projects are listed?",
  "k": 4
}
```

Response:

```json
{
  "reply": "..."
}
```

---

## Common Issues

- **Blank React page**
  - Open browser console and check first red error.
  - Confirm frontend is running on `http://localhost:5173`.

- **Chat fails / backend unreachable**
  - Ensure FastAPI is running on port `8000`.
  - Check health endpoint `/api/health`.

- **No retrieval results**
  - Run `python -m src.ingest` again.
  - Verify `data/resume.txt` is not empty.

- **Hugging Face 400 model/provider error**
  - Change `HF_LLM_MODEL` in `.env` to a model your account can use.

---

## Dev Notes

- `.env` is git-ignored.
- `chroma_db/` is generated and git-ignored.
- Frontend is React-only UI; Streamlit path was removed.

