"""FastAPI backend for the React resume chatbot UI."""

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.ingest import ensure_index_exists
from src.rag import answer


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="User question about the resume")
    k: int = Field(default=4, ge=1, le=8, description="Top-k chunks to retrieve")


class ChatResponse(BaseModel):
    reply: str


app = FastAPI(title="Resume Chat API", version="1.0.0")

# Keep CORS open in local dev; Vite proxy also works without CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_ingest() -> None:
    # Render free tier doesn't provide shell/one-off jobs; auto-prepare index on boot.
    auto = os.getenv("AUTO_INGEST_ON_STARTUP", "true").lower()
    if auto not in {"1", "true", "yes", "on"}:
        return
    try:
        count = ensure_index_exists()
        print(f"[startup] resume index ready with {count} chunks")
    except Exception as exc:
        # Keep startup alive; /api/chat will return error details if index/model unavailable.
        print(f"[startup] index preparation failed: {exc}")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        reply = answer(payload.message.strip(), k=payload.k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(reply=reply)

