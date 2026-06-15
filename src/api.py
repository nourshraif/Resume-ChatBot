"""FastAPI backend for the React resume chatbot UI."""

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.ingest import ensure_index_fresh
from src.rag import answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "chroma_db"

limiter = Limiter(key_func=get_remote_address)


def _index_ready() -> bool:
    if not CHROMA_DIR.is_dir():
        return False
    try:
        return any(CHROMA_DIR.iterdir())
    except OSError:
        return False


def _parse_origins(raw: str) -> list[str]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return parts if parts else ["*"]


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="User question about the resume",
    )
    k: int = Field(default=8, ge=1, le=12, description="Top-k chunks to retrieve")


class ChatResponse(BaseModel):
    reply: str
    response: str = Field(description="Same body as reply (alias for newer clients).")
    sources: list[str] = Field(default_factory=list)
    sources_used: int = Field(
        default=0,
        description="Number of CV chunks retrieved for this answer.",
    )
    answer: Optional[str] = Field(
        default=None,
        description="Same text as reply; legacy clients.",
    )


app = FastAPI(title="Nour CV ChatBot", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(os.getenv("ALLOWED_ORIGINS", "*")),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.on_event("startup")
def startup_ingest() -> None:
    auto = os.getenv("AUTO_INGEST_ON_STARTUP", "true").lower()
    if auto not in {"1", "true", "yes", "on"}:
        return
    try:
        count = ensure_index_fresh()
        logger.info("resume index ready with %s chunks", count)
    except Exception as exc:
        logger.warning("index preparation failed: %s", exc)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "index_ready": _index_ready()}


@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    try:
        logger.info("Chat query: %s", payload.message[:80])
        result = answer(
            payload.message.strip(),
            k=payload.k,
        )
    except Exception as exc:
        logger.exception("Chat error")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate response.",
        ) from exc

    text = result["answer"]
    sources = result.get("sources") or []
    num_chunks = int(result.get("num_chunks") or len(sources))
    return ChatResponse(
        reply=text,
        response=text,
        sources=sources,
        sources_used=num_chunks,
        answer=text,
    )
