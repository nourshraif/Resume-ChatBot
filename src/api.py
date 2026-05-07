"""FastAPI backend for the React resume chatbot UI."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

