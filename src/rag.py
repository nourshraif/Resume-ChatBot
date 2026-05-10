"""Retrieve chunks from Chroma, then answer with Hugging Face Inference API."""

import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "chroma_db"

load_dotenv(ROOT / ".env")


def _collection():
    if not CHROMA_DIR.is_dir():
        raise FileNotFoundError(
            f"No index at {CHROMA_DIR}. Run: python -m src.ingest"
        )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_collection(name="resume", embedding_function=ef)


def retrieve(question: str, k: int = 4) -> tuple[list[str], list[str]]:
    """Return retrieved chunk texts and deduplicated source labels from chunk metadata."""
    col = _collection()
    res = col.query(query_texts=[question], n_results=k)
    docs = res.get("documents") or []
    metas_raw = res.get("metadatas") or []
    if not docs or not docs[0]:
        return [], []
    doc_list = list(docs[0])
    meta_row = list(metas_raw[0]) if metas_raw and metas_raw[0] else []
    labels: list[str] = []
    for i, _doc in enumerate(doc_list):
        m = meta_row[i] if i < len(meta_row) else None
        if not isinstance(m, dict):
            m = {}
        labels.append(m.get("source", "resume.txt"))
    sources = list(dict.fromkeys(labels))
    return doc_list, sources


def _hf_client() -> InferenceClient:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in .env — "
            "https://huggingface.co/settings/tokens"
        )
    # Default must be a model your HF account can route (see inference/models on HF).
    model = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    return InferenceClient(model=model, token=token)


GUARDRAIL_SYSTEM = (
    "You are Nour Shraif's resume assistant. Only answer using the provided context "
    "from CV and projects. If the answer is not in context, respond exactly: "
    "'That's not in my resume yet, but I'm a fast learner. Want to connect and discuss?' "
    "Never invent dates, companies, or metrics."
)


def generate_answer(
    question: str,
    context_chunks: list[str],
    max_tokens: int = 768,
) -> str:
    """Call the LLM given pre-retrieved chunks (avoids double embedding/search)."""
    context = "\n\n---\n\n".join(context_chunks)
    system = GUARDRAIL_SYSTEM
    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    client = _hf_client()
    temperature = float(os.getenv("RAG_TEMPERATURE", "0.38"))
    try:
        out = client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except BadRequestError as e:
        mid = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        raise RuntimeError(
            f"Hugging Face returned 400 (often: model not available on your enabled providers).\n"
            f"Model in use: {mid}\n"
            f"Set HF_LLM_MODEL in .env to another chat model from "
            f"https://huggingface.co/inference/models\n"
            f"Original error: {e}"
        ) from e
    msg = out.choices[0].message
    return (msg.content or "").strip()


def answer(
    question: str,
    k: int = 6,
    max_tokens: int = 768,
) -> dict[str, Any]:
    context_chunks, sources = retrieve(question, k=k)
    if not context_chunks:
        return {
            "answer": (
                "No chunks retrieved. Run ingest and check that chroma_db exists "
                "and resume.txt is not empty."
            ),
            "sources": [],
            "num_chunks": 0,
        }
    llm_response = generate_answer(
        question,
        context_chunks,
        max_tokens=max_tokens,
    )
    return {
        "answer": llm_response,
        "sources": sources,
        "num_chunks": len(context_chunks),
    }
