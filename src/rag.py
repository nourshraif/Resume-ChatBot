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



def _wants_project_list(question: str) -> bool:
    q = question.lower()
    if "project" not in q:
        return False
    return any(
        w in q
        for w in (
            "what",
            "which",
            "list",
            "all",
            "every",
            "name",
            "tell",
            "show",
            "did she",
            "did he",
            "did they",
            "has she",
            "has he",
            "work on",
            "built",
        )
    )


def _wants_broad_coverage(question: str) -> bool:
    """Detect questions that need most/all CV chunks (e.g. list every skill)."""
    if _wants_project_list(question):
        return True
    q = question.lower()
    if "skill" in q and any(w in q for w in ("what", "which", "list", "all")):
        return True
    return False


def _chunk_id_sort_key(chunk_id: str) -> int:
    suffix = chunk_id[1:] if chunk_id.startswith("c") else chunk_id
    return int(suffix) if suffix.isdigit() else 0


def retrieve_all_chunks() -> tuple[list[str], list[str]]:
    """Return every indexed chunk in original document order."""
    col = _collection()
    data = col.get(include=["documents", "metadatas"])
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    if not ids or not docs:
        return [], []

    rows = sorted(zip(ids, docs, metas), key=lambda row: _chunk_id_sort_key(row[0]))
    doc_list = [doc for _, doc, _ in rows]
    labels: list[str] = []
    for _, _, meta in rows:
        if not isinstance(meta, dict):
            meta = {}
        labels.append(meta.get("source", "resume.txt"))
    sources = list(dict.fromkeys(labels))
    return doc_list, sources


def retrieve_project_chunks() -> tuple[list[str], list[str]]:
    """Return chunks from the Selected Projects section onward."""
    doc_list, sources = retrieve_all_chunks()
    start = next((i for i, doc in enumerate(doc_list) if "Selected Projects" in doc), 0)
    return doc_list[start:], sources


def retrieve(question: str, k: int = 4) -> tuple[list[str], list[str]]:
    """Return retrieved chunk texts and deduplicated source labels from chunk metadata."""
    if _wants_project_list(question):
        return retrieve_project_chunks()

    col = _collection()
    total = col.count()
    if _wants_broad_coverage(question):
        return retrieve_all_chunks()

    n = min(max(k, 1), total)
    res = col.query(query_texts=[question], n_results=n)
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
    "from CV and projects. When asked to list items (projects, skills, etc.), include "
    "every matching item from the context — do not omit entries that are present. "
    "If the answer is not in context, respond exactly: "
    "'That's not in my resume yet, but I'm a fast learner. Want to connect and discuss?' "
    "Never invent dates, companies, or metrics."
)


def _listing_instruction(question: str) -> str:
    if _wants_project_list(question):
        return (
            "\n\nINSTRUCTION: List every distinct project from the context. "
            "Use one bullet per project in the form: **Project Name** — short description. "
            "Do not merge projects, skip any, or invent details."
        )
    if _wants_broad_coverage(question):
        return (
            "\n\nINSTRUCTION: Include every matching item from the context. "
            "Do not omit entries that are present."
        )
    return ""


def generate_answer(
    question: str,
    context_chunks: list[str],
    max_tokens: int = 768,
) -> str:
    """Call the LLM given pre-retrieved chunks (avoids double embedding/search)."""
    context = "\n\n---\n\n".join(context_chunks)
    system = GUARDRAIL_SYSTEM
    user = (
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        f"{_listing_instruction(question)}"
    )

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
    k: int = 8,
    max_tokens: int = 768,
) -> dict[str, Any]:
    if _wants_project_list(question):
        max_tokens = max(max_tokens, 1024)
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
