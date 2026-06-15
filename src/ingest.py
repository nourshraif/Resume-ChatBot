"""
Build the vector index from data/resume.txt (embed + store in Chroma).

Run from project root with venv activated:
    python -m src.ingest

Re-run whenever you change resume.txt or chunking settings.
"""

import hashlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.chunking import chunk_text
from src.io_util import configure_utf8_stdout

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "resume.txt"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "resume"
FINGERPRINT_PATH = CHROMA_DIR / ".resume_fingerprint"


def _resume_fingerprint() -> str:
    return hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()


def build_index(force_rebuild: bool = True) -> int:
    """
    Build the Chroma index from resume.txt.

    Returns number of stored chunks.
    """
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    raw = DATA_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(raw)
    if not chunks:
        raise ValueError("No chunks produced (is resume.txt empty?)")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    ef = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing and force_rebuild:
        client.delete_collection(COLLECTION_NAME)
    if COLLECTION_NAME in existing and not force_rebuild:
        return client.get_collection(name=COLLECTION_NAME).count()

    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)
    ids = [f"c{i}" for i in range(len(chunks))]
    source_label = DATA_PATH.name
    metadatas = [{"source": source_label} for _ in chunks]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    FINGERPRINT_PATH.write_text(_resume_fingerprint(), encoding="utf-8")
    return len(chunks)


def ensure_index_exists() -> int:
    """
    Ensure index exists for hosting environments without shell access.

    If no collection is present, build it. Otherwise return existing count.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        return client.get_collection(name=COLLECTION_NAME).count()
    return build_index(force_rebuild=True)


def ensure_index_fresh() -> int:
    """
    Build or rebuild the index when resume.txt changes.

    Used on API startup so deployed hosts pick up CV edits after redeploy.
    """
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    current_fp = _resume_fingerprint()
    stored_fp = (
        FINGERPRINT_PATH.read_text(encoding="utf-8").strip()
        if FINGERPRINT_PATH.is_file()
        else ""
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing and stored_fp == current_fp:
        return client.get_collection(name=COLLECTION_NAME).count()

    return build_index(force_rebuild=True)


def main() -> None:
    configure_utf8_stdout()
    try:
        count = build_index(force_rebuild=True)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Stored {count} chunks in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
