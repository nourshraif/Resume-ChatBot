"""
Build the vector index from data/resume.txt (embed + store in Chroma).

Run from project root with venv activated:
    python -m src.ingest

Re-run whenever you change resume.txt or chunking settings.
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.chunking import chunk_text
from src.io_util import configure_utf8_stdout

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "resume.txt"
CHROMA_DIR = ROOT / "chroma_db"


def main() -> None:
    configure_utf8_stdout()
    if not DATA_PATH.is_file():
        raise SystemExit(f"Missing {DATA_PATH}")

    raw = DATA_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(raw)
    if not chunks:
        raise SystemExit("No chunks produced (is resume.txt empty?)")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    ef = embedding_functions.DefaultEmbeddingFunction()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = "resume"
    # Chroma no longer allows delete(where={}); replace index by dropping the collection.
    for col in client.list_collections():
        if col.name == name:
            client.delete_collection(name)
            break
    collection = client.create_collection(name=name, embedding_function=ef)

    ids = [f"c{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks)

    print(f"Stored {len(chunks)} chunks in {CHROMA_DIR}")
    print("First chunk preview:", chunks[0][:160].replace("\n", " "), "...")


if __name__ == "__main__":
    main()
