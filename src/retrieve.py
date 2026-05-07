"""
Print top matching resume chunks for a question (retrieval only, no LLM).

  python -m src.retrieve
  python -m src.retrieve "What ML projects did they do?"
"""

import argparse

from src.io_util import configure_utf8_stdout
from src.rag import retrieve


def main() -> None:
    configure_utf8_stdout()
    p = argparse.ArgumentParser(description="Query the resume vector index")
    p.add_argument(
        "question",
        nargs="?",
        default="What are this person's technical skills?",
        help="Question to embed and search with",
    )
    p.add_argument("-k", type=int, default=4, help="Number of chunks to return")
    args = p.parse_args()

    chunks = retrieve(args.question, k=args.k)
    if not chunks:
        print("No results (empty index or ingest not run).")
        return

    print("Question:", args.question)
    print()
    for i, ch in enumerate(chunks, start=1):
        preview = ch.strip().replace("\n", " ")
        if len(preview) > 400:
            preview = preview[:400] + "..."
        print(f"--- Chunk {i} ---")
        print(preview)
        print()


if __name__ == "__main__":
    main()
