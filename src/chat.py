"""Interactive resume Q&A (RAG + Hugging Face). Run: python -m src.chat"""

from src.io_util import configure_utf8_stdout
from src.rag import answer


def main() -> None:
    configure_utf8_stdout()
    print("Resume chat — questions are answered from your indexed CV + HF model.")
    print("Empty line to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        try:
            out = answer(q)
        except Exception as e:
            out = f"Error: {e}"
        print(f"Bot: {out}\n")


if __name__ == "__main__":
    main()
