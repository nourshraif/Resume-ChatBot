"""Split resume text into chunks for embedding and search."""


def _sliding_windows(segment: str, max_chars: int, overlap: int) -> list[str]:
    """Cover a long single line (or any string) with overlapping fixed-size windows."""
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    segment = segment.strip()
    if not segment:
        return []
    out: list[str] = []
    start = 0
    n = len(segment)
    while start < n:
        end = min(start + max_chars, n)
        out.append(segment[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return out


def chunk_text(text: str, max_chars: int = 300, overlap: int = 60) -> list[str]:
    """
    Split into chunks in two modes:

    - **Default behavior**: pack whole lines (split on ``\\n``) into chunks until
      adding the next line would exceed ``max_chars``. That usually yields **more
      chunks** than a plain character slide on a bullet-style CV, and keeps
      bullets / headings intact when they fit in one window.
    - **Long lines**: if a single line is longer than ``max_chars``, only that
      line is split using a sliding window with ``overlap``.

    For a ~2.4k character CV, ``max_chars=300`` yields about **10 chunks**; use **450**
    if you prefer fewer, larger pieces (often ~6).
    """
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    # Word "Symbol" bullets (private use) break Windows cp1252 terminals and embeddings noise
    text = text.replace("\uf0b7", "-").replace("\u2022", "-").replace("\u2219", "-")
    if not text:
        return []

    lines = text.split("\n")
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0  # total chars if we join buf with newlines

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n".join(buf))
            buf = []
            buf_len = 0

    for line in lines:
        line_len = len(line)
        sep = 1 if buf else 0
        if line_len > max_chars:
            flush()
            chunks.extend(_sliding_windows(line, max_chars, overlap))
            continue

        if buf_len + sep + line_len > max_chars:
            flush()
            buf = [line]
            buf_len = line_len
        else:
            buf.append(line)
            buf_len += sep + line_len

    flush()
    return chunks
