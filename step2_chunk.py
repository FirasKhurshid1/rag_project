"""
STEP 2 -- CHUNK   (blocks -> searchable chunks with taxonomy metadata)

Job: group blocks into chunks. This is where your TAXONOMY is used:
each chunk is tagged with the same TYPE / SUB-TYPE vocabulary as Taxonomy.xlsx.

    taxonomy TYPE / SUB-TYPE         chunk produced
    text content / main heading      title + subtitle + date
    text content / body             one chunk per section's paragraphs
    list / bullet list               one chunk per section's bullets
    table content / rows and columns one chunk per table (rows -> sentences)
    visual content / bar graph|image text from visuals.json

Charts and pictures contain no extractable text, so their descriptions are
written by hand in visuals.json (you could automate it later with Claude vision).
"""
import json
import re
from pathlib import Path

import config
from step1_extract import Block, extract_file


def split_text(text: str, max_chars: int) -> list[str]:
    """Split long text at sentence ends so no piece exceeds max_chars."""
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(cur) + len(sent) + 1 > max_chars:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    return parts + [cur]


def table_to_text(rows: list[list[str]]) -> str:
    """Turn a table into self-describing sentences, e.g.
    'Row 1: Site: Site A; Panels planned: 120; Status: Complete.'
    Keeping the column name next to each value lets search match
    questions like 'how many panels at Site B?'."""
    header, body = rows[0], rows[1:]
    lines = []
    for i, r in enumerate(body, 1):
        cells = "; ".join(f"{h}: {v}" for h, v in zip(header, r))
        lines.append(f"Row {i}: {cells}.")
    return "\n".join(lines)


def blocks_to_chunks(file: Path, blocks: list[Block], visuals: list[dict]) -> list[dict]:
    source = f"{file.suffix.lstrip('.')} file"                 # 'pdf file' / 'docx file'
    title = next((b.text for b in blocks if b.kind == "title"), file.stem)
    chunks = []

    def add(type_, sub_type, section, text):
        for piece in split_text(text, config.MAX_CHUNK_CHARS):
            chunks.append({
                "id": f"{file.stem}::{len(chunks) + 1:02d}",
                "doc": file.name, "doc_title": title, "source": source,
                "type": type_, "sub_type": sub_type, "location": section,
                "text": piece,
                # text actually embedded: chunk + a context header so a lone
                # sentence still "knows" which document/section it came from
                "embed_text": f"Document: {title} | Section: {section} | {sub_type}\n{piece}",
            })

    # 1) title + meta lines -> main heading chunk
    head = [b.text for b in blocks if b.kind in ("title", "meta")]
    add("text content", "main heading", "page 1", ". ".join(head))

    # 2) walk the sections
    section, paras, bullets = None, [], []

    def close_section():
        if paras:
            add("text content", "body", section, " ".join(paras))
        if bullets:
            add("list", "bullet list", section, "\n".join(f"- {b}" for b in bullets))
        paras.clear(); bullets.clear()

    for b in blocks:
        if b.kind == "heading":
            close_section(); section = b.text
        elif b.kind == "para":
            paras.append(b.text)
        elif b.kind == "bullet":
            bullets.append(b.text)
        elif b.kind == "table":
            add("table content", "rows and columns", section, table_to_text(b.rows))
    close_section()

    # 3) charts / images described in visuals.json
    for v in visuals:
        if v["doc"] == file.name:
            add("visual content", v["sub_type"], v["section"], v["text"])
    return chunks


def build_all_chunks() -> list[dict]:
    visuals = json.loads(config.VISUALS_FILE.read_text(encoding="utf-8"))
    chunks = []
    for f in sorted(config.DATA_DIR.iterdir()):
        if f.suffix.lower() in (".docx", ".pdf"):
            chunks += blocks_to_chunks(f, extract_file(f), visuals)
    return chunks


if __name__ == "__main__":                     # python step2_chunk.py
    for c in build_all_chunks():
        print(f"{c['id']:40} {c['type']:15} {c['sub_type']:17} | {c['location']}")
