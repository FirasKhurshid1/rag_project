"""
STEP 1 -- EXTRACT   (file format -> neutral "blocks")

Job: open each .docx / .pdf and turn it into a simple, ordered list of
blocks, no matter what the original format was:

    title    the document title
    meta     subtitle / date lines under the title
    heading  a section heading
    para     a paragraph of body text
    bullet   one bullet-list item
    table    a table (list of rows)

This is the ONLY file that knows about Word and PDF internals.
Everything after this step works on blocks and does not care about the format.
"""
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

import config


@dataclass
class Block:
    kind: str            # title | meta | heading | para | bullet | table
    text: str = ""
    rows: list = None    # only used when kind == "table"


def _skip(text: str) -> bool:
    return not text or text.lower().startswith(config.SKIP_PREFIXES)


# ---------------------------------------------------------------- DOCX -----
def extract_docx(path: Path) -> list[Block]:
    doc = Document(path)
    blocks, seen_heading = [], False

    for el in doc.element.body.iterchildren():
        if el.tag == qn("w:p"):
            p = Paragraph(el, doc)
            text, style = p.text.strip(), p.style.name
            if _skip(text):
                continue
            if style.startswith("Heading"):
                if text.lower() == config.STOP_HEADING:
                    break                                  # taxonomy table starts here
                seen_heading = True
                blocks.append(Block("heading", text))
            elif not seen_heading:                         # lines above the 1st heading
                blocks.append(Block("title" if not blocks else "meta", text))
            elif "List" in style:
                blocks.append(Block("bullet", text))
            else:
                blocks.append(Block("para", text))

        elif el.tag == qn("w:tbl"):
            rows = [[c.text.strip() for c in r.cells] for r in Table(el, doc).rows]
            blocks.append(Block("table", rows=rows))
    return blocks


# ----------------------------------------------------------------- PDF -----
def _in_table(line, bboxes) -> bool:
    return any(b[1] - 1 <= line["top"] and line["bottom"] <= b[3] + 1 for b in bboxes)


def extract_pdf(path: Path) -> list[Block]:
    blocks, seen_heading, para_buf = [], False, []

    def flush():                       # join wrapped lines into one paragraph
        if para_buf:
            blocks.append(Block("para", " ".join(para_buf)))
            para_buf.clear()

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tables = page.find_tables()
            bboxes = [t.bbox for t in tables]

            # collect text lines (outside tables) and tables, ordered top->bottom
            items = [(l["top"], "line", l) for l in page.extract_text_lines()
                     if not _in_table(l, bboxes)]
            items += [(t.bbox[1], "table", t.extract()) for t in tables]
            items.sort(key=lambda x: x[0])

            for _, kind, payload in items:
                if kind == "table":
                    flush()
                    blocks.append(Block("table", rows=payload))
                    continue

                text = payload["text"].strip()
                char = payload["chars"][0]
                is_heading = round(char["size"]) >= 13 and "Bold" in char["fontname"]

                if _skip(text):
                    continue
                if is_heading:
                    flush()
                    if text.lower() == config.STOP_HEADING:
                        return blocks                      # taxonomy table starts here
                    if not blocks:
                        blocks.append(Block("title", text))
                    else:
                        seen_heading = True
                        blocks.append(Block("heading", text))
                elif not seen_heading:
                    blocks.append(Block("meta", text))
                elif text.startswith(("(cid:127)", "•")):
                    flush()
                    blocks.append(Block("bullet", text.replace("(cid:127)", "").lstrip("• ").strip()))
                else:
                    para_buf.append(text)
        flush()
    return blocks


# -------------------------------------------------------------- ENTRY ------
def extract_file(path: Path) -> list[Block]:
    if path.suffix.lower() == ".docx":
        return extract_docx(path)
    if path.suffix.lower() == ".pdf":
        return extract_pdf(path)
    raise ValueError(f"Unsupported file type: {path.name}")


if __name__ == "__main__":                     # python step1_extract.py
    for f in sorted(config.DATA_DIR.iterdir()):
        if f.suffix.lower() in (".docx", ".pdf"):
            print(f"\n=== {f.name}")
            for b in extract_file(f):
                print(f"  {b.kind:8} {b.text or b.rows}")
