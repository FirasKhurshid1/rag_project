"""
STEP 3 -- INDEX   (chunks -> vectors saved to disk)

Job: embed every chunk and save three files in index/:
    chunks.json     the chunk texts + taxonomy metadata
    vectors.npy     one vector per chunk (same order)
    embedder.json   which embedder was used (needed to embed questions later)

Run this once, or again whenever your documents change.
"""
import json

import numpy as np

import config
from embedder import get_embedder
from step2_chunk import build_all_chunks


def build_index(backend: str = config.EMBED_BACKEND) -> int:
    chunks = build_all_chunks()
    texts = [c["embed_text"] for c in chunks]

    embedder = get_embedder(backend).fit(texts)
    vectors = embedder.encode(texts)

    config.INDEX_DIR.mkdir(exist_ok=True)
    (config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    np.save(config.INDEX_DIR / "vectors.npy", vectors)
    (config.INDEX_DIR / "embedder.json").write_text(
        json.dumps({"backend": embedder.name, "state": embedder.state()}), encoding="utf-8")
    print(f"Indexed {len(chunks)} chunks from {len({c['doc'] for c in chunks})} files "
          f"using '{embedder.name}' embeddings -> {config.INDEX_DIR}/")
    return len(chunks)


if __name__ == "__main__":
    build_index()
