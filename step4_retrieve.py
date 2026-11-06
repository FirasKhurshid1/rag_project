"""
STEP 4 -- RETRIEVE   (question -> most relevant chunks)

Job: embed the question with the SAME embedder, compare it to every chunk
vector (cosine similarity = dot product, since vectors are unit length) and
return the best matches.

Because chunks carry taxonomy metadata you can also FILTER, e.g.
    search("panels", doc="Sample_1", type_="table content")
"""
import json

import numpy as np

import config
from embedder import load_embedder


class Retriever:
    def __init__(self):
        d = config.INDEX_DIR
        if not (d / "chunks.json").exists():
            raise SystemExit("No index found. Run:  python main.py build")
        self.chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        self.vectors = np.load(d / "vectors.npy")
        self.embedder = load_embedder(json.loads((d / "embedder.json").read_text(encoding="utf-8")))

    def search(self, query: str, top_k: int = config.TOP_K, doc: str | None = None,
               type_: str | None = None) -> list[tuple[float, dict]]:
        scores = self.vectors @ self.embedder.encode([query])[0]
        order = np.argsort(-scores)
        hits = []
        for i in order:
            c = self.chunks[i]
            if doc and doc.lower() not in c["doc"].lower():
                continue
            if type_ and type_.lower() != c["type"].lower():
                continue
            hits.append((float(scores[i]), c))
            if len(hits) == top_k:
                break
        return hits
