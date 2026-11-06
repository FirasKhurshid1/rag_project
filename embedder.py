"""
EMBEDDER  --  text -> numbers (vectors)

An "embedding" turns a piece of text into a list of numbers so that texts with
similar MEANING end up with similar numbers. Search = find the closest vectors.

Two interchangeable backends (same interface):
  * SbertEmbedder  semantic, understands synonyms ("fix" ~ "repair"). Needs
                   `pip install sentence-transformers` (downloads a small model once).
  * TfidfEmbedder  keyword-based, pure numpy, zero downloads. Works offline,
                   but only matches shared words, not meaning.
"auto" picks SBERT if installed, otherwise TF-IDF.
"""
import math
import re

import numpy as np

import config

STOPWORDS = set("a an and are as at be by for from has have how in is it of on or the to was were what when which who with".split())


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in STOPWORDS]


class TfidfEmbedder:
    name = "tfidf"

    def fit(self, texts):
        docs = [set(_tokens(t)) for t in texts]
        vocab = sorted(set().union(*docs))
        self.index = {w: i for i, w in enumerate(vocab)}
        n = len(texts)
        self.idf = [math.log((1 + n) / (1 + sum(w in d for d in docs))) + 1 for w in vocab]
        return self

    def encode(self, texts) -> np.ndarray:
        out = np.zeros((len(texts), len(self.index)), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in _tokens(t):
                if w in self.index:
                    out[r, self.index[w]] += self.idf[self.index[w]]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)       # unit length -> dot product = cosine

    def state(self) -> dict:
        return {"index": self.index, "idf": self.idf}

    @classmethod
    def from_state(cls, s):
        e = cls(); e.index, e.idf = s["index"], s["idf"]; return e


class SbertEmbedder:
    name = "sbert"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(config.SBERT_MODEL)

    def fit(self, texts):          # pretrained: nothing to learn
        return self

    def encode(self, texts) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)

    def state(self) -> dict:
        return {}


def get_embedder(backend: str = config.EMBED_BACKEND):
    if backend in ("auto", "sbert"):
        try:
            return SbertEmbedder()
        except ImportError:
            if backend == "sbert":
                raise
            print("[embedder] sentence-transformers not installed -> using TF-IDF fallback")
    return TfidfEmbedder()


def load_embedder(info: dict):
    return TfidfEmbedder.from_state(info["state"]) if info["backend"] == "tfidf" else SbertEmbedder()
