import json
import re

from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

from src import config

MODES = ["vector", "keyword", "hybrid", "hybrid_rerank"]

STOPWORDS = {
    "the", "a", "an", "of", "is", "are", "does", "do", "how", "what", "which", "who",
    "in", "on", "to", "and", "its", "it", "from", "for", "was", "by", "with", "that",
    "this", "be", "as", "at", "or",
}


def tokenize(text):
    """Lowercase, split into words, drop very common words."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class Retriever:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
        self.db = Chroma(
            persist_directory=str(config.DB_DIR),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        # Plain copy of the chunks, used for keyword search
        self.chunks = json.loads(config.CHUNKS_FILE.read_text())
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.reranker = CrossEncoder(config.RERANK_MODEL)

    @staticmethod
    def _pack(chunk, score):
        return {**chunk, "score": float(score)}

    def vector_search(self, query, k=20, doc=None):
        """Search by meaning. Optionally only inside one PDF."""
        flt = {"doc": doc} if doc else None
        hits = self.db.similarity_search_with_score(query, k=k, filter=flt)
        return [self._pack(self.by_id[d.metadata["chunk_id"]], 1 - dist) for d, dist in hits]

    def keyword_search(self, query, k=20, doc=None):
        """Search by exact words (BM25)."""
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in order:
            if scores[i] <= 0:
                break
            if doc and self.chunks[i]["doc"] != doc:
                continue
            results.append(self._pack(self.chunks[i], scores[i]))
            if len(results) == k:
                break
        return results

    def hybrid_search(self, query, k=20, doc=None):
        """Run both searches and merge them with Reciprocal Rank Fusion."""
        fused = {}
        for hits in (self.vector_search(query, k, doc), self.keyword_search(query, k, doc)):
            for rank, hit in enumerate(hits, start=1):
                fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (60 + rank)
        ordered = sorted(fused, key=fused.get, reverse=True)
        return [self._pack(self.by_id[i], fused[i]) for i in ordered]

    def rerank(self, query, candidates, top_k=5):
        """A careful second model scores each candidate against the question."""
        if not candidates:
            return []
        scores = self.reranker.predict([(query, c["text"]) for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [self._pack(c, s) for c, s in ranked[:top_k]]

    def search(self, query, mode="hybrid_rerank", top_k=5, fetch_k=20, doc=None):
        if mode == "vector":
            return self.vector_search(query, top_k, doc)
        if mode == "keyword":
            return self.keyword_search(query, top_k, doc)
        if mode == "hybrid":
            return self.hybrid_search(query, fetch_k, doc)[:top_k]
        if mode == "hybrid_rerank":
            return self.rerank(query, self.hybrid_search(query, fetch_k, doc), top_k)
        raise ValueError(f"Unknown mode: {mode}")


def main():
    retriever = Retriever()
    print("Ready. Type a question, or 'quit'.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue
        for mode in MODES:
            print(f"\n[{mode}]")
            for i, hit in enumerate(retriever.search(question, mode=mode, top_k=3), 1):
                preview = hit["text"][:90].replace("\n", " ")
                print(f"  {i}. {hit['doc']} p.{hit['page']}  ({hit['score']:.3f})  {preview}...")
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()