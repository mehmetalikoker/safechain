from __future__ import annotations
import json
import math
import os
from typing import Any, List, Optional, Tuple

from safechain.schema import Document


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class InMemoryVectorStore:
    """Saf Python cosine similarity ile in-memory vektör deposu."""

    def __init__(self, embedding_fn: Any) -> None:
        self.embedding_fn = embedding_fn
        self._documents: List[Document] = []
        self._embeddings: List[List[float]] = []

    # ------------------------------------------------------------------
    # Yükleme
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> "InMemoryVectorStore":
        texts = [d.page_content for d in documents]
        embeddings = self.embedding_fn.embed_documents(texts)
        self._documents.extend(documents)
        self._embeddings.extend(embeddings)
        return self

    @classmethod
    def from_documents(
        cls, documents: List[Document], embedding_fn: Any
    ) -> "InMemoryVectorStore":
        store = cls(embedding_fn)
        store.add_documents(documents)
        return store

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding_fn: Any,
        metadatas: Optional[List[dict]] = None,
    ) -> "InMemoryVectorStore":
        metas = metadatas or [{} for _ in texts]
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metas)]
        return cls.from_documents(docs, embedding_fn)

    # ------------------------------------------------------------------
    # Arama
    # ------------------------------------------------------------------

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k)]

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        q_emb = self.embedding_fn.embed_query(query)
        scored = [
            (doc, _cosine(q_emb, emb))
            for doc, emb in zip(self._documents, self._embeddings)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def as_retriever(self, k: int = 4) -> "VectorStoreRetriever":
        return VectorStoreRetriever(vectorstore=self, k=k)

    # ------------------------------------------------------------------
    # Kaydet / yükle (JSON, stdlib only)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        data = {
            "documents": [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in self._documents
            ],
            "embeddings": self._embeddings,
        }
        with open(os.path.join(path, "store.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, embedding_fn: Any) -> "InMemoryVectorStore":
        store = cls(embedding_fn)
        with open(os.path.join(path, "store.json"), encoding="utf-8") as f:
            data = json.load(f)
        store._documents = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in data["documents"]
        ]
        store._embeddings = data["embeddings"]
        return store


class VectorStoreRetriever:
    def __init__(self, vectorstore: InMemoryVectorStore, k: int = 4) -> None:
        self.vectorstore = vectorstore
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=self.k)

    def __call__(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)
