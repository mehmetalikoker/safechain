from __future__ import annotations
import json
import math
import os
from typing import Any, List, Optional, Tuple

from safechain.schema import Document


def _cosine(a: List[float], b: List[float]) -> float:
    """İki vektör arasındaki kosinüs benzerliğini hesaplar.

    Sıfır vektörlerden birinin büyüklüğü sıfırsa 0.0 döner.

    Args:
        a: Birinci float vektörü.
        b: İkinci float vektörü (``a`` ile aynı boyutta olmalı).

    Returns:
        [-1.0, 1.0] aralığında kosinüs benzerlik skoru.
    """
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
        """Belgeleri gömer ve depoya ekler.

        Args:
            documents: Eklenecek Document nesnelerinin listesi.

        Returns:
            Kendisini (zincirleme kullanım için).
        """
        texts = [d.page_content for d in documents]
        embeddings = self.embedding_fn.embed_documents(texts)
        self._documents.extend(documents)
        self._embeddings.extend(embeddings)
        return self

    @classmethod
    def from_documents(
        cls, documents: List[Document], embedding_fn: Any
    ) -> "InMemoryVectorStore":
        """Document listesinden vektör deposu oluşturur.

        Args:
            documents: Depoya eklenecek Document nesneleri.
            embedding_fn: ``embed_documents`` metoduna sahip embedding nesnesi.

        Returns:
            Belgelerin yüklendiği yeni InMemoryVectorStore örneği.
        """
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
        """Ham metin listesinden vektör deposu oluşturur.

        Args:
            texts: Depoya eklenecek metin dizeleri.
            embedding_fn: ``embed_documents`` metoduna sahip embedding nesnesi.
            metadatas: Her metne karşılık gelen metadata sözlükleri.
                       ``None`` ise boş dict kullanılır.

        Returns:
            Metinlerin yüklendiği yeni InMemoryVectorStore örneği.
        """
        metas = metadatas or [{} for _ in texts]
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metas)]
        return cls.from_documents(docs, embedding_fn)

    # ------------------------------------------------------------------
    # Arama
    # ------------------------------------------------------------------

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Sorguya en benzer ``k`` belgeyi döner.

        Args:
            query: Aranacak sorgu metni.
            k: Döndürülecek maksimum belge sayısı. Varsayılan: 4.

        Returns:
            En yüksek kosinüs benzerliğine sahip ``k`` Document nesnesi.
        """
        return [doc for doc, _ in self.similarity_search_with_score(query, k)]

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Sorguya en benzer ``k`` belgeyi skor ile birlikte döner.

        Args:
            query: Aranacak sorgu metni.
            k: Döndürülecek maksimum sonuç sayısı. Varsayılan: 4.

        Returns:
            ``(Document, kosinüs_skoru)`` demetlerinin azalan skor sıralı listesi.
        """
        q_emb = self.embedding_fn.embed_query(query)
        scored = [
            (doc, _cosine(q_emb, emb))
            for doc, emb in zip(self._documents, self._embeddings)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def as_retriever(self, k: int = 4) -> "VectorStoreRetriever":
        """Bu depoyu saran bir VectorStoreRetriever döner.

        Args:
            k: Her sorguda döndürülecek maksimum belge sayısı. Varsayılan: 4.

        Returns:
            Bu depoya bağlı VectorStoreRetriever nesnesi.
        """
        return VectorStoreRetriever(vectorstore=self, k=k)

    # ------------------------------------------------------------------
    # Kaydet / yükle (JSON, stdlib only)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Depoyu JSON formatında diske kaydeder.

        ``path`` dizini yoksa otomatik oluşturulur. Belgeler ve
        gömme vektörleri ``store.json`` dosyasına yazılır.

        Args:
            path: Kaydedilecek dizin yolu.
        """
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
        """Daha önce kaydedilmiş bir depoyu JSON'dan yükler.

        Args:
            path: ``save`` ile kaydedilen dizin yolu.
            embedding_fn: Yeni sorgular için kullanılacak embedding nesnesi.

        Returns:
            Belgeler ve vektörlerin yeniden yüklendiği InMemoryVectorStore.
        """
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
    """InMemoryVectorStore üzerinde çalışan basit belge alıcısı.

    ``get_relevant_documents`` ve doğrudan çağrı (``retriever(query)``)
    arayüzlerini sağlar. RAG zincirlerinde standart retriever arayüzü
    olarak kullanılır.
    """

    def __init__(self, vectorstore: InMemoryVectorStore, k: int = 4) -> None:
        """VectorStoreRetriever oluşturur.

        Args:
            vectorstore: Sorguların yönlendirileceği vektör deposu.
            k: Her sorguda döndürülecek maksimum belge sayısı. Varsayılan: 4.
        """
        self.vectorstore = vectorstore
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Sorgu için en ilgili belgeleri döner.

        Args:
            query: Doğal dil sorgu metni.

        Returns:
            En benzer ``k`` Document nesnesinin listesi.
        """
        return self.vectorstore.similarity_search(query, k=self.k)

    def __call__(self, query: str) -> List[Document]:
        """Alıcıyı ``retriever(query)`` biçiminde çağrılabilir kılar.

        Args:
            query: Doğal dil sorgu metni.

        Returns:
            En benzer ``k`` Document nesnesinin listesi.
        """
        return self.get_relevant_documents(query)
