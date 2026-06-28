from __future__ import annotations
import json
import math
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional


class OpenAIEmbeddings:
    """OpenAI embedding API'sini doğrudan urllib ile çağırır (SDK gerekmez)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        dimensions: Optional[int] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Metin listesini vektör listesine dönüştürür.

        Args:
            texts: Gömülecek metin dizeleri listesi.

        Returns:
            Her metne karşılık gelen float vektörlerinin listesi.
        """
        return self._call(texts)

    def embed_query(self, text: str) -> List[float]:
        """Tek bir sorgu metnini vektöre dönüştürür.

        Args:
            text: Gömülecek sorgu metni.

        Returns:
            Metnin float vektörü.
        """
        return self._call([text])[0]

    def _call(self, texts: List[str]) -> List[List[float]]:
        """OpenAI Embeddings API'sine HTTP isteği gönderir.

        Sonuçları indeks sırasına göre sıralayarak döner (API sıralamanın
        korunmasını garanti etmez).

        Args:
            texts: Gömülecek metin dizeleri listesi.

        Returns:
            Sıralı float vektörlerinin listesi.

        Raises:
            RuntimeError: HTTP isteği başarısız olduğunda fırlatılır.
        """
        body = {"model": self.model, "input": texts}
        if self.dimensions:
            body["dimensions"] = self.dimensions  # type: ignore[assignment]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        url = f"{self.base_url}/embeddings"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Embedding API error {exc.code}: {exc.read().decode()}") from exc
        return [item["embedding"] for item in sorted(result["data"], key=lambda x: x["index"])]


class TFIDFEmbeddings:
    """Sıfır API çağrısı, sıfır bağımlılık: stdlib TF-IDF vektörleri.

    Nöral embedding kadar güçlü değil ama network erişimi olmayan
    ortamlarda veya prototipleme için kullanışlı.
    """

    def __init__(self) -> None:
        self._vocab: List[str] = []
        self._idf: dict = {}
        self._fitted = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Metni küçük harfe çevirip kelime tokenlarına ayırır.

        Args:
            text: Tokenlanacak ham metin.

        Returns:
            Küçük harfli kelime tokenlarının listesi.
        """
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, texts: List[str]) -> "TFIDFEmbeddings":
        """Kelime dağarcığını ve IDF ağırlıklarını corpus'tan öğrenir.

        Her kelime için ``log((N+1) / (df+1))`` formülüyle IDF hesaplanır
        (sıfıra bölmeyi önlemek için +1 düzeltmesi uygulanır).

        Args:
            texts: Eğitim corpus'u; tüm belgeler.

        Returns:
            Kendisini (zincirleme kullanım için).
        """
        N = len(texts)
        df: dict = {}
        for text in texts:
            for word in set(self._tokenize(text)):
                df[word] = df.get(word, 0) + 1
        self._vocab = sorted(df)
        self._idf = {w: math.log((N + 1) / (df[w] + 1)) for w in self._vocab}
        self._fitted = True
        return self

    def _vectorize(self, text: str) -> List[float]:
        """Tek bir metni TF-IDF vektörüne dönüştürür.

        TF (term frequency) = kelime sayısı / toplam token sayısı.
        Sonuç vektörü ``vocab`` sözlüğündeki sırayla döner.

        Args:
            text: Vektörleştirilecek metin.

        Returns:
            Kelime dağarcığı boyutunda float vektörü.
        """
        tokens = self._tokenize(text)
        n = len(tokens) or 1
        tf: dict = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1 / n
        return [tf.get(w, 0.0) * self._idf.get(w, 0.0) for w in self._vocab]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Metin listesini TF-IDF vektörlerine dönüştürür.

        İlk çağrıda ``fit`` otomatik olarak çalıştırılır; corpus bu
        metinlerden oluşur.

        Args:
            texts: Gömülecek metin dizeleri listesi.

        Returns:
            Her metne karşılık gelen TF-IDF vektörlerinin listesi.
        """
        if not self._fitted:
            self.fit(texts)
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Tek bir sorgu metnini TF-IDF vektörüne dönüştürür.

        ``embed_documents`` ile önceden ``fit`` edilmiş olması gerekir.

        Args:
            text: Gömülecek sorgu metni.

        Returns:
            Metnin TF-IDF vektörü.

        Raises:
            RuntimeError: ``embed_documents`` daha önce çağrılmamışsa fırlatılır.
        """
        if not self._fitted:
            raise RuntimeError("embed_documents() ile önce fit edilmeli.")
        return self._vectorize(text)
