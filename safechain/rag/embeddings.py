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
        return self._call(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._call([text])[0]

    def _call(self, texts: List[str]) -> List[List[float]]:
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
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, texts: List[str]) -> "TFIDFEmbeddings":
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
        tokens = self._tokenize(text)
        n = len(tokens) or 1
        tf: dict = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1 / n
        return [tf.get(w, 0.0) * self._idf.get(w, 0.0) for w in self._vocab]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            self.fit(texts)
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if not self._fitted:
            raise RuntimeError("embed_documents() ile önce fit edilmeli.")
        return self._vectorize(text)
