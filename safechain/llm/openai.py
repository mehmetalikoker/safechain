from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from safechain.llm.base import BaseLLM
from safechain.schema import Generation, Message


class OpenAI(BaseLLM):
    """OpenAI Chat Completions API için LLM entegrasyonu.

    Harici SDK bağımlılığı olmadan stdlib ``urllib`` ile OpenAI
    ``/chat/completions`` endpoint'ine doğrudan HTTP istekleri gönderir.
    OpenAI uyumlu diğer sağlayıcılar (örn. Azure, Ollama) için
    ``base_url`` parametresi özelleştirilebilir.

    Attributes:
        model: Kullanılacak model kimliği (örn. "gpt-4o").
        api_key: OpenAI API anahtarı. Verilmezse ``OPENAI_API_KEY``
                 ortam değişkeninden okunur.
        base_url: API'nin temel URL'si. Varsayılan: "https://api.openai.com/v1".
        max_tokens: Yanıtta üretilebilecek maksimum token sayısı.
        temperature: Örnekleme sıcaklığı (0.0–2.0).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> None:
        """OpenAI istemcisini yapılandırır.

        Args:
            model: Kullanılacak OpenAI model kimliği. Varsayılan: "gpt-4o".
            api_key: OpenAI API anahtarı. ``None`` ise ``OPENAI_API_KEY``
                     ortam değişkenine bakılır.
            base_url: Chat Completions endpoint'inin temel URL'si.
                      Sondaki "/" karakterleri otomatik temizlenir.
            max_tokens: Yanıtta üretilebilecek maksimum token sayısı.
                        Varsayılan: 4096.
            temperature: Örnekleme sıcaklığı. Varsayılan: 1.0.
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Generation:
        """OpenAI Chat Completions API'sini çağırarak yanıt üretir.

        Araç tanımları verildiğinde ``tool_choice`` otomatik olarak
        "auto" değerine ayarlanır. Araç çağrısı sonuçları
        ``generation_info["tool_calls"]`` içinde döner.

        Args:
            messages: Konuşma geçmişi. "system", "user", "assistant" ve
                      "tool" rollerini içerebilir.
            tools: OpenAI function-calling şemasına uygun araç tanımları
                   listesi. ``None`` ise araç desteği etkinleştirilmez.
            **kwargs: API isteğinin gövdesine doğrudan eklenmek üzere ek
                      parametreler (örn. ``top_p``, ``response_format``).

        Returns:
            Generation nesnesi. Alanlar:
            - ``text``: Modelin metin yanıtı (araç çağrısı varsa boş olabilir).
            - ``generation_info["tool_calls"]``: OpenAI tool_calls listesi.
            - ``generation_info["finish_reason"]``: Durdurma nedeni
              ("stop", "tool_calls", "length" vb.).
            - ``generation_info["usage"]``: Token kullanım istatistikleri.
            - ``generation_info["raw_message"]``: Ham asistan mesajı nesnesi.
            - ``generation_info["raw"]``: API'den gelen tam JSON yanıtı.

        Raises:
            RuntimeError: HTTP isteği başarısız olduğunda hata kodu ve
                          mesajıyla birlikte fırlatılır.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        body.update(kwargs)

        url = f"{self.base_url}/chat/completions"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                result: Dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"OpenAI API error {exc.code}: {exc.read().decode()}"
            ) from exc

        choice = result["choices"][0]
        msg = choice["message"]
        text: str = msg.get("content") or ""
        tool_calls: List[Dict[str, Any]] = msg.get("tool_calls") or []

        return Generation(
            text=text,
            generation_info={
                "tool_calls": tool_calls,
                "finish_reason": choice.get("finish_reason"),
                "usage": result.get("usage", {}),
                "raw_message": msg,
                "raw": result,
            },
        )
