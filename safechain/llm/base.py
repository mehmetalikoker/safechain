from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from safechain.schema import Generation, Message


class BaseLLM(ABC):
    """Tüm LLM entegrasyonları için ortak arayüz.

    Cache desteği şablon metot kalıbıyla sağlanır: alt sınıflar yalnızca
    ``_generate`` metodunu uygular; ``generate`` cache kontrolünü otomatik
    yönetir. Cache verilmezse davranış öncekiyle aynıdır.
    """

    def __init__(self, cache: Optional[Any] = None) -> None:
        """BaseLLM'i cache ile yapılandırır.

        Args:
            cache: ``BaseCache`` örneği. ``None`` ise cache devre dışı.
        """
        self.cache = cache

    @abstractmethod
    def _generate(self, messages: List[Message], **kwargs: Any) -> Generation:
        """Sağlayıcıya özgü API çağrısını gerçekleştirir.

        Alt sınıflar bu metodu uygular. Cache kontrolü yapılmaz;
        doğrudan API'ye gidilir.

        Args:
            messages: Konuşma geçmişini oluşturan Message listesi.
            **kwargs: Sağlayıcıya özgü ek parametreler.

        Returns:
            API'den dönen ham Generation nesnesi.
        """

    def generate(self, messages: List[Message], **kwargs: Any) -> Generation:
        """Cache kontrolü yaparak LLM yanıtı üretir.

        Cache etkinse önce anahtarı arar; bulursa API'ye gitmeden döner.
        Miss durumunda ``_generate`` çağrılır ve sonuç cache'e yazılır.

        Args:
            messages: Konuşma geçmişini oluşturan Message listesi.
            **kwargs: ``_generate`` metoduna iletilecek ek parametreler.

        Returns:
            Cache'den veya API'den gelen Generation nesnesi.
        """
        if self.cache is not None:
            from safechain.cache.base import _cache_key
            model = getattr(self, "model", "unknown")
            key = _cache_key(model, messages, kwargs)
            cached = self.cache.lookup(key)
            if cached is not None:
                return cached

        result = self._generate(messages, **kwargs)

        if self.cache is not None:
            self.cache.update(key, result)  # type: ignore[possibly-undefined]

        return result

    def predict(self, prompt: str, **kwargs: Any) -> str:
        """Tek satır kullanıcı prompt'u için kısa yol.

        Args:
            prompt: Modele gönderilecek kullanıcı mesajı metni.
            **kwargs: ``generate`` metoduna iletilecek ek parametreler.

        Returns:
            Modelin ürettiği ham metin yanıtı.
        """
        return self.generate([Message(role="user", content=prompt)], **kwargs).text

    def predict_messages(self, messages: List[Message], **kwargs: Any) -> str:
        """Tam mesaj listesiyle modeli çağırır ve metni döner.

        Args:
            messages: Modele gönderilecek Message nesneleri listesi.
            **kwargs: ``generate`` metoduna iletilecek ek parametreler.

        Returns:
            Modelin ürettiği ham metin yanıtı.
        """
        return self.generate(messages, **kwargs).text

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """``llm("soru")`` biçiminde çağrılabilir kılar.

        Args:
            prompt: Modele gönderilecek kullanıcı mesajı metni.
            **kwargs: ``predict`` metoduna iletilecek ek parametreler.

        Returns:
            Modelin ürettiği ham metin yanıtı.
        """
        return self.predict(prompt, **kwargs)
