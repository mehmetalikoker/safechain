from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from safechain.schema import Generation, Message


class BaseLLM(ABC):
    """Tüm LLM entegrasyonları için ortak arayüz.

    Alt sınıflar yalnızca ``generate`` metodunu uygulamak zorundadır;
    diğer kolaylık metotları bu temel üzerine inşa edilmiştir.
    """

    @abstractmethod
    def generate(self, messages: List[Message], **kwargs: Any) -> Generation:
        """Mesaj listesinden LLM yanıtı üretir.

        Args:
            messages: Konuşma geçmişini oluşturan Message nesnelerinin listesi.
                      Sistem, kullanıcı ve asistan rolleri içerebilir.
            **kwargs: Sağlayıcıya özgü ek parametreler (örn. ``tools``,
                      ``temperature`` override).

        Returns:
            Üretilen metni ve meta verileri içeren Generation nesnesi.
        """

    def predict(self, prompt: str, **kwargs: Any) -> str:
        """Tek satır kullanıcı prompt'u için kısa yol.

        Prompt'u bir "user" mesajına sararak ``generate`` çağırır ve
        üretilen metni doğrudan döner.

        Args:
            prompt: Modele gönderilecek kullanıcı mesajı metni.
            **kwargs: ``generate`` metoduna iletilecek ek parametreler.

        Returns:
            Modelin ürettiği ham metin yanıtı.
        """
        return self.generate([Message(role="user", content=prompt)], **kwargs).text

    def predict_messages(self, messages: List[Message], **kwargs: Any) -> str:
        """Tam mesaj listesiyle modeli çağırır ve metni döner.

        ``generate`` metodunun ``text`` alanına doğrudan erişmek için
        kullanılan kolaylık sarmalayıcısı.

        Args:
            messages: Modele gönderilecek Message nesneleri listesi.
            **kwargs: ``generate`` metoduna iletilecek ek parametreler.

        Returns:
            Modelin ürettiği ham metin yanıtı.
        """
        return self.generate(messages, **kwargs).text

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """LLM örneğini doğrudan çağrılabilir (callable) hale getirir.

        ``llm("soru")`` biçimindeki kullanımı mümkün kılar; ``predict``
        metoduna yönlendirir.

        Args:
            prompt: Modele gönderilecek kullanıcı mesajı metni.
            **kwargs: ``predict`` metoduna iletilecek ek parametreler.

        Returns:
            Modelin ürettiği ham metin yanıtı.
        """
        return self.predict(prompt, **kwargs)
