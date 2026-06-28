from __future__ import annotations
import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from safechain.schema import Generation, Message


def _cache_key(model: str, messages: List[Message], kwargs: Dict[str, Any]) -> str:
    """LLM çağrısı için deterministik cache anahtarı üretir.

    Model adı, mesaj içerikleri ve ek parametreler SHA-256 ile özetlenir.
    Aynı giriş her zaman aynı anahtarı üretir; parametre sırası önemsizdir.

    Args:
        model: Kullanılan model kimliği (örn. "claude-sonnet-4-6").
        messages: Konuşma geçmişini oluşturan Message listesi.
        kwargs: ``generate`` çağrısına iletilen ek parametreler.

    Returns:
        64 karakterlik hex SHA-256 özeti.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": m.role, "content": m.content}
            for m in messages
        ],
        "kwargs": {k: v for k, v in sorted(kwargs.items()) if _is_serializable(v)},
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_serializable(value: Any) -> bool:
    """Değerin JSON ile serileştirilebileceğini kontrol eder."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


class BaseCache(ABC):
    """LLM yanıtlarını önbelleğe alan sınıflar için ortak arayüz.

    Alt sınıflar ``lookup``, ``update`` ve ``clear`` metodlarını uygulamalıdır.
    """

    @abstractmethod
    def lookup(self, key: str) -> Optional[Generation]:
        """Cache'de anahtar varsa Generation döner, yoksa None.

        Args:
            key: ``_cache_key`` ile üretilen SHA-256 hash string.

        Returns:
            Önbellekteki Generation nesnesi veya ``None``.
        """

    @abstractmethod
    def update(self, key: str, value: Generation) -> None:
        """Cache'e yeni bir giriş ekler veya mevcut girişi günceller.

        Args:
            key: Cache anahtarı.
            value: Saklanacak Generation nesnesi.
        """

    @abstractmethod
    def clear(self) -> None:
        """Cache'deki tüm girişleri siler."""
