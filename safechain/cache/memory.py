from __future__ import annotations
from typing import Dict, Optional

from safechain.cache.base import BaseCache
from safechain.schema import Generation


class InMemoryCache(BaseCache):
    """LLM yanıtlarını uygulama belleğinde önbelleğe alır.

    Process yeniden başlatıldığında cache sıfırlanır. Geliştirme ve
    test ortamları ya da kısa ömürlü servisler için uygundur.

    Attributes:
        _store: Cache anahtarlarını Generation nesnelerine eşleyen sözlük.
    """

    def __init__(self) -> None:
        """Boş bir InMemoryCache oluşturur."""
        self._store: Dict[str, Generation] = {}

    def lookup(self, key: str) -> Optional[Generation]:
        """Cache'de anahtarı arar ve varsa Generation döner.

        Args:
            key: SHA-256 cache anahtarı.

        Returns:
            Önbellekteki Generation veya ``None``.
        """
        return self._store.get(key)

    def update(self, key: str, value: Generation) -> None:
        """Generation nesnesini cache'e kaydeder.

        Args:
            key: SHA-256 cache anahtarı.
            value: Saklanacak Generation nesnesi.
        """
        self._store[key] = value

    def clear(self) -> None:
        """Tüm önbellek girişlerini siler."""
        self._store.clear()

    def __len__(self) -> int:
        """Cache'deki giriş sayısını döner."""
        return len(self._store)
