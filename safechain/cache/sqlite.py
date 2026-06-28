from __future__ import annotations
import json
import sqlite3
import time
from typing import Any, Dict, Optional

from safechain.cache.base import BaseCache
from safechain.schema import Generation


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key   TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    info        TEXT NOT NULL,
    created_at  REAL NOT NULL
)
"""


def _serialize_info(info: Dict[str, Any]) -> str:
    """generation_info'yu JSON string'e dönüştürür.

    Serileştirilemeyen değerler (raw Anthropic/OpenAI nesneleri gibi)
    string temsiline çevrilir; veri kaybı olmaz, sadece tip değişir.

    Args:
        info: Generation.generation_info sözlüğü.

    Returns:
        JSON string.
    """
    def _safe(obj: Any) -> Any:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    safe_info = {k: _safe(v) for k, v in info.items()}
    return json.dumps(safe_info, ensure_ascii=False)


class SQLiteCache(BaseCache):
    """LLM yanıtlarını SQLite veritabanında kalıcı olarak saklar.

    Uygulama yeniden başlatıldıktan sonra da cache geçerliliğini korur.
    Aynı DB dosyasını birden fazla süreç paylaşabilir (SQLite WAL modu).
    Sıfır dış bağımlılık — yalnızca stdlib ``sqlite3`` kullanılır.

    Attributes:
        db_path: SQLite veritabanı dosyasının yolu.
                 ``":memory:"`` geçilirse bellekte geçici DB oluşturulur.
        ttl: Saniye cinsinden cache geçerlilik süresi.
             ``None`` ise girişler hiç sona ermez.
    """

    def __init__(self, db_path: str = "safechain_cache.db", ttl: Optional[float] = None) -> None:
        """SQLiteCache oluşturur ve tabloyu yoksa yaratır.

        Args:
            db_path: SQLite dosya yolu veya ``":memory:"``.
                     Varsayılan: ``"safechain_cache.db"``.
            ttl: Cache geçerlilik süresi (saniye). ``None`` ise sonsuz.
        """
        self.db_path = db_path
        self.ttl = ttl
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def lookup(self, key: str) -> Optional[Generation]:
        """Cache'de anahtarı arar; süresi dolmuşsa siler ve None döner.

        Args:
            key: SHA-256 cache anahtarı.

        Returns:
            Geçerli önbellek girişi veya ``None``.
        """
        row = self._conn.execute(
            "SELECT text, info, created_at FROM llm_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()

        if row is None:
            return None

        text, info_json, created_at = row

        if self.ttl is not None and (time.time() - created_at) > self.ttl:
            self._delete(key)
            return None

        return Generation(text=text, generation_info=json.loads(info_json))

    def update(self, key: str, value: Generation) -> None:
        """Generation nesnesini veritabanına kaydeder veya günceller.

        Args:
            key: SHA-256 cache anahtarı.
            value: Saklanacak Generation nesnesi.
        """
        self._conn.execute(
            """
            INSERT INTO llm_cache (cache_key, text, info, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                text       = excluded.text,
                info       = excluded.info,
                created_at = excluded.created_at
            """,
            (key, value.text, _serialize_info(value.generation_info), time.time()),
        )
        self._conn.commit()

    def clear(self) -> None:
        """Tablodaki tüm cache girişlerini siler."""
        self._conn.execute("DELETE FROM llm_cache")
        self._conn.commit()

    def _delete(self, key: str) -> None:
        """Tek bir cache girişini siler (TTL sona erince çağrılır)."""
        self._conn.execute("DELETE FROM llm_cache WHERE cache_key = ?", (key,))
        self._conn.commit()

    def __len__(self) -> int:
        """Cache'deki toplam giriş sayısını döner."""
        row = self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Veritabanı bağlantısını kapatır."""
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
