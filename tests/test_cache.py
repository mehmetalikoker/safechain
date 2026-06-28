"""safechain.cache modülü testleri."""
import os
import tempfile
import time
import unittest
from io import BytesIO
from unittest.mock import patch

from safechain.cache.base import BaseCache, _cache_key
from safechain.cache.memory import InMemoryCache
from safechain.cache.sqlite import SQLiteCache
from safechain.schema import Generation, Message


# ---------------------------------------------------------------------------
# _cache_key
# ---------------------------------------------------------------------------

class TestCacheKey(unittest.TestCase):
    def _messages(self):
        return [Message(role="user", content="merhaba")]

    def test_same_input_same_key(self):
        msgs = self._messages()
        k1 = _cache_key("model-x", msgs, {})
        k2 = _cache_key("model-x", msgs, {})
        self.assertEqual(k1, k2)

    def test_different_model_different_key(self):
        msgs = self._messages()
        self.assertNotEqual(
            _cache_key("model-a", msgs, {}),
            _cache_key("model-b", msgs, {}),
        )

    def test_different_message_different_key(self):
        m1 = [Message(role="user", content="soru1")]
        m2 = [Message(role="user", content="soru2")]
        self.assertNotEqual(_cache_key("m", m1, {}), _cache_key("m", m2, {}))

    def test_different_kwargs_different_key(self):
        msgs = self._messages()
        self.assertNotEqual(
            _cache_key("m", msgs, {"temperature": 0.5}),
            _cache_key("m", msgs, {"temperature": 1.0}),
        )

    def test_kwargs_order_independent(self):
        msgs = self._messages()
        k1 = _cache_key("m", msgs, {"a": 1, "b": 2})
        k2 = _cache_key("m", msgs, {"b": 2, "a": 1})
        self.assertEqual(k1, k2)

    def test_returns_64_char_hex(self):
        key = _cache_key("m", self._messages(), {})
        self.assertEqual(len(key), 64)
        int(key, 16)  # hex olduğunu doğrular

    def test_nonserializable_kwargs_ignored(self):
        msgs = self._messages()
        obj = object()
        k1 = _cache_key("m", msgs, {"fn": obj})
        k2 = _cache_key("m", msgs, {})
        self.assertEqual(k1, k2)


# ---------------------------------------------------------------------------
# InMemoryCache
# ---------------------------------------------------------------------------

class TestInMemoryCache(unittest.TestCase):
    def setUp(self):
        self.cache = InMemoryCache()
        self.gen = Generation(text="yanıt", generation_info={"stop_reason": "end_turn"})

    def test_miss_returns_none(self):
        self.assertIsNone(self.cache.lookup("bilinmeyen"))

    def test_update_and_lookup(self):
        self.cache.update("anahtar", self.gen)
        result = self.cache.lookup("anahtar")
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "yanıt")

    def test_generation_info_preserved(self):
        self.cache.update("k", self.gen)
        result = self.cache.lookup("k")
        self.assertEqual(result.generation_info["stop_reason"], "end_turn")

    def test_overwrite_existing(self):
        self.cache.update("k", self.gen)
        new_gen = Generation(text="yeni yanıt")
        self.cache.update("k", new_gen)
        self.assertEqual(self.cache.lookup("k").text, "yeni yanıt")

    def test_clear(self):
        self.cache.update("k1", self.gen)
        self.cache.update("k2", self.gen)
        self.cache.clear()
        self.assertIsNone(self.cache.lookup("k1"))
        self.assertIsNone(self.cache.lookup("k2"))

    def test_len(self):
        self.assertEqual(len(self.cache), 0)
        self.cache.update("a", self.gen)
        self.cache.update("b", self.gen)
        self.assertEqual(len(self.cache), 2)

    def test_independent_instances(self):
        cache2 = InMemoryCache()
        self.cache.update("k", self.gen)
        self.assertIsNone(cache2.lookup("k"))


# ---------------------------------------------------------------------------
# SQLiteCache
# ---------------------------------------------------------------------------

class TestSQLiteCache(unittest.TestCase):
    def setUp(self):
        self.cache = SQLiteCache(":memory:")
        self.gen = Generation(text="db yanıtı", generation_info={"usage": {"tokens": 10}})

    def tearDown(self):
        self.cache.close()

    def test_miss_returns_none(self):
        self.assertIsNone(self.cache.lookup("yok"))

    def test_update_and_lookup(self):
        self.cache.update("k", self.gen)
        result = self.cache.lookup("k")
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "db yanıtı")

    def test_generation_info_preserved(self):
        self.cache.update("k", self.gen)
        result = self.cache.lookup("k")
        self.assertEqual(result.generation_info["usage"]["tokens"], 10)

    def test_overwrite(self):
        self.cache.update("k", self.gen)
        self.cache.update("k", Generation(text="yeni"))
        self.assertEqual(self.cache.lookup("k").text, "yeni")

    def test_clear(self):
        self.cache.update("k", self.gen)
        self.cache.clear()
        self.assertIsNone(self.cache.lookup("k"))

    def test_len(self):
        self.assertEqual(len(self.cache), 0)
        self.cache.update("a", self.gen)
        self.assertEqual(len(self.cache), 1)

    def test_ttl_valid_entry(self):
        cache = SQLiteCache(":memory:", ttl=60)
        cache.update("k", self.gen)
        self.assertIsNotNone(cache.lookup("k"))
        cache.close()

    def test_ttl_expired_entry(self):
        cache = SQLiteCache(":memory:", ttl=0.01)
        cache.update("k", self.gen)
        time.sleep(0.05)
        self.assertIsNone(cache.lookup("k"))
        cache.close()

    def test_persistent_on_disk(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            c1 = SQLiteCache(db_path)
            c1.update("k", Generation(text="kalıcı"))
            c1.close()

            c2 = SQLiteCache(db_path)
            result = c2.lookup("k")
            c2.close()

            self.assertIsNotNone(result)
            self.assertEqual(result.text, "kalıcı")
        finally:
            os.unlink(db_path)

    def test_complex_generation_info_serialized(self):
        gen = Generation(
            text="test",
            generation_info={
                "tool_uses": [{"id": "1", "name": "hesapla"}],
                "usage": {"input": 5, "output": 10},
            },
        )
        self.cache.update("k", gen)
        result = self.cache.lookup("k")
        self.assertEqual(result.generation_info["tool_uses"][0]["name"], "hesapla")


# ---------------------------------------------------------------------------
# LLM entegrasyonu (Claude + cache)
# ---------------------------------------------------------------------------

import json
from unittest.mock import MagicMock
from safechain.llm.anthropic import Claude
from safechain.llm.openai import OpenAI

ANTHROPIC_RESPONSE = {
    "content": [{"type": "text", "text": "Önbellek testi"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3, "output_tokens": 5},
}

OPENAI_RESPONSE = {
    "choices": [{"message": {"content": "OA önbellek", "tool_calls": None}, "finish_reason": "stop"}],
    "usage": {},
}


def _mock_http(body: dict):
    resp = MagicMock()
    resp.read.return_value = json.dumps(body).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestLLMWithCache(unittest.TestCase):
    def test_claude_cache_hit_skips_api(self):
        cache = InMemoryCache()
        llm = Claude(api_key="test", cache=cache)
        msgs = [Message(role="user", content="test")]

        call_count = 0
        def fake_urlopen(req):
            nonlocal call_count
            call_count += 1
            return _mock_http(ANTHROPIC_RESPONSE)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            gen1 = llm.generate(msgs)
            gen2 = llm.generate(msgs)  # cache hit — API çağrılmaz

        self.assertEqual(call_count, 1)
        self.assertEqual(gen1.text, gen2.text)

    def test_claude_different_messages_different_cache_entry(self):
        cache = InMemoryCache()
        llm = Claude(api_key="test", cache=cache)

        with patch("urllib.request.urlopen", return_value=_mock_http(ANTHROPIC_RESPONSE)):
            llm.generate([Message(role="user", content="soru1")])
            llm.generate([Message(role="user", content="soru2")])

        self.assertEqual(len(cache), 2)

    def test_openai_cache_hit_skips_api(self):
        cache = InMemoryCache()
        llm = OpenAI(api_key="test", cache=cache)
        msgs = [Message(role="user", content="merhaba")]

        call_count = 0
        def fake_urlopen(req):
            nonlocal call_count
            call_count += 1
            return _mock_http(OPENAI_RESPONSE)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.generate(msgs)
            llm.generate(msgs)

        self.assertEqual(call_count, 1)

    def test_no_cache_always_calls_api(self):
        llm = Claude(api_key="test")  # cache=None
        msgs = [Message(role="user", content="test")]

        call_count = 0
        def fake_urlopen(req):
            nonlocal call_count
            call_count += 1
            return _mock_http(ANTHROPIC_RESPONSE)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.generate(msgs)
            llm.generate(msgs)

        self.assertEqual(call_count, 2)

    def test_sqlite_cache_with_llm(self):
        cache = SQLiteCache(":memory:")
        llm = Claude(api_key="test", cache=cache)
        msgs = [Message(role="user", content="kalıcı soru")]

        with patch("urllib.request.urlopen", return_value=_mock_http(ANTHROPIC_RESPONSE)):
            gen1 = llm.generate(msgs)

        gen2 = llm.generate(msgs)  # API çağrısı yok — SQLite'dan
        self.assertEqual(gen1.text, gen2.text)
        cache.close()

    def test_top_level_import(self):
        from safechain import InMemoryCache, SQLiteCache
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
