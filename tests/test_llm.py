"""safechain.llm modülü testleri (HTTP çağrıları mock'lanmıştır)."""
import json
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from safechain.llm.anthropic import Claude
from safechain.llm.openai import OpenAI
from safechain.schema import Generation, Message


def _mock_response(body: dict, status: int = 200):
    """urllib.request.urlopen için sahte yanıt oluşturur."""
    encoded = json.dumps(body).encode("utf-8")
    mock = MagicMock()
    mock.read.return_value = encoded
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


ANTHROPIC_RESPONSE = {
    "content": [{"type": "text", "text": "Merhaba!"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 3},
}

OPENAI_RESPONSE = {
    "choices": [
        {
            "message": {"content": "Merhaba!", "tool_calls": None},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
}

ANTHROPIC_TOOL_RESPONSE = {
    "content": [
        {"type": "tool_use", "id": "tu_1", "name": "hesapla", "input": {"x": 1}},
    ],
    "stop_reason": "tool_use",
    "usage": {},
}

OPENAI_TOOL_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "hesapla", "arguments": '{"x": 1}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {},
}


class TestClaudeGenerate(unittest.TestCase):
    def _call(self, response_body=None, api_key="test-key"):
        body = response_body or ANTHROPIC_RESPONSE
        with patch("urllib.request.urlopen", return_value=_mock_response(body)):
            llm = Claude(api_key=api_key)
            return llm.generate([Message(role="user", content="test")])

    def test_returns_generation(self):
        gen = self._call()
        self.assertIsInstance(gen, Generation)
        self.assertEqual(gen.text, "Merhaba!")

    def test_stop_reason_in_info(self):
        gen = self._call()
        self.assertEqual(gen.generation_info["stop_reason"], "end_turn")

    def test_usage_in_info(self):
        gen = self._call()
        self.assertIn("usage", gen.generation_info)

    def test_tool_use_response(self):
        gen = self._call(ANTHROPIC_TOOL_RESPONSE)
        self.assertEqual(gen.generation_info["stop_reason"], "tool_use")
        self.assertEqual(len(gen.generation_info["tool_uses"]), 1)
        self.assertEqual(gen.generation_info["tool_uses"][0]["name"], "hesapla")

    def test_system_message_separated(self):
        """System mesajı API isteğinden ayrıştırılmalı."""
        captured = {}
        original_urlopen = __import__("urllib.request", fromlist=["urlopen"]).urlopen

        def fake_urlopen(req):
            captured["body"] = json.loads(req.data.decode())
            return _mock_response(ANTHROPIC_RESPONSE)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm = Claude(api_key="key")
            llm.generate([
                Message(role="system", content="Sistem talimatı"),
                Message(role="user", content="kullanıcı sorusu"),
            ])

        body = captured["body"]
        self.assertEqual(body["system"], "Sistem talimatı")
        roles = [m["role"] for m in body["messages"]]
        self.assertNotIn("system", roles)

    def test_http_error_raises_runtime_error(self):
        import urllib.error
        error = urllib.error.HTTPError(
            url="", code=401, msg="Unauthorized",
            hdrs=None, fp=BytesIO(b"invalid key"),  # type: ignore[arg-type]
        )
        with patch("urllib.request.urlopen", side_effect=error):
            llm = Claude(api_key="bad-key")
            with self.assertRaises(RuntimeError) as ctx:
                llm.generate([Message(role="user", content="test")])
        self.assertIn("401", str(ctx.exception))

    def test_predict_shortcut(self):
        with patch("urllib.request.urlopen", return_value=_mock_response(ANTHROPIC_RESPONSE)):
            llm = Claude(api_key="key")
            result = llm.predict("merhaba")
        self.assertEqual(result, "Merhaba!")

    def test_call_operator(self):
        with patch("urllib.request.urlopen", return_value=_mock_response(ANTHROPIC_RESPONSE)):
            llm = Claude(api_key="key")
            result = llm("merhaba")
        self.assertEqual(result, "Merhaba!")

    def test_default_model(self):
        llm = Claude(api_key="key")
        self.assertEqual(llm.model, "claude-sonnet-4-6")

    def test_system_prompt_attribute(self):
        llm = Claude(api_key="key", system="Her zaman Türkçe yanıt ver.")
        self.assertEqual(llm.system, "Her zaman Türkçe yanıt ver.")


class TestOpenAIGenerate(unittest.TestCase):
    def _call(self, response_body=None, api_key="test-key"):
        body = response_body or OPENAI_RESPONSE
        with patch("urllib.request.urlopen", return_value=_mock_response(body)):
            llm = OpenAI(api_key=api_key)
            return llm.generate([Message(role="user", content="test")])

    def test_returns_generation(self):
        gen = self._call()
        self.assertIsInstance(gen, Generation)
        self.assertEqual(gen.text, "Merhaba!")

    def test_finish_reason_in_info(self):
        gen = self._call()
        self.assertEqual(gen.generation_info["finish_reason"], "stop")

    def test_tool_calls_response(self):
        gen = self._call(OPENAI_TOOL_RESPONSE)
        self.assertEqual(gen.generation_info["finish_reason"], "tool_calls")
        calls = gen.generation_info["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "hesapla")

    def test_http_error_raises_runtime_error(self):
        import urllib.error
        error = urllib.error.HTTPError(
            url="", code=429, msg="Rate limit",
            hdrs=None, fp=BytesIO(b"rate limited"),  # type: ignore[arg-type]
        )
        with patch("urllib.request.urlopen", side_effect=error):
            llm = OpenAI(api_key="key")
            with self.assertRaises(RuntimeError) as ctx:
                llm.generate([Message(role="user", content="test")])
        self.assertIn("429", str(ctx.exception))

    def test_predict_messages(self):
        with patch("urllib.request.urlopen", return_value=_mock_response(OPENAI_RESPONSE)):
            llm = OpenAI(api_key="key")
            result = llm.predict_messages([Message(role="user", content="hi")])
        self.assertEqual(result, "Merhaba!")

    def test_base_url_trailing_slash_stripped(self):
        llm = OpenAI(api_key="key", base_url="https://api.openai.com/v1/")
        self.assertEqual(llm.base_url, "https://api.openai.com/v1")

    def test_default_model(self):
        llm = OpenAI(api_key="key")
        self.assertEqual(llm.model, "gpt-4o")


if __name__ == "__main__":
    unittest.main()
