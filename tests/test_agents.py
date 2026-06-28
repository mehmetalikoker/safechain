"""safechain.agents modülü testleri."""
import unittest
from unittest.mock import MagicMock

from safechain.agents.executor import AgentExecutor, _provider
from safechain.schema import Generation
from safechain.tools.base import Tool


def _make_tool(name: str, return_value):
    func = MagicMock(return_value=return_value)
    return Tool(name=name, description=f"{name} aracı", func=func)


def _make_llm(class_name: str, responses: list):
    """Sıralı yanıtlar dönen sahte LLM."""
    llm = MagicMock()
    llm.__class__.__name__ = class_name
    llm.generate.side_effect = responses
    return llm


class TestProvider(unittest.TestCase):
    def test_claude_detected(self):
        """Claude sınıf adı 'anthropic' sağlayıcısını döndürmeli."""
        llm = MagicMock()
        llm.__class__.__name__ = "Claude"
        self.assertEqual(_provider(llm), "anthropic")

    def test_anthropic_in_name(self):
        """Sınıf adında 'anthropic' geçiyorsa 'anthropic' sağlayıcısını döndürmeli."""
        llm = MagicMock()
        llm.__class__.__name__ = "AnthropicLLM"
        self.assertEqual(_provider(llm), "anthropic")

    def test_openai_detected(self):
        """OpenAI sınıf adı 'openai' sağlayıcısını döndürmeli."""
        llm = MagicMock()
        llm.__class__.__name__ = "OpenAI"
        self.assertEqual(_provider(llm), "openai")

    def test_unknown_defaults_to_openai(self):
        """Bilinmeyen sınıf adı varsayılan olarak 'openai' döndürmeli."""
        llm = MagicMock()
        llm.__class__.__name__ = "SomeLLM"
        self.assertEqual(_provider(llm), "openai")


class TestAgentExecutorAnthropic(unittest.TestCase):
    def _make_agent(self, tools=None, responses=None):
        tools = tools or []
        responses = responses or []
        llm = _make_llm("Claude", responses)
        return AgentExecutor(llm=llm, tools=tools), llm

    def test_direct_answer_no_tool(self):
        """Araç çağrısı olmadan gelen yanıt doğrudan döndürülmeli."""
        final = Generation(
            text="Doğrudan cevap",
            generation_info={"stop_reason": "end_turn", "tool_uses": [], "raw_content": []},
        )
        agent, _ = self._make_agent(responses=[final])
        result = agent.run("Soru?")
        self.assertEqual(result, "Doğrudan cevap")

    def test_tool_call_then_answer(self):
        """Araç çağrısı sonucu LLM'e iletilip nihai yanıt alınmalı."""
        tool = _make_tool("topla", 42)
        tool_call_gen = Generation(
            text="",
            generation_info={
                "stop_reason": "tool_use",
                "tool_uses": [{"id": "tu1", "name": "topla", "input": {"x": 1, "y": 2}}],
                "raw_content": [{"type": "tool_use", "id": "tu1", "name": "topla", "input": {}}],
            },
        )
        final_gen = Generation(
            text="Sonuç 42",
            generation_info={"stop_reason": "end_turn", "tool_uses": [], "raw_content": []},
        )
        agent, llm = self._make_agent(tools=[tool], responses=[tool_call_gen, final_gen])
        result = agent.run("1 + 2 = ?")
        self.assertEqual(result, "Sonuç 42")
        self.assertEqual(llm.generate.call_count, 2)

    def test_unknown_tool_returns_error_string(self):
        """Bilinmeyen araç adı hata mesajıyla LLM'e bildirilerek devam etmeli."""
        tool_call_gen = Generation(
            text="",
            generation_info={
                "stop_reason": "tool_use",
                "tool_uses": [{"id": "tu1", "name": "bilinmeyen", "input": {}}],
                "raw_content": [{"type": "tool_use", "id": "tu1", "name": "bilinmeyen", "input": {}}],
            },
        )
        final_gen = Generation(
            text="Araç bulunamadı",
            generation_info={"stop_reason": "end_turn", "tool_uses": [], "raw_content": []},
        )
        agent, _ = self._make_agent(responses=[tool_call_gen, final_gen])
        result = agent.run("test")
        self.assertEqual(result, "Araç bulunamadı")

    def test_tool_exception_does_not_crash_agent(self):
        """Araç hatası agent'ı çökertmemeli; hata mesajı LLM'e iletilmeli."""
        def hata_veren(**kwargs):
            raise ValueError("Araç hatası")

        broken_tool = Tool(name="bozuk", description="hata verir", func=hata_veren)
        tool_call_gen = Generation(
            text="",
            generation_info={
                "stop_reason": "tool_use",
                "tool_uses": [{"id": "tu1", "name": "bozuk", "input": {}}],
                "raw_content": [{"type": "tool_use", "id": "tu1", "name": "bozuk", "input": {}}],
            },
        )
        final_gen = Generation(
            text="Hata yönetildi",
            generation_info={"stop_reason": "end_turn", "tool_uses": [], "raw_content": []},
        )
        agent, _ = self._make_agent(tools=[broken_tool], responses=[tool_call_gen, final_gen])
        result = agent.run("test")
        self.assertEqual(result, "Hata yönetildi")

    def test_max_iterations_warning(self):
        """Maksimum iterasyon aşıldığında uyarı içeren yanıt döndürülmeli."""
        tool = _make_tool("topla", 1)
        infinite_tool_call = Generation(
            text="",
            generation_info={
                "stop_reason": "tool_use",
                "tool_uses": [{"id": "tu1", "name": "topla", "input": {"x": 1, "y": 1}}],
                "raw_content": [{"type": "tool_use"}],
            },
        )
        agent, _ = self._make_agent(
            tools=[tool],
            responses=[infinite_tool_call] * 5,
        )
        agent.max_iterations = 3
        result = agent.invoke({"input": "test"})
        self.assertIn("warning", result)

    def test_tools_dict_built_from_list(self):
        """Araç listesi isimden araç nesnesine eşleyen sözlüğe dönüştürülmeli."""
        t1 = _make_tool("a", 1)
        t2 = _make_tool("b", 2)
        agent, _ = self._make_agent(tools=[t1, t2])
        self.assertIn("a", agent.tools)
        self.assertIn("b", agent.tools)


class TestAgentExecutorOpenAI(unittest.TestCase):
    def _make_agent(self, tools=None, responses=None):
        tools = tools or []
        responses = responses or []
        llm = _make_llm("OpenAI", responses)
        return AgentExecutor(llm=llm, tools=tools), llm

    def test_direct_answer(self):
        """OpenAI sağlayıcısıyla doğrudan yanıt alınabilmeli."""
        final = Generation(
            text="OpenAI cevabı",
            generation_info={"finish_reason": "stop", "tool_calls": [], "raw_message": {"content": ""}},
        )
        agent, _ = self._make_agent(responses=[final])
        result = agent.run("Soru?")
        self.assertEqual(result, "OpenAI cevabı")

    def test_tool_call_then_answer(self):
        """OpenAI sağlayıcısıyla araç çağrısı döngüsü doğru çalışmalı."""
        import json
        tool = _make_tool("carp", 6)
        tool_call_gen = Generation(
            text="",
            generation_info={
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "carp", "arguments": json.dumps({"x": 2, "y": 3})}}
                ],
                "raw_message": {"content": None},
            },
        )
        final_gen = Generation(
            text="2 * 3 = 6",
            generation_info={"finish_reason": "stop", "tool_calls": [], "raw_message": {"content": ""}},
        )
        agent, llm = self._make_agent(tools=[tool], responses=[tool_call_gen, final_gen])
        result = agent.run("2 * 3 = ?")
        self.assertEqual(result, "2 * 3 = 6")
        self.assertEqual(llm.generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
