"""safechain.chains modülü testleri."""
import unittest
from unittest.mock import MagicMock, patch
from safechain.chains.base import Chain
from safechain.chains.llm_chain import LLMChain
from safechain.chains.sequential import SequentialChain, SimpleSequentialChain
from safechain.prompts.template import ChatPromptTemplate, PromptTemplate
from safechain.schema import Generation, Message


def _mock_llm(response: str = "test yanıtı") -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = Generation(text=response)
    return llm


class ConcreteChain(Chain):
    """Test için minimal Chain implementasyonu."""

    @property
    def input_keys(self):
        return ["input"]

    @property
    def output_keys(self):
        return ["output"]

    def invoke(self, inputs):
        return {"output": inputs["input"].upper()}


class TestChainBase(unittest.TestCase):
    def setUp(self):
        self.chain = ConcreteChain()

    def test_invoke(self):
        """Chain.invoke() giriş sözlüğünü işleyerek çıktı sözlüğü döndürmeli."""
        result = self.chain.invoke({"input": "merhaba"})
        self.assertEqual(result["output"], "MERHABA")

    def test_run_positional(self):
        """run() ile pozisyonel argüman chain'in tek girişine atanmalı."""
        result = self.chain.run("dünya")
        self.assertEqual(result, "DÜNYA")

    def test_run_keyword(self):
        """run() ile anahtar kelime argümanı doğru işlenmeli."""
        result = self.chain.run(input="python")
        self.assertEqual(result, "PYTHON")

    def test_call_with_string(self):
        """__call__ string girişi invoke'a iletmeli."""
        result = self.chain("test")
        self.assertEqual(result["output"], "TEST")

    def test_call_with_dict(self):
        """__call__ sözlük girişini doğrudan invoke'a iletmeli."""
        result = self.chain({"input": "dict"})
        self.assertEqual(result["output"], "DICT")

    def test_pipe_creates_sequential(self):
        """| operatörü iki chain'i SimpleSequentialChain ile birleştirmeli."""
        chain2 = ConcreteChain()
        seq = self.chain | chain2
        self.assertIsInstance(seq, SimpleSequentialChain)


class TestLLMChain(unittest.TestCase):
    def test_basic_invoke_with_prompt_template(self):
        """PromptTemplate ile LLMChain çağrısı LLM'e iletilmeli."""
        llm = _mock_llm("cevap")
        prompt = PromptTemplate.from_template("{soru}")
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.invoke({"soru": "test?"})
        self.assertEqual(result["text"], "cevap")
        llm.generate.assert_called_once()

    def test_basic_invoke_with_chat_prompt(self):
        """ChatPromptTemplate ile LLMChain doğru çalışmalı."""
        llm = _mock_llm("chat cevap")
        prompt = ChatPromptTemplate.from_messages([("user", "{q}")])
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.invoke({"q": "merhaba"})
        self.assertEqual(result["text"], "chat cevap")

    def test_custom_output_key(self):
        """output_key parametresi çıktı sözlüğü anahtarını belirlemeli."""
        llm = _mock_llm("sonuç")
        prompt = PromptTemplate.from_template("{x}")
        chain = LLMChain(llm=llm, prompt=prompt, output_key="sonuç")
        result = chain.invoke({"x": "giriş"})
        self.assertIn("sonuç", result)

    def test_predict(self):
        """predict() invoke() sonucunun metin kısmını döndürmeli."""
        llm = _mock_llm("predict yanıtı")
        prompt = PromptTemplate.from_template("{x}")
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.predict(x="değer")
        self.assertEqual(result, "predict yanıtı")

    def test_input_keys_without_memory(self):
        """Bellek yokken input_keys şablondaki değişkenlerden oluşmalı."""
        llm = _mock_llm()
        prompt = PromptTemplate.from_template("{a} ve {b}")
        chain = LLMChain(llm=llm, prompt=prompt)
        self.assertIn("a", chain.input_keys)
        self.assertIn("b", chain.input_keys)

    def test_input_keys_excludes_memory_variables(self):
        """Bellek değişkenleri input_keys listesinden çıkarılmalı."""
        from safechain.memory.buffer import ConversationBufferMemory
        llm = _mock_llm()
        prompt = PromptTemplate.from_template("{history} {input}")
        memory = ConversationBufferMemory(memory_key="history")
        chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
        self.assertIn("input", chain.input_keys)
        self.assertNotIn("history", chain.input_keys)

    def test_output_keys(self):
        """Varsayılan output_keys ['text'] olmalı."""
        llm = _mock_llm()
        prompt = PromptTemplate.from_template("{x}")
        chain = LLMChain(llm=llm, prompt=prompt)
        self.assertEqual(chain.output_keys, ["text"])

    def test_memory_is_updated_after_invoke(self):
        """Invoke sonrasında belleğe hem kullanıcı hem bot mesajı eklenmeli."""
        from safechain.memory.buffer import ConversationBufferMemory
        llm = _mock_llm("bot cevabı")
        prompt = PromptTemplate.from_template("{history}\n{input}")
        memory = ConversationBufferMemory(memory_key="history")
        chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
        chain.invoke({"input": "kullanıcı sorusu"})
        self.assertEqual(len(memory.messages), 2)
        self.assertEqual(memory.messages[1].content, "bot cevabı")


class TestSimpleSequentialChain(unittest.TestCase):
    def _make_chain(self, response):
        llm = _mock_llm(response)
        prompt = PromptTemplate.from_template("{text}")
        return LLMChain(llm=llm, prompt=prompt, output_key="text")

    def test_two_chains(self):
        """İki chain sırayla çalışmalı; ilkinin çıktısı ikinciye girdi olmalı."""
        llm1 = _mock_llm("ara çıktı")
        llm2 = _mock_llm("nihai çıktı")
        c1 = LLMChain(llm=llm1, prompt=PromptTemplate.from_template("{text}"), output_key="text")
        c2 = LLMChain(llm=llm2, prompt=PromptTemplate.from_template("{text}"), output_key="text")
        seq = SimpleSequentialChain(chains=[c1, c2])
        result = seq.invoke({"text": "başlangıç"})
        self.assertEqual(result["text"], "nihai çıktı")

    def test_input_keys_from_first_chain(self):
        """Zincirin input_keys'i ilk zincirden alınmalı."""
        c1 = self._make_chain("a")
        c2 = self._make_chain("b")
        seq = SimpleSequentialChain(chains=[c1, c2])
        self.assertEqual(seq.input_keys, c1.input_keys)

    def test_output_keys_from_last_chain(self):
        """Zincirin output_keys'i son zincirden alınmalı."""
        c1 = self._make_chain("a")
        c2 = self._make_chain("b")
        seq = SimpleSequentialChain(chains=[c1, c2])
        self.assertEqual(seq.output_keys, c2.output_keys)

    def test_pipe_syntax(self):
        """| operatörü SimpleSequentialChain oluşturmalı."""
        c1 = self._make_chain("ara")
        c2 = self._make_chain("son")
        seq = c1 | c2
        self.assertIsInstance(seq, SimpleSequentialChain)

    def test_verbose_does_not_break(self):
        """verbose=True modu hatasız çalışmalı."""
        import io, sys
        c1 = self._make_chain("x")
        c2 = self._make_chain("y")
        seq = SimpleSequentialChain(chains=[c1, c2], verbose=True)
        captured = io.StringIO()
        sys.stdout = captured
        result = seq.invoke({"text": "giriş"})
        sys.stdout = sys.__stdout__
        self.assertIn("text", result)


class TestSequentialChain(unittest.TestCase):
    def test_explicit_io_variables(self):
        """Açık giriş/çıkış değişkenleri doğru yönlendirilmeli."""
        llm1 = _mock_llm("çevrilmiş metin")
        llm2 = _mock_llm("özetlenmiş metin")
        c1 = LLMChain(
            llm=llm1,
            prompt=PromptTemplate.from_template("{metin}"),
            output_key="çeviri",
        )
        c2 = LLMChain(
            llm=llm2,
            prompt=PromptTemplate.from_template("{çeviri}"),
            output_key="özet",
        )
        seq = SequentialChain(
            chains=[c1, c2],
            input_variables=["metin"],
            output_variables=["özet"],
        )
        result = seq.invoke({"metin": "orijinal metin"})
        self.assertIn("özet", result)
        self.assertNotIn("çeviri", result)

    def test_input_output_keys(self):
        """input_variables ve output_variables doğru atanmalı."""
        c1 = LLMChain(
            llm=_mock_llm("x"),
            prompt=PromptTemplate.from_template("{a}"),
            output_key="b",
        )
        seq = SequentialChain(
            chains=[c1],
            input_variables=["a"],
            output_variables=["b"],
        )
        self.assertEqual(seq.input_keys, ["a"])
        self.assertEqual(seq.output_keys, ["b"])


if __name__ == "__main__":
    unittest.main()
