"""safechain.prompts modülü testleri."""
import unittest
from safechain.prompts.template import (
    AIMessage,
    ChatPromptTemplate,
    HumanMessage,
    PromptTemplate,
    SystemMessage,
)
from safechain.schema import Message


class TestPromptTemplate(unittest.TestCase):
    def test_format(self):
        """Şablon değişkenleri doğru yerleştirilmeli."""
        pt = PromptTemplate(template="{isim} kaç yaşında?")
        result = pt.format(isim="Ali")
        self.assertEqual(result, "Ali kaç yaşında?")

    def test_auto_input_variables(self):
        """Şablondaki değişkenler otomatik algılanmalı."""
        pt = PromptTemplate(template="{a} ve {b}")
        self.assertIn("a", pt.input_variables)
        self.assertIn("b", pt.input_variables)

    def test_explicit_input_variables(self):
        """Açıkça verilen input_variables listesi korunmalı."""
        pt = PromptTemplate(template="{a} ve {b}", input_variables=["a"])
        self.assertEqual(pt.input_variables, ["a"])

    def test_from_template(self):
        """from_template sınıf metodu değişkenleri otomatik çıkarmalı."""
        pt = PromptTemplate.from_template("Merhaba {isim}!")
        self.assertIn("isim", pt.input_variables)
        self.assertEqual(pt.format(isim="Dünya"), "Merhaba Dünya!")

    def test_no_variables(self):
        """Değişken içermeyen şablon sabit metin döndürmeli."""
        pt = PromptTemplate(template="sabit metin")
        self.assertEqual(pt.input_variables, [])
        self.assertEqual(pt.format(), "sabit metin")

    def test_pipe_operator_returns_llmchain(self):
        """| operatörü prompt ve LLM'i birleştirerek LLMChain oluşturmalı."""
        from unittest.mock import MagicMock
        from safechain.chains.llm_chain import LLMChain
        pt = PromptTemplate.from_template("{q}")
        llm = MagicMock()
        chain = pt | llm
        self.assertIsInstance(chain, LLMChain)
        self.assertIs(chain.llm, llm)
        self.assertIs(chain.prompt, pt)


class TestMessageHelpers(unittest.TestCase):
    def test_system_message(self):
        """SystemMessage şablonu system rolünde Message üretmeli."""
        tmpl = SystemMessage("Sen yardımsever bir {rol}sun.")
        msg = tmpl.format(rol="asistan")
        self.assertIsInstance(msg, Message)
        self.assertEqual(msg.role, "system")
        self.assertEqual(msg.content, "Sen yardımsever bir asistansun.")

    def test_human_message(self):
        """HumanMessage şablonu user rolünde Message üretmeli."""
        tmpl = HumanMessage("{soru}")
        msg = tmpl.format(soru="Nasılsın?")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Nasılsın?")

    def test_ai_message(self):
        """AIMessage şablonu assistant rolünde Message üretmeli."""
        tmpl = AIMessage("İyiyim, {teşekkür}!")
        msg = tmpl.format(teşekkür="teşekkürler")
        self.assertEqual(msg.role, "assistant")


class TestChatPromptTemplate(unittest.TestCase):
    def test_from_tuples(self):
        """Tuple listesinden ChatPromptTemplate oluşturulup formatlanabilmeli."""
        tmpl = ChatPromptTemplate.from_messages([
            ("system", "Sen {görev} yapan bir botsun."),
            ("user", "{soru}"),
        ])
        msgs = tmpl.format_messages(görev="çeviri", soru="Bu ne demek?")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[1].role, "user")
        self.assertIn("çeviri", msgs[0].content)

    def test_from_message_templates(self):
        """Mesaj şablon nesnelerinden ChatPromptTemplate oluşturulabilmeli."""
        tmpl = ChatPromptTemplate.from_messages([
            SystemMessage("Sistem: {s}"),
            HumanMessage("Kullanıcı: {h}"),
        ])
        msgs = tmpl.format_messages(s="talimat", h="soru")
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[1].role, "user")

    def test_input_variables_collected(self):
        """Tüm mesaj şablonlarındaki değişkenler toplanmalı."""
        tmpl = ChatPromptTemplate.from_messages([
            ("system", "Dil: {dil}"),
            ("user", "{metin}"),
        ])
        self.assertIn("dil", tmpl.input_variables)
        self.assertIn("metin", tmpl.input_variables)

    def test_format_returns_string(self):
        """format metodu mesajları birleşik metin olarak döndürmeli."""
        tmpl = ChatPromptTemplate.from_messages([("user", "{q}")])
        text = tmpl.format(q="test")
        self.assertIn("user:", text)
        self.assertIn("test", text)

    def test_pipe_operator(self):
        """| operatörü ChatPromptTemplate ve LLM'i birleştirmeli."""
        from unittest.mock import MagicMock
        from safechain.chains.llm_chain import LLMChain
        tmpl = ChatPromptTemplate.from_messages([("user", "{q}")])
        llm = MagicMock()
        chain = tmpl | llm
        self.assertIsInstance(chain, LLMChain)


if __name__ == "__main__":
    unittest.main()
