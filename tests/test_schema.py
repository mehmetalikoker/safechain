"""safechain.schema modülü testleri."""
import unittest
from safechain.schema import Document, Generation, Message


class TestMessage(unittest.TestCase):
    def test_basic_creation(self):
        """Temel rol ve içerikle Message nesnesi oluşturulabilmeli."""
        msg = Message(role="user", content="merhaba")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "merhaba")

    def test_roles(self):
        """Geçerli tüm roller (system, user, assistant, tool) kabul edilmeli."""
        for role in ("system", "user", "assistant", "tool"):
            msg = Message(role=role, content="test")
            self.assertEqual(msg.role, role)

    def test_content_as_list(self):
        """content alanı liste (çok-bloklı) içerik kabul etmeli."""
        blocks = [{"type": "text", "text": "hello"}, {"type": "tool_use", "id": "1"}]
        msg = Message(role="assistant", content=blocks)
        self.assertIsInstance(msg.content, list)
        self.assertEqual(len(msg.content), 2)

    def test_equality(self):
        """Aynı rol ve içeriğe sahip iki Message eşit olmalı."""
        a = Message(role="user", content="hi")
        b = Message(role="user", content="hi")
        self.assertEqual(a, b)

    def test_inequality(self):
        """Farklı role sahip iki Message eşit olmamalı."""
        a = Message(role="user", content="hi")
        b = Message(role="assistant", content="hi")
        self.assertNotEqual(a, b)


class TestDocument(unittest.TestCase):
    def test_basic_creation(self):
        """Boş metadata ile Document oluşturulabilmeli."""
        doc = Document(page_content="test metni")
        self.assertEqual(doc.page_content, "test metni")
        self.assertEqual(doc.metadata, {})

    def test_with_metadata(self):
        """Metadata sözlüğü doğru şekilde saklanmalı."""
        doc = Document(page_content="içerik", metadata={"source": "dosya.txt", "row": 3})
        self.assertEqual(doc.metadata["source"], "dosya.txt")
        self.assertEqual(doc.metadata["row"], 3)

    def test_metadata_default_is_independent(self):
        """Her Document örneğinin metadata'sı bağımsız olmalı."""
        doc1 = Document(page_content="a")
        doc2 = Document(page_content="b")
        doc1.metadata["key"] = "val"
        self.assertNotIn("key", doc2.metadata)


class TestGeneration(unittest.TestCase):
    def test_basic_creation(self):
        """Boş generation_info ile Generation oluşturulabilmeli."""
        gen = Generation(text="yanıt metni")
        self.assertEqual(gen.text, "yanıt metni")
        self.assertEqual(gen.generation_info, {})

    def test_with_info(self):
        """generation_info sözlüğü doğru şekilde saklanmalı."""
        info = {"stop_reason": "end_turn", "usage": {"input_tokens": 10}}
        gen = Generation(text="cevap", generation_info=info)
        self.assertEqual(gen.generation_info["stop_reason"], "end_turn")

    def test_generation_info_default_independent(self):
        """Her Generation örneğinin generation_info'su bağımsız olmalı."""
        g1 = Generation(text="a")
        g2 = Generation(text="b")
        g1.generation_info["x"] = 1
        self.assertNotIn("x", g2.generation_info)


if __name__ == "__main__":
    unittest.main()
