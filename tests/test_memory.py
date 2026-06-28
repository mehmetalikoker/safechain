"""safechain.memory modülü testleri."""
import unittest
from safechain.memory.buffer import ConversationBufferMemory, ConversationBufferWindowMemory
from safechain.schema import Message


class TestConversationBufferMemory(unittest.TestCase):
    def setUp(self):
        self.mem = ConversationBufferMemory()

    def test_memory_variables(self):
        """Varsayılan memory_key 'history' olmalı."""
        self.assertEqual(self.mem.memory_variables, ["history"])

    def test_custom_memory_key(self):
        """Özel memory_key doğru şekilde saklanmalı."""
        mem = ConversationBufferMemory(memory_key="chat_history")
        self.assertEqual(mem.memory_variables, ["chat_history"])

    def test_empty_load(self):
        """Boş bellekte history boş string döndürmeli."""
        result = self.mem.load_memory_variables({})
        self.assertEqual(result["history"], "")

    def test_save_and_load_text(self):
        """Kaydedilen konuşma geçmişi metin olarak doğru yüklenmelidir."""
        self.mem.save_context({"input": "Merhaba"}, {"output": "Nasılsın?"})
        result = self.mem.load_memory_variables({})
        history = result["history"]
        self.assertIn("Merhaba", history)
        self.assertIn("Nasılsın?", history)
        self.assertIn("Human:", history)
        self.assertIn("AI:", history)

    def test_custom_prefixes(self):
        """Özel human ve ai prefiksleri history metninde görünmeli."""
        mem = ConversationBufferMemory(human_prefix="Kullanıcı", ai_prefix="Bot")
        mem.save_context({"input": "hi"}, {"output": "hello"})
        result = mem.load_memory_variables({})
        self.assertIn("Kullanıcı:", result["history"])
        self.assertIn("Bot:", result["history"])

    def test_return_messages_mode(self):
        """return_messages=True iken Message nesneleri listesi döndürmeli."""
        mem = ConversationBufferMemory(return_messages=True)
        mem.save_context({"input": "soru"}, {"output": "cevap"})
        result = mem.load_memory_variables({})
        msgs = result["history"]
        self.assertIsInstance(msgs, list)
        self.assertEqual(len(msgs), 2)
        self.assertIsInstance(msgs[0], Message)
        self.assertEqual(msgs[0].role, "user")
        self.assertEqual(msgs[1].role, "assistant")

    def test_multiple_turns(self):
        """Çoklu konuşma turları sırayla saklanmalı."""
        self.mem.save_context({"input": "1"}, {"output": "a"})
        self.mem.save_context({"input": "2"}, {"output": "b"})
        self.mem.save_context({"input": "3"}, {"output": "c"})
        self.assertEqual(len(self.mem.messages), 6)

    def test_messages_returns_copy(self):
        """messages özelliği değiştirilemez kopya döndürmeli."""
        self.mem.save_context({"input": "x"}, {"output": "y"})
        copy = self.mem.messages
        copy.append(Message(role="user", content="extra"))
        self.assertEqual(len(self.mem.messages), 2)

    def test_clear(self):
        """clear() metodu bellekten tüm mesajları silmeli."""
        self.mem.save_context({"input": "a"}, {"output": "b"})
        self.mem.clear()
        self.assertEqual(self.mem.messages, [])
        result = self.mem.load_memory_variables({})
        self.assertEqual(result["history"], "")

    def test_save_context_uses_first_value_as_fallback(self):
        """input/output anahtarı yoksa sözlükteki ilk değer kullanılmalı."""
        self.mem.save_context({"query": "deneme"}, {"response": "tamam"})
        result = self.mem.load_memory_variables({})
        self.assertIn("deneme", result["history"])
        self.assertIn("tamam", result["history"])


class TestConversationBufferWindowMemory(unittest.TestCase):
    def test_keeps_last_k_turns(self):
        """Yalnızca son k tur bellekte tutulmalı, eskiler silinmeli."""
        mem = ConversationBufferWindowMemory(k=2)
        for i in range(5):
            mem.save_context({"input": f"soru{i}"}, {"output": f"cevap{i}"})
        result = mem.load_memory_variables({})
        history = result["history"]
        self.assertIn("soru3", history)
        self.assertIn("soru4", history)
        self.assertNotIn("soru0", history)
        self.assertNotIn("soru1", history)

    def test_return_messages_mode(self):
        """return_messages=True iken son k turun mesajları döndürülmeli."""
        mem = ConversationBufferWindowMemory(k=1, return_messages=True)
        mem.save_context({"input": "eski"}, {"output": "eski_cevap"})
        mem.save_context({"input": "yeni"}, {"output": "yeni_cevap"})
        result = mem.load_memory_variables({})
        msgs = result["history"]
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].content, "yeni")

    def test_k_larger_than_history(self):
        """k, geçmiş sayısından büyükse tüm geçmiş döndürülmeli."""
        mem = ConversationBufferWindowMemory(k=10)
        mem.save_context({"input": "a"}, {"output": "b"})
        result = mem.load_memory_variables({})
        self.assertIn("a", result["history"])

    def test_inherits_clear(self):
        """clear() metodu Window belleğinde de çalışmalı."""
        mem = ConversationBufferWindowMemory(k=3)
        mem.save_context({"input": "x"}, {"output": "y"})
        mem.clear()
        self.assertEqual(mem.messages, [])


if __name__ == "__main__":
    unittest.main()
