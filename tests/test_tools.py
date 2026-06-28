"""safechain.tools modülü testleri."""
import unittest
from safechain.tools.base import Tool, _infer_json_schema, tool


def toplama(x: int, y: int) -> int:
    """İki sayıyı toplar."""
    return x + y


def selamlama(isim: str, resmi: bool = False) -> str:
    if resmi:
        return f"Sayın {isim}"
    return f"Merhaba {isim}"


class TestInferJsonSchema(unittest.TestCase):
    def test_basic_types(self):
        """Python tip ipuçlarından JSON Schema tipi doğru çıkarılmalı."""
        schema = _infer_json_schema(toplama)
        self.assertEqual(schema["type"], "object")
        self.assertIn("x", schema["properties"])
        self.assertIn("y", schema["properties"])
        self.assertEqual(schema["properties"]["x"]["type"], "integer")
        self.assertEqual(schema["properties"]["y"]["type"], "integer")

    def test_required_fields(self):
        """Varsayılan değeri olmayan parametreler required listesinde yer almalı."""
        schema = _infer_json_schema(toplama)
        self.assertIn("x", schema["required"])
        self.assertIn("y", schema["required"])

    def test_optional_not_required(self):
        """Varsayılan değeri olan parametreler required dışında tutulmalı."""
        schema = _infer_json_schema(selamlama)
        self.assertIn("isim", schema["required"])
        self.assertNotIn("resmi", schema["required"])

    def test_string_type(self):
        """str tipi 'string' JSON Schema tipine dönüştürülmeli."""
        schema = _infer_json_schema(selamlama)
        self.assertEqual(schema["properties"]["isim"]["type"], "string")

    def test_self_excluded(self):
        """Metot 'self' parametresi şemaya dahil edilmemeli."""
        class MyClass:
            def method(self, x: int) -> int:
                return x
        schema = _infer_json_schema(MyClass().method)
        self.assertNotIn("self", schema.get("properties", {}))


class TestTool(unittest.TestCase):
    def setUp(self):
        self.tool = Tool(name="toplama", description="İki sayı toplar", func=toplama)

    def test_run(self):
        """run() metodu araç fonksiyonunu kwargs ile çağırmalı."""
        result = self.tool.run(x=3, y=4)
        self.assertEqual(result, 7)

    def test_call(self):
        """__call__ operatörü run() ile aynı sonucu vermeli."""
        result = self.tool(x=10, y=5)
        self.assertEqual(result, 15)

    def test_schema_inferred(self):
        """args_schema verilmezse şema fonksiyondan otomatik çıkarılmalı."""
        schema = self.tool.schema
        self.assertIn("x", schema["properties"])
        self.assertIn("y", schema["properties"])

    def test_custom_schema(self):
        """Özel args_schema verildiğinde bu şema kullanılmalı."""
        custom = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
        t = Tool(name="test", description="desc", func=lambda n: n, args_schema=custom)
        self.assertEqual(t.schema, custom)

    def test_to_anthropic_schema(self):
        """Anthropic tool-use şeması doğru formatta üretilmeli."""
        schema = self.tool.to_anthropic_schema()
        self.assertEqual(schema["name"], "toplama")
        self.assertEqual(schema["description"], "İki sayı toplar")
        self.assertIn("input_schema", schema)

    def test_to_openai_schema(self):
        """OpenAI function-calling şeması doğru formatta üretilmeli."""
        schema = self.tool.to_openai_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "toplama")
        self.assertIn("parameters", schema["function"])

    def test_name_and_description(self):
        """name ve description alanları doğru saklanmalı."""
        self.assertEqual(self.tool.name, "toplama")
        self.assertEqual(self.tool.description, "İki sayı toplar")


class TestToolDecorator(unittest.TestCase):
    def test_bare_decorator(self):
        """@tool dekoratörü fonksiyondan Tool nesnesi oluşturmalı."""
        @tool
        def kare(x: int) -> int:
            """Sayının karesini alır."""
            return x * x

        self.assertIsInstance(kare, Tool)
        self.assertEqual(kare.name, "kare")
        self.assertEqual(kare.run(x=4), 16)

    def test_decorator_with_params(self):
        """@tool(name=, description=) parametreleri doğru atanmalı."""
        @tool(name="abs_val", description="Mutlak değer")
        def mutlak(x: float) -> float:
            return abs(x)

        self.assertIsInstance(mutlak, Tool)
        self.assertEqual(mutlak.name, "abs_val")
        self.assertEqual(mutlak.description, "Mutlak değer")
        self.assertEqual(mutlak.run(x=-5.0), 5.0)

    def test_decorator_uses_docstring_as_description(self):
        """Açıklama verilmezse fonksiyon docstring'i description olarak kullanılmalı."""
        @tool
        def carp(x: int, y: int) -> int:
            """Çarpma işlemi yapar."""
            return x * y

        self.assertEqual(carp.description, "Çarpma işlemi yapar.")

    def test_decorator_with_custom_schema(self):
        """@tool(args_schema=) özel şema doğru uygulanmalı."""
        custom = {"type": "object", "properties": {"val": {"type": "string"}}, "required": ["val"]}

        @tool(args_schema=custom)
        def echo(val: str) -> str:
            return val

        self.assertEqual(echo.schema, custom)


if __name__ == "__main__":
    unittest.main()
