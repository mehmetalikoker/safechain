"""safechain.output_parsers modülü testleri."""
import unittest
from unittest.mock import MagicMock

from safechain.chains.base import Chain
from safechain.output_parsers.base import (
    BaseOutputParser,
    OutputParserException,
    ParsedChain,
)
from safechain.output_parsers.simple import (
    CommaSeparatedListOutputParser,
    NumberedListOutputParser,
    StrOutputParser,
)
from safechain.output_parsers.json_parser import JSONOutputParser
from safechain.output_parsers.structured import ResponseSchema, StructuredOutputParser
from safechain.schema import Generation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FixedChain(Chain):
    """Sabit metin dönen minimal Chain — pipe operatörünü test etmek için."""

    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def input_keys(self):
        return ["input"]

    @property
    def output_keys(self):
        return ["text"]

    def invoke(self, inputs):
        return {"text": self._text}


def _mock_chain(text: str) -> _FixedChain:
    return _FixedChain(text)


# ---------------------------------------------------------------------------
# StrOutputParser
# ---------------------------------------------------------------------------

class TestStrOutputParser(unittest.TestCase):
    def setUp(self):
        self.parser = StrOutputParser()

    def test_returns_stripped_text(self):
        self.assertEqual(self.parser.parse("  merhaba  "), "merhaba")

    def test_empty_string(self):
        self.assertEqual(self.parser.parse(""), "")

    def test_multiline(self):
        result = self.parser.parse("  satır1\nsatır2  ")
        self.assertEqual(result, "satır1\nsatır2")

    def test_callable(self):
        self.assertEqual(self.parser("test"), "test")

    def test_format_instructions_empty(self):
        self.assertEqual(self.parser.get_format_instructions(), "")

    def test_pipe_creates_parsed_chain(self):
        chain = _mock_chain("yanıt")
        parsed = chain | self.parser
        self.assertIsInstance(parsed, ParsedChain)

    def test_pipe_invoke(self):
        chain = _mock_chain("  sonuç  ")
        parsed = chain | self.parser
        result = parsed.invoke({"input": "test"})
        self.assertEqual(result["output"], "sonuç")

    def test_pipe_run(self):
        chain = _mock_chain("çıktı")
        parsed = chain | self.parser
        self.assertEqual(parsed.run("giriş"), "çıktı")


# ---------------------------------------------------------------------------
# CommaSeparatedListOutputParser
# ---------------------------------------------------------------------------

class TestCommaSeparatedListOutputParser(unittest.TestCase):
    def setUp(self):
        self.parser = CommaSeparatedListOutputParser()

    def test_basic_split(self):
        result = self.parser.parse("elma, armut, kiraz")
        self.assertEqual(result, ["elma", "armut", "kiraz"])

    def test_strips_whitespace(self):
        result = self.parser.parse("  a  ,  b  ,  c  ")
        self.assertEqual(result, ["a", "b", "c"])

    def test_no_spaces(self):
        result = self.parser.parse("bir,iki,üç")
        self.assertEqual(result, ["bir", "iki", "üç"])

    def test_empty_string(self):
        result = self.parser.parse("")
        self.assertEqual(result, [])

    def test_single_item(self):
        result = self.parser.parse("yalnızca")
        self.assertEqual(result, ["yalnızca"])

    def test_trailing_comma_ignored(self):
        result = self.parser.parse("a, b, c,")
        self.assertEqual(result, ["a", "b", "c"])

    def test_format_instructions_contains_example(self):
        instructions = self.parser.get_format_instructions()
        self.assertIn("virgül", instructions.lower())

    def test_pipe_with_chain(self):
        chain = _mock_chain("python, java, go")
        parsed = chain | self.parser
        result = parsed.run("diller")
        self.assertEqual(result, ["python", "java", "go"])


# ---------------------------------------------------------------------------
# NumberedListOutputParser
# ---------------------------------------------------------------------------

class TestNumberedListOutputParser(unittest.TestCase):
    def setUp(self):
        self.parser = NumberedListOutputParser()

    def test_dot_format(self):
        text = "1. Birinci\n2. İkinci\n3. Üçüncü"
        result = self.parser.parse(text)
        self.assertEqual(result, ["Birinci", "İkinci", "Üçüncü"])

    def test_paren_format(self):
        text = "1) Alfa\n2) Beta\n3) Gama"
        result = self.parser.parse(text)
        self.assertEqual(result, ["Alfa", "Beta", "Gama"])

    def test_dash_format(self):
        text = "1- Bir\n2- İki"
        result = self.parser.parse(text)
        self.assertEqual(result, ["Bir", "İki"])

    def test_empty_lines_skipped(self):
        text = "1. Madde\n\n2. Diğer"
        result = self.parser.parse(text)
        self.assertEqual(result, ["Madde", "Diğer"])

    def test_no_numbers_in_result(self):
        result = self.parser.parse("1. Test maddesi")
        self.assertFalse(result[0].startswith("1"))

    def test_format_instructions_contains_example(self):
        instructions = self.parser.get_format_instructions()
        self.assertIn("1.", instructions)

    def test_pipe_with_chain(self):
        chain = _mock_chain("1. Python\n2. Java\n3. Go")
        parsed = chain | self.parser
        result = parsed.run("diller")
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "Python")


# ---------------------------------------------------------------------------
# JSONOutputParser
# ---------------------------------------------------------------------------

class TestJSONOutputParser(unittest.TestCase):
    def setUp(self):
        self.parser = JSONOutputParser()

    def test_plain_json(self):
        result = self.parser.parse('{"ad": "Ali", "yaş": 30}')
        self.assertEqual(result["ad"], "Ali")
        self.assertEqual(result["yaş"], 30)

    def test_markdown_json_block(self):
        text = '```json\n{"dil": "Python"}\n```'
        result = self.parser.parse(text)
        self.assertEqual(result["dil"], "Python")

    def test_markdown_block_without_json_label(self):
        text = '```\n{"key": "val"}\n```'
        result = self.parser.parse(text)
        self.assertEqual(result["key"], "val")

    def test_json_embedded_in_text(self):
        text = 'İşte sonuç: {"isim": "Veli", "puan": 95} başarılı.'
        result = self.parser.parse(text)
        self.assertEqual(result["isim"], "Veli")

    def test_list_wrapped_in_items(self):
        result = self.parser.parse("[1, 2, 3]")
        self.assertIn("items", result)
        self.assertEqual(result["items"], [1, 2, 3])

    def test_nested_json(self):
        text = '{"kullanıcı": {"ad": "Ali", "rol": "admin"}}'
        result = self.parser.parse(text)
        self.assertIsInstance(result["kullanıcı"], dict)

    def test_invalid_json_raises(self):
        with self.assertRaises(OutputParserException):
            self.parser.parse("bu geçerli json değil {{{")

    def test_format_instructions_mentions_json(self):
        instructions = self.parser.get_format_instructions()
        self.assertIn("json", instructions.lower())

    def test_pipe_with_chain(self):
        chain = _mock_chain('{"sonuç": 42}')
        parsed = chain | self.parser
        result = parsed.run("soru")
        self.assertEqual(result["sonuç"], 42)

    def test_scalar_value_wrapped(self):
        result = self.parser.parse("42")
        self.assertIn("value", result)
        self.assertEqual(result["value"], 42)


# ---------------------------------------------------------------------------
# StructuredOutputParser
# ---------------------------------------------------------------------------

class TestStructuredOutputParser(unittest.TestCase):
    def setUp(self):
        self.schemas = [
            ResponseSchema(name="özet", description="Kısa özet"),
            ResponseSchema(name="dil", description="Dil adı"),
            ResponseSchema(name="puan", description="Kalite puanı", type="number"),
        ]
        self.parser = StructuredOutputParser.from_response_schemas(self.schemas)

    def test_parse_valid_json(self):
        text = '```json\n{"özet": "İyi metin", "dil": "Türkçe", "puan": 9}\n```'
        result = self.parser.parse(text)
        self.assertEqual(result["özet"], "İyi metin")
        self.assertEqual(result["dil"], "Türkçe")

    def test_parse_plain_json(self):
        text = '{"özet": "özet metni", "dil": "İngilizce", "puan": 8}'
        result = self.parser.parse(text)
        self.assertEqual(result["dil"], "İngilizce")

    def test_all_schema_keys_present(self):
        text = '{"özet": "a", "dil": "b", "puan": 7}'
        result = self.parser.parse(text)
        for schema in self.schemas:
            self.assertIn(schema.name, result)

    def test_missing_field_returns_empty_string(self):
        text = '{"özet": "sadece özet"}'
        result = self.parser.parse(text)
        self.assertEqual(result["dil"], "")

    def test_format_instructions_contains_field_names(self):
        instructions = self.parser.get_format_instructions()
        self.assertIn("özet", instructions)
        self.assertIn("dil", instructions)
        self.assertIn("puan", instructions)

    def test_format_instructions_contains_types(self):
        instructions = self.parser.get_format_instructions()
        self.assertIn("number", instructions)

    def test_from_response_schemas_classmethod(self):
        schemas = [ResponseSchema(name="x", description="x değeri")]
        parser = StructuredOutputParser.from_response_schemas(schemas)
        self.assertIsInstance(parser, StructuredOutputParser)
        self.assertEqual(len(parser.response_schemas), 1)

    def test_pipe_with_chain(self):
        chain = _mock_chain('{"özet": "test özet", "dil": "TR", "puan": 10}')
        parsed = chain | self.parser
        result = parsed.run("giriş")
        self.assertEqual(result["dil"], "TR")


# ---------------------------------------------------------------------------
# ParsedChain
# ---------------------------------------------------------------------------

class TestParsedChain(unittest.TestCase):
    def test_input_keys_forwarded(self):
        chain = _mock_chain("x")
        parsed = ParsedChain(chain=chain, parser=StrOutputParser())
        self.assertEqual(parsed.input_keys, ["input"])

    def test_output_keys(self):
        chain = _mock_chain("x")
        parsed = ParsedChain(chain=chain, parser=StrOutputParser())
        self.assertEqual(parsed.output_keys, ["output"])

    def test_call_operator(self):
        chain = _mock_chain("merhaba")
        parsed = ParsedChain(chain=chain, parser=StrOutputParser())
        result = parsed({"input": "test"})
        self.assertEqual(result["output"], "merhaba")

    def test_chained_parsers(self):
        """İki parser'ı arka arkaya bağlama."""
        chain = _mock_chain("a, b, c")
        parsed = chain | CommaSeparatedListOutputParser()
        self.assertEqual(parsed.run("x"), ["a", "b", "c"])

    def test_top_level_import(self):
        """Ana safechain modülünden import edilebilmeli."""
        from safechain import (
            CommaSeparatedListOutputParser,
            JSONOutputParser,
            NumberedListOutputParser,
            ResponseSchema,
            StrOutputParser,
            StructuredOutputParser,
        )
        self.assertTrue(True)


# ---------------------------------------------------------------------------
# ResponseSchema dataclass
# ---------------------------------------------------------------------------

class TestResponseSchema(unittest.TestCase):
    def test_defaults(self):
        s = ResponseSchema(name="alan", description="açıklama")
        self.assertEqual(s.type, "string")

    def test_custom_type(self):
        s = ResponseSchema(name="puan", description="puan", type="number")
        self.assertEqual(s.type, "number")


if __name__ == "__main__":
    unittest.main()
