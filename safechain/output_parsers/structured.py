from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from safechain.output_parsers.base import BaseOutputParser, OutputParserException


@dataclass
class ResponseSchema:
    """StructuredOutputParser için tek bir alan tanımı.

    Attributes:
        name: Alanın adı (çıkış sözlüğünde anahtar olarak kullanılır).
        description: LLM'e bu alanın ne içermesi gerektiğini açıklar.
        type: Beklenen veri tipi. Varsayılan: ``"string"``.
    """

    name: str
    description: str
    type: str = "string"


class StructuredOutputParser(BaseOutputParser[Dict[str, Any]]):
    """Belirli alanları içeren yapılandırılmış JSON çıktısını ayrıştırır.

    ``ResponseSchema`` nesneleriyle istenen alanlar tanımlanır; LLM bu
    alanlara uygun JSON üretmesi için talimatlandırılır ve çıktı
    doğrulanarak ayrıştırılır.

    Kullanım::

        schemas = [
            ResponseSchema(name="özet", description="Metnin kısa özeti"),
            ResponseSchema(name="dil", description="Metnin dili", type="string"),
        ]
        parser = StructuredOutputParser.from_response_schemas(schemas)
        prompt = PromptTemplate(
            template="{text}\\n\\n{format_instructions}",
            input_variables=["text", "format_instructions"],
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.invoke({
            "text": "...",
            "format_instructions": parser.get_format_instructions(),
        })
        parsed = parser.parse(result["text"])

    Attributes:
        response_schemas: Beklenen alan tanımlarının listesi.
    """

    def __init__(self, response_schemas: List[ResponseSchema]) -> None:
        """StructuredOutputParser oluşturur.

        Args:
            response_schemas: Her biri bir çıkış alanını tanımlayan
                              ResponseSchema nesnelerinin listesi.
        """
        self.response_schemas = response_schemas

    @classmethod
    def from_response_schemas(
        cls, response_schemas: List[ResponseSchema]
    ) -> "StructuredOutputParser":
        """ResponseSchema listesinden parser oluşturur.

        Args:
            response_schemas: Alan tanımlarının listesi.

        Returns:
            Yeni StructuredOutputParser örneği.
        """
        return cls(response_schemas=response_schemas)

    def parse(self, text: str) -> Dict[str, Any]:
        """Metinden tanımlı alanları çıkararak sözlük döner.

        Önce JSON ayrıştırma dener; başarısız olursa her alan için
        metin içinde anahtar:değer araması yapar.

        Args:
            text: LLM'in ürettiği ham metin.

        Returns:
            Alan adları → değerler biçiminde sözlük.

        Raises:
            OutputParserException: Hiçbir alan çıkarılamazsa fırlatılır.
        """
        # Markdown kod bloğunu temizle
        cleaned = re.sub(r"```(?:json)?\s*\n?", "", text).replace("```", "").strip()

        # JSON ayrıştırmayı dene
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return {s.name: data.get(s.name, "") for s in self.response_schemas}
        except json.JSONDecodeError:
            pass

        # JSON başarısız → anahtar:değer satırı ara
        result: Dict[str, Any] = {}
        for schema in self.response_schemas:
            pattern = re.compile(
                rf'"{re.escape(schema.name)}"\s*:\s*"([^"]*)"'
                rf'|"{re.escape(schema.name)}"\s*:\s*([^\n,\}}]+)',
                re.IGNORECASE,
            )
            match = pattern.search(text)
            if match:
                result[schema.name] = (match.group(1) or match.group(2) or "").strip()
            else:
                result[schema.name] = ""

        if not any(result.values()):
            raise OutputParserException(
                f"Yapılandırılmış çıktı ayrıştırılamadı.\nMetin: {text!r}"
            )
        return result

    def get_format_instructions(self) -> str:
        """LLM'e hangi alanları, hangi formatta döndürmesi gerektiğini açıklar.

        Returns:
            Her alan için açıklama içeren JSON format talimatı.
        """
        schema_desc = "\n".join(
            f'  "{s.name}": <{s.type}> — {s.description}'
            for s in self.response_schemas
        )
        example = json.dumps(
            {s.name: f"<{s.type}>" for s in self.response_schemas},
            ensure_ascii=False,
            indent=2,
        )
        return (
            "Yanıtını aşağıdaki alanları içeren JSON nesnesi olarak ver:\n"
            f"{schema_desc}\n\n"
            "Format:\n"
            f"```json\n{example}\n```"
        )
