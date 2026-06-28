from __future__ import annotations
import json
import re
from typing import Any, Dict, Optional

from safechain.output_parsers.base import BaseOutputParser, OutputParserException


class JSONOutputParser(BaseOutputParser[Dict[str, Any]]):
    """LLM çıktısından JSON nesnesini çıkarır ve ayrıştırır.

    Aşağıdaki formatları destekler:
    - Düz JSON: ``{"anahtar": "değer"}``
    - Markdown kod bloğu: ````json\\n{...}\\n` ``` ``

    Attributes:
        encoding_error: ``"strict"`` (varsayılan) veya ``"ignore"``.
                        Geçersiz JSON karakterlerini yönetir.
    """

    def __init__(self, encoding_error: str = "strict") -> None:
        """JSONOutputParser oluşturur.

        Args:
            encoding_error: JSON ayrıştırma hatasında davranış.
                            ``"strict"`` hata fırlatır, ``"ignore"`` atlar.
        """
        self.encoding_error = encoding_error

    def parse(self, text: str) -> Dict[str, Any]:
        """Metinden JSON nesnesini çıkarır ve dict olarak döner.

        Önce ````json ... ` ``` `` markdown bloğunu arar, bulamazsa tüm
        metni JSON olarak ayrıştırmayı dener. Çıkarılan JSON bir liste
        ise ``{"items": [...]}`` biçiminde sarmalar.

        Args:
            text: LLM'in ürettiği ham metin.

        Returns:
            Ayrıştırılmış JSON sözlüğü.

        Raises:
            OutputParserException: Geçerli JSON bulunamazsa fırlatılır.
        """
        json_str = self._extract_json(text.strip())
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise OutputParserException(
                f"Geçerli JSON ayrıştırılamadı.\n"
                f"Hata: {exc}\n"
                f"Metin: {text!r}"
            ) from exc

        if isinstance(parsed, list):
            return {"items": parsed}
        if not isinstance(parsed, dict):
            return {"value": parsed}
        return parsed

    def _extract_json(self, text: str) -> str:
        """Metinden JSON içeriğini ayıklar.

        Markdown ````json``` veya ``````` bloğunu arar; bulamazsa
        ilk ``{`` veya ``[`` karakterinden son ``}`` veya ``]``'a kadar
        olan bölümü keser.

        Args:
            text: İçinde JSON bulunan ham metin.

        Returns:
            Temizlenmiş JSON string'i.
        """
        # Markdown kod bloğu
        md_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
        if md_match:
            return md_match.group(1).strip()

        # İlk { veya [ ile son } veya ] arasını al
        start = min(
            (text.find(c) for c in ("{", "[") if c in text),
            default=-1,
        )
        if start != -1:
            end = max(text.rfind("}"), text.rfind("]"))
            if end != -1:
                return text[start : end + 1]

        return text

    def get_format_instructions(self) -> str:
        """LLM'e JSON formatını açıklar."""
        return (
            "Yanıtını geçerli bir JSON nesnesi olarak ver. "
            "Markdown kod bloğu kullanabilirsin:\n"
            "```json\n"
            '{"anahtar": "değer"}\n'
            "```"
        )
