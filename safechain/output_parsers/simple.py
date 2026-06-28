from __future__ import annotations
import re
from typing import List

from safechain.output_parsers.base import BaseOutputParser


class StrOutputParser(BaseOutputParser[str]):
    """LLM çıktısını olduğu gibi string olarak döner.

    Herhangi bir dönüşüm uygulamaz; ``prompt | llm | StrOutputParser()``
    kalıbında zincirin sonundaki tip dönüşümünü tamamlamak için kullanılır.
    """

    def parse(self, text: str) -> str:
        """Metni baştaki ve sondaki boşluklardan temizleyerek döner.

        Args:
            text: LLM'in ürettiği ham metin.

        Returns:
            strip() uygulanmış metin.
        """
        return text.strip()

    def get_format_instructions(self) -> str:
        return ""


class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """Virgülle ayrılmış LLM çıktısını Python listesine dönüştürür.

    Örnek LLM çıktısı: ``"elma, armut, kiraz"``
    Sonuç: ``["elma", "armut", "kiraz"]``
    """

    def parse(self, text: str) -> List[str]:
        """Virgülle ayrılmış metni temizlenmiş eleman listesine dönüştürür.

        Boş elemanlar ve yalnızca boşluktan oluşan girişler filtrelenir.

        Args:
            text: Virgülle ayrılmış değerleri içeren ham metin.

        Returns:
            Boşluklardan arındırılmış string listesi.
        """
        return [item.strip() for item in text.split(",") if item.strip()]

    def get_format_instructions(self) -> str:
        """LLM'e virgülle ayrılmış liste formatını açıklar."""
        return (
            "Yanıtını virgülle ayrılmış değerler olarak ver. "
            "Örnek: 'değer1, değer2, değer3'"
        )


class NumberedListOutputParser(BaseOutputParser[List[str]]):
    """Numaralı liste biçimindeki LLM çıktısını Python listesine dönüştürür.

    Aşağıdaki formatları tanır:
    - ``1. Madde``
    - ``1) Madde``
    - ``1- Madde``
    """

    # 1. / 1) / 1- biçimlerini eşleştirir
    _PATTERN = re.compile(r"^\s*\d+[.\-\)]\s*", re.MULTILINE)

    def parse(self, text: str) -> List[str]:
        """Numaralandırılmış satırları temizleyerek liste döner.

        Numarasız ve boş satırlar atlanır.

        Args:
            text: Numaralı liste içeren ham metin.

        Returns:
            Numara önekleri kaldırılmış string listesi.
        """
        items: List[str] = []
        for line in text.splitlines():
            cleaned = self._PATTERN.sub("", line).strip()
            if cleaned:
                items.append(cleaned)
        return items

    def get_format_instructions(self) -> str:
        """LLM'e numaralı liste formatını açıklar."""
        return (
            "Yanıtını numaralı liste olarak ver. "
            "Her maddeyi yeni satıra yaz. Örnek:\n"
            "1. Birinci madde\n"
            "2. İkinci madde\n"
            "3. Üçüncü madde"
        )
