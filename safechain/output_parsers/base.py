from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

T = TypeVar("T")


class BaseOutputParser(ABC, Generic[T]):
    """Tüm output parser'lar için ortak arayüz.

    Parser'lar LLM'in ürettiği ham metni yapılandırılmış Python
    nesnelerine dönüştürür. ``|`` operatörü ile zincire bağlanır:

        chain = prompt | llm | parser

    Alt sınıflar yalnızca ``parse`` metodunu uygulamak zorundadır.
    """

    @abstractmethod
    def parse(self, text: str) -> T:
        """Ham metin çıktısını ayrıştırır.

        Args:
            text: LLM'in ürettiği ham metin.

        Returns:
            Ayrıştırılmış Python nesnesi (tip alt sınıfa göre değişir).

        Raises:
            OutputParserException: Metin beklenen formata uymazsa.
        """

    def get_format_instructions(self) -> str:
        """LLM'e prompt'ta eklenecek format talimatını döner.

        Returns:
            Modelin hangi formatta yanıt vermesi gerektiğini açıklayan metin.
            Varsayılan: boş string (format gerektirmeyen parser'lar için).
        """
        return ""

    def __ror__(self, other: Any) -> "ParsedChain":
        """``chain | parser`` sözdizimini etkinleştirir.

        Args:
            other: Solundaki Chain nesnesi.

        Returns:
            Zincir çıktısını otomatik ayrıştıran ParsedChain.
        """
        return ParsedChain(chain=other, parser=self)

    def __call__(self, text: str) -> T:
        """``parser(metin)`` biçiminde çağrılabilir kılar.

        Args:
            text: Ayrıştırılacak ham metin.

        Returns:
            ``parse`` metodunun dönüş değeri.
        """
        return self.parse(text)


class OutputParserException(ValueError):
    """Parser, LLM çıktısını ayrıştıramazsa fırlatılır."""


class ParsedChain:
    """Bir Chain'in çıktısını otomatik olarak parser'dan geçirir.

    ``chain | parser`` ifadesi bu nesneyi üretir. Hem ``invoke`` hem de
    ``run`` arayüzlerine sahiptir; mevcut Chain altyapısıyla uyumludur.

    Attributes:
        chain: Asıl işi yapan Chain nesnesi.
        parser: Çıktıyı dönüştürecek BaseOutputParser örneği.
    """

    def __init__(self, chain: Any, parser: BaseOutputParser) -> None:
        """ParsedChain oluşturur.

        Args:
            chain: Sarmalanacak Chain nesnesi.
            parser: Çıktıyı ayrıştıracak parser.
        """
        self.chain = chain
        self.parser = parser

    @property
    def input_keys(self) -> List[str]:
        """İç zincirin giriş anahtarlarını iletir."""
        return getattr(self.chain, "input_keys", [])

    @property
    def output_keys(self) -> List[str]:
        """Ayrıştırılmış çıktı için tek anahtar döner."""
        return ["output"]

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Zinciri çalıştırır ve çıktıyı parser'dan geçirir.

        Args:
            inputs: İç zincire iletilecek giriş sözlüğü.

        Returns:
            ``{"output": ayrıştırılmış_değer}`` biçiminde sözlük.
        """
        result = self.chain.invoke(inputs)
        out_key = (
            self.chain.output_keys[0]
            if getattr(self.chain, "output_keys", None)
            else next(iter(result))
        )
        text = result.get(out_key, "")
        return {"output": self.parser.parse(str(text))}

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Zinciri çalıştırır ve ayrıştırılmış değeri doğrudan döner.

        Args:
            *args: İlk positional argüman giriş değeri olarak kullanılır.
            **kwargs: Giriş anahtar-değer çiftleri.

        Returns:
            Parser'ın ürettiği Python nesnesi.
        """
        if args and not kwargs:
            key = self.input_keys[0] if self.input_keys else "input"
            inputs: Dict[str, Any] = {key: args[0]}
        else:
            inputs = dict(kwargs)
        return self.invoke(inputs)["output"]

    def __call__(self, inputs: Any, **kwargs: Any) -> Dict[str, Any]:
        """``parsed_chain(inputs)`` biçiminde çağrılabilir kılar."""
        if isinstance(inputs, str):
            key = self.input_keys[0] if self.input_keys else "input"
            inputs = {key: inputs}
        return self.invoke({**inputs, **kwargs})

    def __or__(self, other: Any) -> "ParsedChain":
        """``parsed_chain | başka_parser`` zincirlemeyi destekler."""
        if isinstance(other, BaseOutputParser):
            return ParsedChain(chain=self, parser=other)
        from safechain.chains.sequential import SimpleSequentialChain
        return SimpleSequentialChain(chains=[self, other])  # type: ignore[list-item]
