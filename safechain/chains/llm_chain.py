from __future__ import annotations
from typing import Any, Dict, List, Optional

from safechain.chains.base import Chain
from safechain.schema import Message


class LLMChain(Chain):
    """Bir prompt şablonunu LLM ile birleştiren temel zincir.

    Prompt şablonu ``PromptTemplate`` veya ``ChatPromptTemplate`` olabilir.
    İsteğe bağlı bellek (memory) desteğiyle konuşma geçmişi otomatik
    yönetilebilir.

    Attributes:
        llm: Metni üreten BaseLLM örneği.
        prompt: Girişleri biçimlendiren şablon nesnesi.
        output_key: Çıkış sözlüğünde kullanılacak anahtar adı.
        memory: Konuşma geçmişini saklayan bellek nesnesi (opsiyonel).
    """

    def __init__(
        self,
        llm: Any,
        prompt: Any,
        output_key: str = "text",
        memory: Optional[Any] = None,
    ) -> None:
        """LLMChain oluşturur.

        Args:
            llm: Yanıt üreten LLM nesnesi (BaseLLM örneği).
            prompt: Giriş değişkenlerini metne dönüştüren şablon.
                    ``format_messages`` metodu varsa chat prompt,
                    yoksa ``format`` metodu çağrılır.
            output_key: Çıkış sözlüğündeki anahtar adı.
                        Varsayılan: "text".
            memory: Konuşma geçmişini yönetecek bellek nesnesi.
                    ``None`` ise bellek desteği etkin değildir.
        """
        self.llm = llm
        self.prompt = prompt
        self.output_key = output_key
        self.memory = memory

    @property
    def input_keys(self) -> List[str]:
        """Kullanıcıdan beklenen giriş anahtarları.

        Şablonun ``input_variables`` listesinden türetilir; bellek
        tarafından sağlanan değişkenler çıkarılır (kullanıcı tarafından
        tekrar verilmesine gerek yoktur).

        Returns:
            Kullanıcının sağlaması gereken anahtar isimleri.
        """
        keys = list(getattr(self.prompt, "input_variables", []))
        if self.memory:
            mem_vars = set(getattr(self.memory, "memory_variables", []))
            keys = [k for k in keys if k not in mem_vars]
        return keys

    @property
    def output_keys(self) -> List[str]:
        """Zincirin ürettiği çıkış anahtarları.

        Returns:
            ``[output_key]`` listesi (tek elemanlı).
        """
        return [self.output_key]

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prompt şablonunu doldurur, LLM'i çağırır ve çıktıyı döner.

        Bellek etkinse geçmiş konuşmalar giriş değişkenlerine eklenir ve
        yeni tur tamamlandıktan sonra belleğe kaydedilir.

        Args:
            inputs: Kullanıcının sağladığı giriş anahtar-değer çiftleri.

        Returns:
            ``{output_key: üretilen_metin}`` biçiminde çıkış sözlüğü.
        """
        all_inputs = dict(inputs)
        if self.memory:
            all_inputs.update(self.memory.load_memory_variables(inputs))

        if hasattr(self.prompt, "format_messages"):
            messages = self.prompt.format_messages(**all_inputs)
            generation = self.llm.generate(messages)
        else:
            prompt_str = self.prompt.format(**all_inputs)
            generation = self.llm.generate([Message(role="user", content=prompt_str)])

        output = generation.text

        if self.memory:
            human_key = self.input_keys[0] if self.input_keys else "input"
            self.memory.save_context(
                {"input": inputs.get(human_key, "")},
                {"output": output},
            )

        return {self.output_key: output}

    def predict(self, **kwargs: Any) -> str:
        """Keyword argümanlarla zinciri çağırır ve metin yanıtı döner.

        ``invoke`` metodunun sarmalayıcısıdır; çıkış sözlüğü yerine
        doğrudan metin döner.

        Args:
            **kwargs: Şablonun beklediği giriş anahtar-değer çiftleri.

        Returns:
            LLM'in ürettiği metin yanıtı.
        """
        return self.invoke(kwargs)[self.output_key]
