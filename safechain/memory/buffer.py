from __future__ import annotations
from typing import Any, Dict, List, Optional

from safechain.schema import Message


class ConversationBufferMemory:
    """Tüm konuşma geçmişini bellekte saklayan tampon bellek sınıfı.

    Her ``save_context`` çağrısıyla kullanıcı ve asistan mesajları iç
    listeye eklenir. ``load_memory_variables`` ile geçmiş, prompt şablonuna
    enjekte edilmek üzere düz metin veya Message listesi olarak döndürülür.
    """

    def __init__(
        self,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "history",
        return_messages: bool = False,
    ) -> None:
        """ConversationBufferMemory oluşturur.

        Args:
            human_prefix: Düz metin modunda kullanıcı satırlarının öneki.
                          Varsayılan: "Human".
            ai_prefix: Düz metin modunda asistan satırlarının öneki.
                       Varsayılan: "AI".
            memory_key: Prompt şablonuna enjekte edilecek değişken adı.
                        Varsayılan: "history".
            return_messages: ``True`` ise Message listesi, ``False`` ise
                             düz metin döner. Varsayılan: False.
        """
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.memory_key = memory_key
        self.return_messages = return_messages
        self._messages: List[Message] = []

    @property
    def memory_variables(self) -> List[str]:
        """Belleğin sağladığı değişken isimlerinin listesi.

        LLMChain bu listedeki değişkenleri kullanıcıdan istemez; bellek
        otomatik olarak sağlar.

        Returns:
            ``[memory_key]`` listesi (tek elemanlı).
        """
        return [self.memory_key]

    @property
    def messages(self) -> List[Message]:
        """Saklanan tüm mesajların kopyasını döner.

        Returns:
            İç mesaj listesinin sığ kopyası.
        """
        return list(self._messages)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Geçmiş konuşmayı şablon değişkeni olarak döner.

        ``return_messages=True`` ise Message listesi, aksi hâlde her mesajın
        ``prefix: içerik`` biçiminde birleştirildiği düz metin döner.

        Args:
            inputs: Çağrıyı tetikleyen mevcut giriş (genellikle kullanılmaz).

        Returns:
            ``{memory_key: geçmiş}`` biçiminde sözlük.
        """
        if self.return_messages:
            return {self.memory_key: list(self._messages)}
        lines = []
        for m in self._messages:
            prefix = self.human_prefix if m.role == "user" else self.ai_prefix
            lines.append(f"{prefix}: {m.content}")
        return {self.memory_key: "\n".join(lines)}

    def save_context(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> None:
        """Bir konuşma turunu belleğe kaydeder.

        Kullanıcı mesajını ("input" anahtarından veya ilk değerden) ve
        asistan yanıtını ("output" anahtarından veya ilk değerden) alarak
        sırayla ``_messages`` listesine ekler.

        Args:
            inputs: Kullanıcı mesajını içeren sözlük.
            outputs: Asistan yanıtını içeren sözlük.
        """
        human = inputs.get("input") or next(iter(inputs.values()), "")
        ai = outputs.get("output") or next(iter(outputs.values()), "")
        self._messages.append(Message(role="user", content=str(human)))
        self._messages.append(Message(role="assistant", content=str(ai)))

    def clear(self) -> None:
        """Tüm konuşma geçmişini siler.

        Belleği sıfırlamak ve konuşmayı yeniden başlatmak için kullanılır.
        """
        self._messages.clear()


class ConversationBufferWindowMemory(ConversationBufferMemory):
    """Son ``k`` konuşma turunu saklayan kayan pencere belleği.

    Tüm geçmişi saklamak yerine yalnızca en son ``k`` kullanıcı-asistan
    tur çiftini prompt'a dahil eder. Uzun konuşmalarda token maliyetini
    sınırlamak için idealdir.
    """

    def __init__(self, k: int = 5, **kwargs: Any) -> None:
        """ConversationBufferWindowMemory oluşturur.

        Args:
            k: Tutulacak maksimum konuşma turu sayısı. Her tur bir kullanıcı
               ve bir asistan mesajından oluşur. Varsayılan: 5.
            **kwargs: ``ConversationBufferMemory.__init__`` parametreleri.
        """
        super().__init__(**kwargs)
        self.k = k

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Son ``k`` turu şablon değişkeni olarak döner.

        Tüm mesaj listesinin yalnızca son ``k * 2`` elemanını (her tur iki
        mesaj içerdiği için) alarak biçimlendirir.

        Args:
            inputs: Çağrıyı tetikleyen mevcut giriş (genellikle kullanılmaz).

        Returns:
            ``{memory_key: son_k_tur_geçmişi}`` biçiminde sözlük.
        """
        recent = self._messages[-(self.k * 2):]
        if self.return_messages:
            return {self.memory_key: recent}
        lines = []
        for m in recent:
            prefix = self.human_prefix if m.role == "user" else self.ai_prefix
            lines.append(f"{prefix}: {m.content}")
        return {self.memory_key: "\n".join(lines)}
