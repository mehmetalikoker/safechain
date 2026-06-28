from __future__ import annotations
import string
from typing import Any, Dict, List, Optional, Tuple, Union

from safechain.schema import Message


class PromptTemplate:
    """Değişken yer tutucular içeren tek parça metin şablonu.

    Python'ın ``str.format`` sözdizimini kullanır: ``{degisken_adi}``.
    ``input_variables`` açıkça verilmezse şablondan otomatik çıkarılır.

    Attributes:
        template: ``{key}`` yer tutucularını içeren ham metin şablonu.
        input_variables: Şablonun beklediği değişken isimlerinin listesi.
    """

    def __init__(
        self,
        template: str,
        input_variables: Optional[List[str]] = None,
    ) -> None:
        """PromptTemplate oluşturur.

        Args:
            template: ``{key}`` sözdiziminde yer tutucular içeren metin.
            input_variables: Beklenen değişken isimlerinin listesi.
                             ``None`` ise şablondan otomatik çıkarılır.
        """
        self.template = template
        if input_variables is None:
            formatter = string.Formatter()
            self.input_variables = [
                fname
                for _, fname, _, _ in formatter.parse(template)
                if fname
            ]
        else:
            self.input_variables = input_variables

    @classmethod
    def from_template(cls, template: str) -> "PromptTemplate":
        """Ham metin şablonundan PromptTemplate nesnesi oluşturur.

        ``input_variables`` otomatik olarak şablondan çıkarılır.

        Args:
            template: ``{key}`` yer tutucularını içeren metin şablonu.

        Returns:
            Yeni PromptTemplate nesnesi.
        """
        return cls(template=template)

    def format(self, **kwargs: Any) -> str:
        """Şablona değişken değerlerini yerleştirerek nihai metni döner.

        Args:
            **kwargs: Şablondaki her yer tutucu için karşılık gelen değer.

        Returns:
            Değişkenlerin yerleştirildiği hazır metin.

        Raises:
            KeyError: Şablonda beklenen bir değişken ``kwargs`` içinde
                      bulunamazsa fırlatılır.
        """
        return self.template.format(**kwargs)

    def __or__(self, other: Any) -> Any:
        """``prompt | llm`` sözdizimi ile LLMChain oluşturur.

        Args:
            other: Zincirin LLM bileşeni (BaseLLM örneği).

        Returns:
            prompt ve llm'yi birleştiren LLMChain nesnesi.
        """
        from safechain.chains.llm_chain import LLMChain
        return LLMChain(llm=other, prompt=self)


class _MessageTemplate:
    """Tek bir konuşma mesajı için içi role+içerik şablonu.

    Doğrudan kullanım yerine ``SystemMessage``, ``HumanMessage`` ve
    ``AIMessage`` yardımcı fonksiyonları tercih edilmelidir.

    Attributes:
        role: Mesajın rolü ("system", "user", "assistant").
        content: ``{key}`` yer tutucuları içerebilen içerik şablonu.
    """

    def __init__(self, role: str, content: str) -> None:
        """_MessageTemplate oluşturur.

        Args:
            role: Mesajın rolü.
            content: İçerik şablonu metni.
        """
        self.role = role
        self.content = content

    def format(self, **kwargs: Any) -> Message:
        """Şablona değişkenleri yerleştirerek Message nesnesi döner.

        Args:
            **kwargs: Şablondaki yer tutucuların değerleri.

        Returns:
            Rolü ve doldurulmuş içeriğiyle hazır Message nesnesi.
        """
        return Message(role=self.role, content=self.content.format(**kwargs))


def SystemMessage(content: str) -> _MessageTemplate:
    """Sistem rolünde bir mesaj şablonu oluşturur.

    Args:
        content: Sistem talimatı metni; ``{key}`` yer tutucuları içerebilir.

    Returns:
        role="system" olan _MessageTemplate nesnesi.
    """
    return _MessageTemplate("system", content)


def HumanMessage(content: str) -> _MessageTemplate:
    """Kullanıcı (human) rolünde bir mesaj şablonu oluşturur.

    Args:
        content: Kullanıcı mesajı metni; ``{key}`` yer tutucuları içerebilir.

    Returns:
        role="user" olan _MessageTemplate nesnesi.
    """
    return _MessageTemplate("user", content)


def AIMessage(content: str) -> _MessageTemplate:
    """Asistan (AI) rolünde bir mesaj şablonu oluşturur.

    Args:
        content: Asistan mesajı metni; ``{key}`` yer tutucuları içerebilir.

    Returns:
        role="assistant" olan _MessageTemplate nesnesi.
    """
    return _MessageTemplate("assistant", content)


class ChatPromptTemplate:
    """Çoklu mesajlardan oluşan konuşma şablonu.

    Her mesaj bir ``(role, template)`` demeti veya ``_MessageTemplate``
    nesnesi olabilir. Şablondaki tüm değişkenler ``input_variables``
    listesinde otomatik toplanır.

    Attributes:
        messages: Mesaj şablonlarının listesi.
        input_variables: Tüm mesajlardaki değişken isimlerinin birleşimi.
    """

    def __init__(
        self,
        messages: List[Union[Tuple[str, str], _MessageTemplate]],
    ) -> None:
        """ChatPromptTemplate oluşturur.

        Args:
            messages: ``(role, template_str)`` demetleri veya
                      ``_MessageTemplate`` nesnelerinin listesi.
        """
        self.messages = messages
        formatter = string.Formatter()
        variables: set = set()
        for msg in messages:
            template_str = msg[1] if isinstance(msg, tuple) else msg.content
            for _, fname, _, _ in formatter.parse(template_str):
                if fname:
                    variables.add(fname)
        self.input_variables = list(variables)

    @classmethod
    def from_messages(
        cls,
        messages: List[Union[Tuple[str, str], _MessageTemplate]],
    ) -> "ChatPromptTemplate":
        """Mesaj listesinden ChatPromptTemplate oluşturur.

        Args:
            messages: ``(role, template_str)`` demetleri veya
                      ``_MessageTemplate`` nesnelerinin listesi.

        Returns:
            Yeni ChatPromptTemplate nesnesi.
        """
        return cls(messages=messages)

    def format_messages(self, **kwargs: Any) -> List[Message]:
        """Değişkenleri yerleştirerek Message listesi döner.

        Her şablon metni ``str.format`` ile doldurulur ve karşılık gelen
        role ile birlikte Message nesnesine dönüştürülür.

        Args:
            **kwargs: Tüm mesaj şablonlarındaki yer tutucuların değerleri.

        Returns:
            Hazır Message nesnelerinin listesi.
        """
        result: List[Message] = []
        for msg in self.messages:
            if isinstance(msg, tuple):
                role, template = msg
                result.append(Message(role=role, content=template.format(**kwargs)))
            else:
                result.append(msg.format(**kwargs))
        return result

    def format(self, **kwargs: Any) -> str:
        """Tüm mesajları tek düz metin olarak döner (hata ayıklama için).

        Her mesaj ``role: içerik`` biçiminde birleştirilir.

        Args:
            **kwargs: Şablonlardaki yer tutucuların değerleri.

        Returns:
            Tüm mesajların birleştirildiği okunabilir metin.
        """
        return "\n".join(
            f"{m.role}: {m.content}" for m in self.format_messages(**kwargs)
        )

    def __or__(self, other: Any) -> Any:
        """``chat_prompt | llm`` sözdizimi ile LLMChain oluşturur.

        Args:
            other: Zincirin LLM bileşeni (BaseLLM örneği).

        Returns:
            prompt ve llm'yi birleştiren LLMChain nesnesi.
        """
        from safechain.chains.llm_chain import LLMChain
        return LLMChain(llm=other, prompt=self)
