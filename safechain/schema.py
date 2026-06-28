from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union


@dataclass
class Message:
    """LLM konuşmasındaki tek bir mesajı temsil eder.

    Attributes:
        role: Mesajın sahibi. Geçerli değerler: "system", "user", "assistant", "tool".
        content: Mesaj içeriği. Düz metin (str) ya da Anthropic content block
                 listesi (List[Dict]) olabilir.
    """

    role: str  # "system" | "user" | "assistant" | "tool"
    content: Union[str, List[Dict[str, Any]]]  # str or Anthropic content blocks


@dataclass
class Document:
    """Yüklenmiş veya bölünmüş bir metin belgesini temsil eder.

    Attributes:
        page_content: Belgenin ham metin içeriği.
        metadata: Kaynağa, satır/sütun numarasına veya chunk indeksine dair
                  isteğe bağlı ek bilgiler. Varsayılan olarak boş dict döner.
    """

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Generation:
    """LLM'den dönen tek bir çıktıyı temsil eder.

    Attributes:
        text: Modelin ürettiği metin yanıtı.
        generation_info: Model çıktısına ilişkin ek meta veriler.
                         Tipik anahtarlar: "tool_uses", "stop_reason",
                         "usage", "raw_content", "raw" (Anthropic) ya da
                         "tool_calls", "finish_reason", "raw_message" (OpenAI).
    """

    text: str
    generation_info: Dict[str, Any] = field(default_factory=dict)
