from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: Union[str, List[Dict[str, Any]]]  # str or Anthropic content blocks


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Generation:
    text: str
    generation_info: Dict[str, Any] = field(default_factory=dict)
