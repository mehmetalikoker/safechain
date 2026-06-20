from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from safechain.schema import Generation, Message


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, messages: List[Message], **kwargs: Any) -> Generation:
        pass

    def predict(self, prompt: str, **kwargs: Any) -> str:
        return self.generate([Message(role="user", content=prompt)], **kwargs).text

    def predict_messages(self, messages: List[Message], **kwargs: Any) -> str:
        return self.generate(messages, **kwargs).text

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.predict(prompt, **kwargs)
