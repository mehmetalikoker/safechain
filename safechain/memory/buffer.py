from __future__ import annotations
from typing import Any, Dict, List, Optional

from safechain.schema import Message


class ConversationBufferMemory:
    def __init__(
        self,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "history",
        return_messages: bool = False,
    ) -> None:
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.memory_key = memory_key
        self.return_messages = return_messages
        self._messages: List[Message] = []

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    @property
    def messages(self) -> List[Message]:
        return list(self._messages)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
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
        human = inputs.get("input") or next(iter(inputs.values()), "")
        ai = outputs.get("output") or next(iter(outputs.values()), "")
        self._messages.append(Message(role="user", content=str(human)))
        self._messages.append(Message(role="assistant", content=str(ai)))

    def clear(self) -> None:
        self._messages.clear()


class ConversationBufferWindowMemory(ConversationBufferMemory):
    """Son `k` konuşma turunu saklar."""

    def __init__(self, k: int = 5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.k = k

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        recent = self._messages[-(self.k * 2) :]
        if self.return_messages:
            return {self.memory_key: recent}
        lines = []
        for m in recent:
            prefix = self.human_prefix if m.role == "user" else self.ai_prefix
            lines.append(f"{prefix}: {m.content}")
        return {self.memory_key: "\n".join(lines)}
