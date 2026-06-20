from __future__ import annotations
import string
from typing import Any, Dict, List, Optional, Tuple, Union

from safechain.schema import Message


class PromptTemplate:
    def __init__(
        self,
        template: str,
        input_variables: Optional[List[str]] = None,
    ) -> None:
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
        return cls(template=template)

    def format(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)

    def __or__(self, other: Any) -> Any:
        from safechain.chains.llm_chain import LLMChain
        return LLMChain(llm=other, prompt=self)


# ---------------------------------------------------------------------------
# Chat prompt building blocks
# ---------------------------------------------------------------------------

class _MessageTemplate:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def format(self, **kwargs: Any) -> Message:
        return Message(role=self.role, content=self.content.format(**kwargs))


def SystemMessage(content: str) -> _MessageTemplate:
    return _MessageTemplate("system", content)


def HumanMessage(content: str) -> _MessageTemplate:
    return _MessageTemplate("user", content)


def AIMessage(content: str) -> _MessageTemplate:
    return _MessageTemplate("assistant", content)


class ChatPromptTemplate:
    def __init__(
        self,
        messages: List[Union[Tuple[str, str], _MessageTemplate]],
    ) -> None:
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
        return cls(messages=messages)

    def format_messages(self, **kwargs: Any) -> List[Message]:
        result: List[Message] = []
        for msg in self.messages:
            if isinstance(msg, tuple):
                role, template = msg
                result.append(Message(role=role, content=template.format(**kwargs)))
            else:
                result.append(msg.format(**kwargs))
        return result

    def format(self, **kwargs: Any) -> str:
        return "\n".join(
            f"{m.role}: {m.content}" for m in self.format_messages(**kwargs)
        )

    def __or__(self, other: Any) -> Any:
        from safechain.chains.llm_chain import LLMChain
        return LLMChain(llm=other, prompt=self)
