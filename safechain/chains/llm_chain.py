from __future__ import annotations
from typing import Any, Dict, List, Optional

from safechain.chains.base import Chain
from safechain.schema import Message


class LLMChain(Chain):
    def __init__(
        self,
        llm: Any,
        prompt: Any,
        output_key: str = "text",
        memory: Optional[Any] = None,
    ) -> None:
        self.llm = llm
        self.prompt = prompt
        self.output_key = output_key
        self.memory = memory

    @property
    def input_keys(self) -> List[str]:
        keys = list(getattr(self.prompt, "input_variables", []))
        if self.memory:
            mem_vars = set(getattr(self.memory, "memory_variables", []))
            keys = [k for k in keys if k not in mem_vars]
        return keys

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
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
        return self.invoke(kwargs)[self.output_key]
