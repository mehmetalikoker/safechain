from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Chain(ABC):
    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    @abstractmethod
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def run(self, *args: Any, **kwargs: Any) -> str:
        if args and not kwargs:
            key = self.input_keys[0] if self.input_keys else "input"
            inputs: Dict[str, Any] = {key: args[0]}
        else:
            inputs = dict(kwargs)
        result = self.invoke(inputs)
        out_key = self.output_keys[0] if self.output_keys else next(iter(result))
        return result[out_key]

    def __call__(self, inputs: Any, **kwargs: Any) -> Dict[str, Any]:
        if isinstance(inputs, str):
            key = self.input_keys[0] if self.input_keys else "input"
            inputs = {key: inputs}
        return self.invoke({**inputs, **kwargs})

    def __or__(self, other: "Chain") -> "Chain":
        from safechain.chains.sequential import SimpleSequentialChain
        return SimpleSequentialChain(chains=[self, other])
