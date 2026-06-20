from __future__ import annotations
from typing import Any, Dict, List

from safechain.chains.base import Chain


class SimpleSequentialChain(Chain):
    """Bir chain'in çıktısını sıradaki chain'e input olarak aktarır."""

    def __init__(self, chains: List[Chain], verbose: bool = False) -> None:
        self.chains = chains
        self.verbose = verbose

    @property
    def input_keys(self) -> List[str]:
        return self.chains[0].input_keys

    @property
    def output_keys(self) -> List[str]:
        return self.chains[-1].output_keys

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        state = dict(inputs)
        for chain in self.chains:
            if self.verbose:
                print(f"[{type(chain).__name__}] input: {state}")
            output = chain.invoke(state)
            if self.verbose:
                print(f"[{type(chain).__name__}] output: {output}")
            state.update(output)
        return state


class SequentialChain(Chain):
    """Birden fazla input/output değişkenini destekleyen sequential chain."""

    def __init__(
        self,
        chains: List[Chain],
        input_variables: List[str],
        output_variables: List[str],
        verbose: bool = False,
    ) -> None:
        self.chains = chains
        self._input_variables = input_variables
        self._output_variables = output_variables
        self.verbose = verbose

    @property
    def input_keys(self) -> List[str]:
        return self._input_variables

    @property
    def output_keys(self) -> List[str]:
        return self._output_variables

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        known = dict(inputs)
        for chain in self.chains:
            chain_inputs = {k: known[k] for k in chain.input_keys if k in known}
            output = chain.invoke(chain_inputs)
            if self.verbose:
                print(f"[{type(chain).__name__}] {output}")
            known.update(output)
        return {k: known[k] for k in self._output_variables if k in known}
