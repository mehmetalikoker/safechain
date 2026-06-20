from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from safechain.schema import Message
from safechain.tools.base import Tool


def _provider(llm: Any) -> str:
    name = type(llm).__name__.lower()
    return "anthropic" if "claude" in name or "anthropic" in name else "openai"


class AgentExecutor:
    """LLM + araçları (tools) birleştiren ajan çalıştırıcı.

    Anthropic ve OpenAI function-calling protokollerini otomatik algılar.
    """

    def __init__(
        self,
        llm: Any,
        tools: List[Tool],
        max_iterations: int = 10,
        verbose: bool = False,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.tools: Dict[str, Tool] = {t.name: t for t in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.system_prompt = (
            system_prompt
            or "Sen yardımsever bir asistandsın. Gerektiğinde araçları kullan."
        )

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_input: str = inputs.get("input") or next(iter(inputs.values()), "")
        provider = _provider(self.llm)

        tool_schemas = (
            [t.to_anthropic_schema() for t in self.tools.values()]
            if provider == "anthropic"
            else [t.to_openai_schema() for t in self.tools.values()]
        )

        messages: List[Message] = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_input),
        ]

        for iteration in range(self.max_iterations):
            generation = self.llm.generate(messages, tools=tool_schemas)

            if self.verbose:
                print(f"\n[Agent iter {iteration + 1}] {generation.text or '(tool call)'}")

            if provider == "anthropic":
                stop_reason = generation.generation_info.get("stop_reason")
                tool_uses: List[Dict[str, Any]] = generation.generation_info.get("tool_uses", [])

                if stop_reason != "tool_use" or not tool_uses:
                    return {"output": generation.text}

                # Asistan mesajını ham content bloklarıyla ekle
                raw_content = generation.generation_info["raw_content"]
                messages.append(Message(role="assistant", content=raw_content))

                # Her tool'u çalıştır, sonuçları topla
                tool_results: List[Dict[str, Any]] = []
                for tu in tool_uses:
                    result = self._run_tool(tu["name"], tu["input"])
                    if self.verbose:
                        print(f"  [Tool] {tu['name']}({tu['input']}) -> {result}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu["id"],
                            "content": str(result),
                        }
                    )
                messages.append(Message(role="user", content=tool_results))

            else:  # OpenAI
                finish_reason = generation.generation_info.get("finish_reason")
                oa_tool_calls: List[Dict[str, Any]] = generation.generation_info.get("tool_calls", [])

                if finish_reason != "tool_calls" or not oa_tool_calls:
                    return {"output": generation.text}

                # Asistan mesajını raw OpenAI format ile ekle
                raw_msg = generation.generation_info["raw_message"]
                messages.append(Message(role="assistant", content=raw_msg.get("content") or ""))

                for tc in oa_tool_calls:
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"])
                    result = self._run_tool(fn_name, fn_args)
                    if self.verbose:
                        print(f"  [Tool] {fn_name}({fn_args}) -> {result}")
                    messages.append(
                        Message(
                            role="tool",
                            content=str(result),
                        )
                    )

        return {"output": generation.text, "warning": "max_iterations reached"}

    def run(self, input: str) -> str:
        return self.invoke({"input": input})["output"]

    def _run_tool(self, name: str, args: Dict[str, Any]) -> Any:
        if name not in self.tools:
            return f"Bilinmeyen araç: {name}"
        try:
            return self.tools[name].run(**args)
        except Exception as exc:
            return f"Hata: {exc}"
