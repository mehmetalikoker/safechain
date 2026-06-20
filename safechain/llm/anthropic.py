from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from safechain.llm.base import BaseLLM
from safechain.schema import Generation, Message

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class Claude(BaseLLM):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system = system

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Generation:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }

        # Separate system messages from conversation
        system_text: Optional[str] = self.system
        api_messages: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_text = m.content  # type: ignore[assignment]
            else:
                api_messages.append({"role": m.role, "content": m.content})

        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": api_messages,
        }
        if system_text:
            body["system"] = system_text
        if tools:
            body["tools"] = tools
        body.update(kwargs)

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(_API_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                result: Dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Anthropic API error {exc.code}: {exc.read().decode()}"
            ) from exc

        content_blocks: List[Dict[str, Any]] = result.get("content", [])
        text = "".join(b["text"] for b in content_blocks if b.get("type") == "text")
        tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]

        return Generation(
            text=text,
            generation_info={
                "tool_uses": tool_uses,
                "stop_reason": result.get("stop_reason"),
                "usage": result.get("usage", {}),
                "raw_content": content_blocks,
                "raw": result,
            },
        )
