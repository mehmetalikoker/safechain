from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from safechain.llm.base import BaseLLM
from safechain.schema import Generation, Message


class OpenAI(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Generation:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        body.update(kwargs)

        url = f"{self.base_url}/chat/completions"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                result: Dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"OpenAI API error {exc.code}: {exc.read().decode()}"
            ) from exc

        choice = result["choices"][0]
        msg = choice["message"]
        text: str = msg.get("content") or ""
        tool_calls: List[Dict[str, Any]] = msg.get("tool_calls") or []

        return Generation(
            text=text,
            generation_info={
                "tool_calls": tool_calls,
                "finish_reason": choice.get("finish_reason"),
                "usage": result.get("usage", {}),
                "raw_message": msg,
                "raw": result,
            },
        )
