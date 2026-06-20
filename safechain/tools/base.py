from __future__ import annotations
import inspect
from typing import Any, Callable, Dict, List, Optional, get_type_hints

_PY_TO_JSON: Dict[type, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _infer_json_schema(func: Callable) -> Dict[str, Any]:
    sig = inspect.signature(func)
    hints: Dict[str, Any] = {}
    try:
        hints = get_type_hints(func)
    except Exception:
        pass

    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        py_type = hints.get(name, str)
        json_type = _PY_TO_JSON.get(py_type, "string")
        properties[name] = {"type": json_type}
        # Add description from param annotation docstring if available
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {"type": "object", "properties": properties, "required": required}


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        args_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self._args_schema = args_schema

    def run(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    def __call__(self, **kwargs: Any) -> Any:
        return self.run(**kwargs)

    @property
    def schema(self) -> Dict[str, Any]:
        return self._args_schema or _infer_json_schema(self.func)

    def to_anthropic_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.schema,
        }

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


def tool(
    _func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Dict[str, Any]] = None,
) -> Any:
    """Fonksiyonu Tool nesnesine dönüştüren decorator.

    Kullanım:
        @tool
        def hesapla(x: int, y: int) -> int: ...

        @tool(name="toplama", description="İki sayı toplar")
        def toplama(x: int, y: int) -> int: ...
    """
    def decorator(func: Callable) -> Tool:
        return Tool(
            name=name or func.__name__,
            description=description or (func.__doc__ or "").strip(),
            func=func,
            args_schema=args_schema,
        )

    if _func is not None:
        return decorator(_func)
    return decorator
