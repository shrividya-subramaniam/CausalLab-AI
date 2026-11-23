"""Thin wrapper around langchain-anthropic.

Model IDs come from config.yaml (llm section). Offline mode (--no-llm) never imports this.
"""
from __future__ import annotations

import os
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _model(role: str, cfg: dict):
    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env, or run with --no-llm.")
    llm_cfg = cfg["llm"]
    max_tokens = llm_cfg.get("max_tokens_by_role", {}).get(role, llm_cfg.get("max_tokens", 4000))
    kwargs = {"model": llm_cfg[role], "max_tokens": max_tokens}
    if llm_cfg.get("temperature") is not None:
        kwargs["temperature"] = llm_cfg["temperature"]
    return ChatAnthropic(**kwargs)


def structured(role: str, cfg: dict, schema: type[T], system: str, user: str) -> T:
    """Ask Claude for an instance of `schema`.

    Uses Anthropic's native JSON-schema output by default: current models do not support the
    forced tool call that LangChain otherwise uses for structured output."""
    method = cfg["llm"].get("structured_output_method", "json_schema")
    return _model(role, cfg).with_structured_output(schema, method=method).invoke([("system", system), ("human", user)])


def text(role: str, cfg: dict, system: str, user: str) -> str:
    msg = _model(role, cfg).invoke([("system", system), ("human", user)])
    content = msg.content
    if isinstance(content, list):  # content blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content).strip()
