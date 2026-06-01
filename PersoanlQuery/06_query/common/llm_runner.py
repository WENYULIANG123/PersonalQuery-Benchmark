"""Shared MiniMax LLM client and call helper for 06_query stage-6 scripts.

The original `06_generate_by_persona_placeholder_*.py` files held the
`load_minimax_client` / `call_llm` pair at module scope; the
`06_generate_by_syntax_depth_no_depth_check_10_*.py` variants dynamic-loaded
those modules to reuse them. Once the placeholder scripts were removed,
this module re-homes the shared client lifecycle.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


_minimax_client = None
_first_request = True


def load_minimax_client(use_minimaxio: bool = False):
    """Load the MiniMax API client (Anthropic-compatible by default)."""
    global _minimax_client
    if _minimax_client is not None:
        return _minimax_client

    from .attribute_helpers import log

    sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery')
    from llm_client import MiniMaxAnthropicClient, MiniMaxIOAnthropicClient

    if use_minimaxio:
        _minimax_client = MiniMaxIOAnthropicClient()
        log("MiniMaxIO API 客户端初始化完成")
    else:
        _minimax_client = MiniMaxAnthropicClient()
        log("MiniMax API 客户端初始化完成")
    return _minimax_client


def reset_first_request() -> None:
    """Re-arm the cache-creation flag (call between independent prompt streams)."""
    global _first_request
    _first_request = True


def call_llm_no_empty_retry(prompt: str, system_base: Optional[str], step_name: str) -> str:
    """Call the LLM with prompt-cache on. Returns "" if the response is empty
    (caller treats this as a failed candidate, not a retriable error).
    """
    from .attribute_helpers import log

    global _first_request

    if _minimax_client is None:
        raise RuntimeError("LLM client is not loaded; call load_minimax_client() first")

    cache_info = {
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    if system_base and _first_request:
        log(f"[Request] {step_name} system_base (FIRST REQUEST - cache creation):\n{system_base}")
        _first_request = False

    log(f"[Request] {step_name} user_content:\n{prompt}")

    response, cache_info = _minimax_client.call_with_cache(
        system_base=system_base,
        user_content=prompt,
        max_tokens=32768,
        temperature=0.8,
        retry_on_empty_response=False,
        stream=True,
    )

    if not response:
        log(f"[Cache] {cache_info}")
        log(f"[ERROR] {step_name} empty response, marked failed without retry")
        return ""

    log(f"[Cache] {cache_info}")
    log(f"[Response] {step_name} response:\n{response[:1500]}")
    return response
