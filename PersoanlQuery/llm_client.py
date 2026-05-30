#!/usr/bin/env python3
"""LLM Clients (MiniMax)."""

import atexit
import socket
import subprocess
import threading
import time
import json
import traceback
from datetime import datetime
from typing import Optional

VLLM_CONFIG_FILE = '/workspace/PersonalQuery/PersoanlQuery/06_query/vllm_config.json'

_MINIMAX_SSH_TARGET = "m3-login2"
_MINIMAX_HOST_IP_MAP = {
    "api.minimaxi.com": "47.79.2.234",
    "api.minimax.io": "47.252.72.253",
}
_MINIMAX_NETWORK_READY = False
_MINIMAX_TUNNEL_PROCESS = None
_MINIMAX_TUNNEL_PORT = None
_MINIMAX_NETWORK_LOCK = threading.Lock()
_ORIGINAL_GETADDRINFO = socket.getaddrinfo
_ORIGINAL_SOCKET_CLASS = socket.socket


def _log(msg: str):
    """带时间戳的日志打印"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _get_raw_socket_class():
    if getattr(socket.socket, "__module__", "") == "socks":
        import socks
        return socks._orgsocket
    return _ORIGINAL_SOCKET_CLASS


def _choose_local_socks_port() -> int:
    raw_socket_class = _get_raw_socket_class()
    with raw_socket_class(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_for_socks_tunnel(process, port: int, timeout_seconds: int = 10):
    raw_socket_class = _get_raw_socket_class()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            stderr = process.stderr.read().strip() if process.stderr else ""
            raise RuntimeError(
                f"MiniMax SSH SOCKS 隧道启动失败: return_code={return_code}, stderr={stderr}"
            )

        with raw_socket_class(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return

        time.sleep(0.1)

    process.terminate()
    raise TimeoutError(f"MiniMax SSH SOCKS 隧道启动超时: 127.0.0.1:{port}")


def _cleanup_minimax_socks_tunnel():
    global _MINIMAX_TUNNEL_PROCESS
    if _MINIMAX_TUNNEL_PROCESS is None:
        return
    if _MINIMAX_TUNNEL_PROCESS.poll() is None:
        _MINIMAX_TUNNEL_PROCESS.terminate()
        try:
            _MINIMAX_TUNNEL_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _MINIMAX_TUNNEL_PROCESS.kill()
            _MINIMAX_TUNNEL_PROCESS.wait(timeout=5)


def _patch_minimax_dns():
    def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        lookup_host = host.decode("ascii") if isinstance(host, bytes) else host
        mapped_host = _MINIMAX_HOST_IP_MAP.get(lookup_host)
        if mapped_host is not None:
            return _ORIGINAL_GETADDRINFO(mapped_host, port, family, type, proto, flags)
        return _ORIGINAL_GETADDRINFO(host, port, family, type, proto, flags)

    socket.getaddrinfo = patched_getaddrinfo


def _ensure_minimax_compute_node_network():
    """计算节点外网 DNS/直连不可用时，通过登录节点 SSH SOCKS 隧道访问 MiniMax。"""
    global _MINIMAX_NETWORK_READY, _MINIMAX_TUNNEL_PROCESS, _MINIMAX_TUNNEL_PORT
    if _MINIMAX_NETWORK_READY:
        return

    with _MINIMAX_NETWORK_LOCK:
        if _MINIMAX_NETWORK_READY:
            return

        import socks

        port = _choose_local_socks_port()
        cmd = [
            "ssh",
            "-q",
            "-N",
            "-D",
            f"127.0.0.1:{port}",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            _MINIMAX_SSH_TARGET,
        ]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        _wait_for_socks_tunnel(process, port)

        socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", port, rdns=True)
        socket.socket = socks.socksocket
        _patch_minimax_dns()

        _MINIMAX_TUNNEL_PROCESS = process
        _MINIMAX_TUNNEL_PORT = port
        _MINIMAX_NETWORK_READY = True
        atexit.register(_cleanup_minimax_socks_tunnel)
        _log(
            f"MiniMax 计算节点网络补丁已启用: socks=127.0.0.1:{port}, "
            f"ssh_target={_MINIMAX_SSH_TARGET}, dns_map={_MINIMAX_HOST_IP_MAP}"
        )


def _log_exception_details(prefix: str, exc: Exception):
    """打印底层 API 异常细节，避免只看到 SDK 的泛化错误。"""
    exc_type = type(exc)
    _log(f"{prefix} exception_type: {exc_type.__module__}.{exc_type.__name__}")
    _log(f"{prefix} exception_repr: {repr(exc)}")
    _log(f"{prefix} exception_message: {str(exc)}")

    for attr in ("status_code", "request_id", "code", "type"):
        if hasattr(exc, attr):
            _log(f"{prefix} {attr}: {getattr(exc, attr)}")

    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "reason_phrase"):
            if hasattr(response, attr):
                _log(f"{prefix} response.{attr}: {getattr(response, attr)}")
        response_text = None
        response_type = type(response)
        if response_type.__module__.startswith("httpx") and response_type.__name__ == "Response":
            if hasattr(response, "is_stream_consumed") and not getattr(response, "is_stream_consumed"):
                _log(f"{prefix} response.text skipped: streaming response not read")
            else:
                try:
                    response_text = response.text
                except Exception as response_exc:
                    _log(f"{prefix} response.text unavailable: {type(response_exc).__name__}: {response_exc}")
        else:
            try:
                response_text = getattr(response, "text", None)
            except Exception as response_exc:
                _log(f"{prefix} response.text unavailable: {type(response_exc).__name__}: {response_exc}")
        if response_text:
            _log(f"{prefix} response.text(first_2000): {response_text[:2000]}")

    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        cause_type = type(cause)
        _log(f"{prefix} cause_type: {cause_type.__module__}.{cause_type.__name__}")
        _log(f"{prefix} cause_repr: {repr(cause)}")
        _log(f"{prefix} cause_message: {str(cause)}")

    context = getattr(exc, "__context__", None)
    if context is not None and context is not cause:
        context_type = type(context)
        _log(f"{prefix} context_type: {context_type.__module__}.{context_type.__name__}")
        _log(f"{prefix} context_repr: {repr(context)}")
        _log(f"{prefix} context_message: {str(context)}")

    tb = "".join(traceback.format_exception(exc_type, exc, exc.__traceback__))
    _log(f"{prefix} traceback:\n{tb}")


def _usage_value(usage, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(key, 0)
    else:
        value = getattr(usage, key, 0)
    if value is None:
        return 0
    return int(value)


def _message_usage(message):
    if message is None:
        return None
    if isinstance(message, dict):
        return message.get("usage")
    return getattr(message, "usage", None)


def _append_message_text_content(message, text_parts: list[str]) -> None:
    if message is None:
        return
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if not content:
        return
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            continue
        if getattr(block, "type", None) == "text":
            text_parts.append(getattr(block, "text", ""))


class MiniMaxAnthropicClient:
    """MiniMax LLM Client using Anthropic SDK.

    Uses the Anthropic-compatible API endpoint at https://api.minimaxi.com/anthropic
    Supports MiniMax-M2.7 and other models with thinking/text content blocks.
    """
    def __init__(self, model: str = "MiniMax-M2.7-highspeed"):
        self.model = model
        _ensure_minimax_compute_node_network()
        import anthropic
        self.client = anthropic.Anthropic(
            base_url="https://api.minimaxi.com/anthropic",
            api_key="sk-cp-jqg2XWIob99HfZTveS5CqjO1h8BAQguTCcHG0p_vZlQ_rNqJgQLqNMwJ7AHMMwRhogi2I8A7o9FZ-f1dR2jsVNfwUsdLzicgrXm9tM8bqodav3ZhtQ0Ig-Y"
        )

    def call_with_thinking(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: Optional[float] = None,
        max_retries: int = 100,
    ) -> tuple:
        """Call MiniMax API and return both thinking and text.

        Returns:
            tuple: (thinking_text, text_content)
        """
        import anthropic

        safe_prompt = prompt.strip() if isinstance(prompt, str) else str(prompt)
        if not safe_prompt:
            return "", ""

        safe_max_tokens = max(128, int(max_tokens))
        safe_temp = temperature if temperature is not None else 0.7

        retry_count = 0
        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=safe_max_tokens,
                    temperature=safe_temp,
                    messages=[
                        {"role": "user", "content": safe_prompt}
                    ]
                )

                thinking_text = ""
                text_content = ""

                for block in message.content:
                    if block.type == "thinking":
                        thinking_text = block.thinking
                    elif block.type == "text":
                        text_content = block.text

                text_content = text_content.strip()
                # 空响应也需要重试
                if not text_content:
                    if attempt < max_retries - 1:
                        wait_time = min(60, (2 ** attempt) * 3)
                        print(f"[MiniMax-Anthropic] Empty response. Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    else:
                        return "", ""
                if retry_count > 0:
                    print(f"[MiniMax-Anthropic] Empty response retry succeeded after {retry_count} retries.")
                return thinking_text, text_content

            except Exception as e:
                error_str = str(e)
                # 429/529/500/502/503/504/520/522/530 等错误都需要重试
                retryable = any(code in error_str for code in ["429", "529", "500", "502", "503", "504", "520", "522", "530"])
                if retryable:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"[MiniMax-Anthropic] Rate limited ({e}). Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                print(f"[MiniMax-Anthropic] Error calling API: {e}")
                return "", ""

        return "", ""

    def call(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_retries: int = 100,
    ) -> str:
        """Call MiniMax API and return text content only.

        Returns:
            str: The text content from the response.
        """
        _, text_content = self.call_with_thinking(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )
        return text_content

    def call_with_cache(
        self,
        system_base: str,
        user_content: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_retries: int = 100,
        retry_on_empty_response: bool = True,
        stream: bool = False,
    ) -> tuple:
        """Call MiniMax API with prompt caching.

        Uses MiniMax's built-in ephemeral cache mechanism.
        The system content with cache_control is cached by MiniMax automatically.

        Args:
            system_base: Static system prompt (with cache_control on first call)
            user_content: Dynamic user content (varies per call)
            max_tokens: Max output tokens
            temperature: Sampling temperature
            max_retries: Max retry attempts

        Returns:
            tuple: (text_content, cache_read_input_tokens)
        """
        import anthropic

        safe_system = system_base.strip() if isinstance(system_base, str) else str(system_base)
        safe_user = user_content.strip() if isinstance(user_content, str) else str(user_content)
        if not safe_system or not safe_user:
            return "", 0

        safe_max_tokens = max(128, int(max_tokens))
        safe_temp = temperature if temperature is not None else 0.7

        if stream:
            return self._call_with_cache_streaming(
                safe_system=safe_system,
                safe_user=safe_user,
                safe_max_tokens=safe_max_tokens,
                safe_temp=safe_temp,
                max_retries=max_retries,
                retry_on_empty_response=retry_on_empty_response,
                log_prefix="[MiniMax-Cache]",
            )

        retry_count = 0
        for attempt in range(max_retries):
            try:
                # 构建 messages
                messages = [{"role": "user", "content": safe_user}]

                # 始终使用 cache_control，让 MiniMax 自动管理缓存
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=safe_max_tokens,
                    temperature=safe_temp,
                    system=[
                        {"type": "text", "text": safe_system, "cache_control": {"type": "ephemeral"}}
                    ],
                    messages=messages
                )

                text_content = ""
                cache_read_input_tokens = 0
                cache_creation_input_tokens = 0
                input_tokens = 0
                output_tokens = 0

                for block in message.content:
                    if block.type == "text":
                        text_content = block.text
                    elif block.type == "thinking":
                        pass

                # 提取所有 token 相关字段
                if hasattr(message, 'usage') and message.usage:
                    cache_read_input_tokens = getattr(message.usage, 'cache_read_input_tokens', 0)
                    cache_creation_input_tokens = getattr(message.usage, 'cache_creation_input_tokens', 0)
                    input_tokens = getattr(message.usage, 'input_tokens', 0)
                    output_tokens = getattr(message.usage, 'output_tokens', 0)

                text_content = text_content.strip()
                if not text_content:
                    if retry_on_empty_response and attempt < max_retries - 1:
                        wait_time = min(60, (2 ** attempt) * 3)
                        _log(f"[MiniMax-Cache] Empty response. Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    else:
                        return "", {"cache_creation_input_tokens": cache_creation_input_tokens, "cache_read_input_tokens": cache_read_input_tokens, "input_tokens": input_tokens, "output_tokens": output_tokens}

                if retry_count > 0:
                    _log(f"[MiniMax-Cache] Empty response retry succeeded after {retry_count} retries.")
                return text_content, {"cache_creation_input_tokens": cache_creation_input_tokens, "cache_read_input_tokens": cache_read_input_tokens, "input_tokens": input_tokens, "output_tokens": output_tokens}

            except Exception as e:
                error_str = str(e)
                retryable = any(code in error_str for code in ["429", "529", "500", "502", "503", "504", "520", "522", "530"])
                if retryable:
                    wait_time = min(60, (2 ** attempt) * 3)
                    _log(f"[MiniMax-Cache] Rate limited ({e}). Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                _log(f"[MiniMax-Cache] Error calling API: {e}")
                return "", {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

        return "", {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    def _call_with_cache_streaming(
        self,
        safe_system: str,
        safe_user: str,
        safe_max_tokens: int,
        safe_temp: float,
        max_retries: int,
        retry_on_empty_response: bool,
        log_prefix: str,
    ) -> tuple:
        messages = [{"role": "user", "content": safe_user}]
        retry_count = 0

        for attempt in range(max_retries):
            text_parts = []
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0
            input_tokens = 0
            output_tokens = 0
            try:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=safe_max_tokens,
                    temperature=safe_temp,
                    system=[
                        {"type": "text", "text": safe_system, "cache_control": {"type": "ephemeral"}}
                    ],
                    messages=messages,
                ) as stream_manager:
                    for text in stream_manager.text_stream:
                        text_parts.append(text)
                    message = None
                    try:
                        message = stream_manager.get_final_message()
                    except AttributeError as exc:
                        if "model_dump" not in str(exc):
                            raise
                        _log(f"{log_prefix} get_final_message failed after text stream; using streamed text without final message: {exc}")

                usage = _message_usage(message)
                if usage:
                    cache_read_input_tokens = _usage_value(usage, 'cache_read_input_tokens')
                    cache_creation_input_tokens = _usage_value(usage, 'cache_creation_input_tokens')
                    input_tokens = _usage_value(usage, 'input_tokens')
                    output_tokens = _usage_value(usage, 'output_tokens')

                text_content = "".join(text_parts).strip()
                if not text_content:
                    fallback_parts = []
                    _append_message_text_content(message, fallback_parts)
                    text_content = "".join(fallback_parts)
                    text_content = text_content.strip()

                if not text_content:
                    if retry_on_empty_response and attempt < max_retries - 1:
                        wait_time = min(60, (2 ** attempt) * 3)
                        _log(f"{log_prefix} Empty streaming response. Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    return "", {"cache_creation_input_tokens": cache_creation_input_tokens, "cache_read_input_tokens": cache_read_input_tokens, "input_tokens": input_tokens, "output_tokens": output_tokens}

                if retry_count > 0:
                    _log(f"{log_prefix} Empty streaming response retry succeeded after {retry_count} retries.")
                return text_content, {"cache_creation_input_tokens": cache_creation_input_tokens, "cache_read_input_tokens": cache_read_input_tokens, "input_tokens": input_tokens, "output_tokens": output_tokens}

            except Exception as e:
                error_str = str(e)
                retryable = any(code in error_str for code in ["429", "529", "500", "502", "503", "504", "520", "522", "530"])
                if retryable:
                    wait_time = min(60, (2 ** attempt) * 3)
                    _log(f"{log_prefix} Rate limited ({e}). Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                _log(f"{log_prefix} Error calling streaming API: {e}")
                _log_exception_details(log_prefix, e)
                return "", {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

        return "", {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}


class MiniMaxIOAnthropicClient:
    """MiniMax IO LLM Client using Anthropic SDK.

    Uses the Anthropic-compatible API endpoint at https://api.minimax.io/anthropic
    Supports MiniMax-M2.7 and other models with thinking/text content blocks.
    """
    def __init__(self, model: str = "MiniMax-M2.7"):
        self.model = model
        _ensure_minimax_compute_node_network()
        import anthropic
        self.client = anthropic.Anthropic(
            base_url="https://api.minimax.io/anthropic",
            api_key="sk-cp-GpGDO4bCprD-LScnuPL2JUzK0B3VNC6e6zFFkjRYP7OPTgY-sxuezrlwM-qLcri2zl0VmgGXmfv_FaNlOp-lVGDeiJnR6qVppp8fW4280iC4k4gSggDXWn8"
        )

    def call_with_thinking(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: Optional[float] = None,
        max_retries: int = 100,
    ) -> tuple:
        """Call MiniMax IO API and return both thinking and text.

        Returns:
            tuple: (thinking_text, text_content)
        """
        import anthropic

        safe_prompt = prompt.strip() if isinstance(prompt, str) else str(prompt)
        if not safe_prompt:
            return "", ""

        safe_max_tokens = max(128, int(max_tokens))
        safe_temp = temperature if temperature is not None else 0.7

        retry_count = 0
        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=safe_max_tokens,
                    temperature=safe_temp,
                    messages=[
                        {"role": "user", "content": safe_prompt}
                    ]
                )

                thinking_text = ""
                text_content = ""

                for block in message.content:
                    if block.type == "thinking":
                        thinking_text = block.thinking
                    elif block.type == "text":
                        text_content = block.text

                text_content = text_content.strip()
                if not text_content:
                    if attempt < max_retries - 1:
                        wait_time = min(60, (2 ** attempt) * 3)
                        print(f"[MiniMaxIO-Anthropic] Empty response. Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    else:
                        return "", ""
                if retry_count > 0:
                    print(f"[MiniMaxIO-Anthropic] Empty response retry succeeded after {retry_count} retries.")
                return thinking_text, text_content

            except Exception as e:
                error_str = str(e)
                retryable = any(code in error_str for code in ["429", "529", "500", "502", "503", "504", "520", "522", "530"])
                if retryable:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"[MiniMaxIO-Anthropic] Rate limited ({e}). Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                _log(f"[MiniMaxIO-Anthropic] Error calling API: {e}")
                _log_exception_details("[MiniMaxIO-Anthropic]", e)
                return "", ""

        return "", ""

    def call(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_retries: int = 100,
    ) -> str:
        """Call MiniMax IO API and return text content only."""
        _, text_content = self.call_with_thinking(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )
        return text_content

    def call_with_cache(
        self,
        system_base: str,
        user_content: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_retries: int = 100,
        retry_on_empty_response: bool = True,
        stream: bool = False,
    ) -> tuple:
        """Call MiniMax IO API with prompt caching."""
        import anthropic

        safe_system = system_base.strip() if isinstance(system_base, str) else str(system_base)
        safe_user = user_content.strip() if isinstance(user_content, str) else str(user_content)
        if not safe_system or not safe_user:
            return "", 0

        safe_max_tokens = max(128, int(max_tokens))
        safe_temp = temperature if temperature is not None else 0.7

        if stream:
            return self._call_with_cache_streaming(
                safe_system=safe_system,
                safe_user=safe_user,
                safe_max_tokens=safe_max_tokens,
                safe_temp=safe_temp,
                max_retries=max_retries,
                retry_on_empty_response=retry_on_empty_response,
                log_prefix="[MiniMaxIO-Cache]",
            )

        retry_count = 0
        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": safe_user}]

                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=safe_max_tokens,
                    temperature=safe_temp,
                    system=[
                        {"type": "text", "text": safe_system, "cache_control": {"type": "ephemeral"}}
                    ],
                    messages=messages
                )

                text_content = ""
                cache_read_input_tokens = 0
                cache_creation_input_tokens = 0
                input_tokens = 0
                output_tokens = 0

                for block in message.content:
                    if block.type == "text":
                        text_content = block.text
                    elif block.type == "thinking":
                        pass

                if hasattr(message, 'usage') and message.usage:
                    cache_read_input_tokens = getattr(message.usage, 'cache_read_input_tokens', 0)
                    cache_creation_input_tokens = getattr(message.usage, 'cache_creation_input_tokens', 0)
                    input_tokens = getattr(message.usage, 'input_tokens', 0)
                    output_tokens = getattr(message.usage, 'output_tokens', 0)

                text_content = text_content.strip()
                if not text_content:
                    if retry_on_empty_response and attempt < max_retries - 1:
                        wait_time = min(60, (2 ** attempt) * 3)
                        _log(f"[MiniMaxIO-Cache] Empty response. Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    else:
                        return "", {"cache_creation_input_tokens": cache_creation_input_tokens, "cache_read_input_tokens": cache_read_input_tokens, "input_tokens": input_tokens, "output_tokens": output_tokens}

                if retry_count > 0:
                    _log(f"[MiniMaxIO-Cache] Empty response retry succeeded after {retry_count} retries.")
                return text_content, {"cache_creation_input_tokens": cache_creation_input_tokens, "cache_read_input_tokens": cache_read_input_tokens, "input_tokens": input_tokens, "output_tokens": output_tokens}

            except Exception as e:
                error_str = str(e)
                retryable = any(code in error_str for code in ["429", "529", "500", "502", "503", "504", "520", "522", "530"])
                if retryable:
                    wait_time = min(60, (2 ** attempt) * 3)
                    _log(f"[MiniMaxIO-Cache] Rate limited ({e}). Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                _log(f"[MiniMaxIO-Cache] Error calling API: {e}")
                _log_exception_details("[MiniMaxIO-Cache]", e)
                return "", {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

        return "", {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    def _call_with_cache_streaming(
        self,
        safe_system: str,
        safe_user: str,
        safe_max_tokens: int,
        safe_temp: float,
        max_retries: int,
        retry_on_empty_response: bool,
        log_prefix: str,
    ) -> tuple:
        messages = [{"role": "user", "content": safe_user}]
        retry_count = 0

        for attempt in range(max_retries):
            text_parts = []
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0
            input_tokens = 0
            output_tokens = 0
            try:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=safe_max_tokens,
                    temperature=safe_temp,
                    system=[
                        {"type": "text", "text": safe_system, "cache_control": {"type": "ephemeral"}}
                    ],
                    messages=messages,
                ) as stream_manager:
                    for text in stream_manager.text_stream:
                        text_parts.append(text)
                    message = None
                    try:
                        message = stream_manager.get_final_message()
                    except AttributeError as exc:
                        if "model_dump" not in str(exc):
                            raise
                        _log(f"{log_prefix} get_final_message failed after text stream; using streamed text without final message: {exc}")

                usage = _message_usage(message)
                if usage:
                    cache_read_input_tokens = _usage_value(usage, 'cache_read_input_tokens')
                    cache_creation_input_tokens = _usage_value(usage, 'cache_creation_input_tokens')
                    input_tokens = _usage_value(usage, 'input_tokens')
                    output_tokens = _usage_value(usage, 'output_tokens')

                text_content = "".join(text_parts).strip()
                if not text_content:
                    fallback_parts = []
                    _append_message_text_content(message, fallback_parts)
                    text_content = "".join(fallback_parts)
                    text_content = text_content.strip()

                if not text_content:
                    if retry_on_empty_response and attempt < max_retries - 1:
                        wait_time = min(60, (2 ** attempt) * 3)
                        _log(f"{log_prefix} Empty streaming response. Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    return "", {"cache_creation_input_tokens": cache_creation_input_tokens, "cache_read_input_tokens": cache_read_input_tokens, "input_tokens": input_tokens, "output_tokens": output_tokens}

                if retry_count > 0:
                    _log(f"{log_prefix} Empty streaming response retry succeeded after {retry_count} retries.")
                return text_content, {"cache_creation_input_tokens": cache_creation_input_tokens, "cache_read_input_tokens": cache_read_input_tokens, "input_tokens": input_tokens, "output_tokens": output_tokens}

            except Exception as e:
                error_str = str(e)
                retryable = any(code in error_str for code in ["429", "529", "500", "502", "503", "504", "520", "522", "530"])
                if retryable:
                    wait_time = min(60, (2 ** attempt) * 3)
                    _log(f"{log_prefix} Rate limited ({e}). Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                _log(f"{log_prefix} Error calling streaming API: {e}")
                _log_exception_details(log_prefix, e)
                return "", {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

        return "", {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

