#!/usr/bin/env python3
import requests
import time
import anthropic
import re
import sys
from typing import Any, Dict, Optional


class LLMClient:
    #def __init__(self, model: str = "MiniMax-M2.5-highspeed"):
    def __init__(self, model: str = "glm-4.7"):
        self.model = model
        # MiniMax config
        self.minimax_api_key = "sk-cp-jqg2XWIob99HfZTveS5CqjO1h8BAQguTCcHG0p_vZlQ_rNqJgQLqNMwJ7AHMMwRhogi2I8A7o9FZ-f1dR2jsVNfwUsdLzicgrXm9tM8bqodav3ZhtQ0Ig-Y"
        self.minimax_base_url = "https://api.minimaxi.com/v1"
        # GLM config - using Anthropic SDK
        self.glm_client = anthropic.Anthropic(
            base_url="https://api.z.ai/api/anthropic",
            api_key="db2682f8a0024278a672f762ce36d7cd.RC8PtxIy5xdlh8Uj"
        )

    def _is_glm_model(self) -> bool:
        return self.model.upper().startswith("GLM")

    def _normalize_glm_model_name(self, model_name: str) -> str:
        if not isinstance(model_name, str):
            return "glm-4.7"
        if model_name.upper().startswith("GLM"):
            return model_name.lower()
        return model_name

    def _extract_http_status(self, error: Exception) -> int:
        status = getattr(error, "status_code", None)
        if isinstance(status, int):
            return status

        error_str = str(error)
        match = re.search(r"Error code:\s*(\d{3})", error_str)
        if match:
            return int(match.group(1))

        match = re.search(r"HTTP/1\.1\s+(\d{3})", error_str)
        if match:
            return int(match.group(1))
        return -1

    def _extract_error_detail(self, error: Exception) -> Dict[str, Any]:
        detail: Dict[str, Any] = {
            "status": self._extract_http_status(error),
            "code": None,
            "message": None,
            "request_id": None,
        }

        response = getattr(error, "response", None)
        if response is not None:
            body = getattr(response, "json", None)
            if callable(body):
                try:
                    payload: Any = body()
                    err_obj: Dict[str, Any] = {}
                    if isinstance(payload, dict):
                        maybe_err = payload.get("error", {})
                        if isinstance(maybe_err, dict):
                            err_obj = maybe_err
                    detail["code"] = err_obj.get("code")
                    detail["message"] = err_obj.get("message")
                    if isinstance(payload, dict):
                        detail["request_id"] = payload.get("request_id")
                except Exception:
                    pass

        text = str(error)
        if detail["code"] is None:
            m = re.search(r"'code'\s*:\s*'?(\d+)'?", text)
            if m:
                detail["code"] = m.group(1)
        if detail["message"] is None:
            m = re.search(r"'message'\s*:\s*'([^']+)'", text)
            if m:
                detail["message"] = m.group(1)
        if detail["request_id"] is None:
            m = re.search(r"'request_id'\s*:\s*'([^']+)'", text)
            if m:
                detail["request_id"] = m.group(1)

        return detail

    def _preprocess_glm_prompt(self, prompt: str, max_chars: int = 8000) -> str:
        text = prompt if isinstance(prompt, str) else str(prompt)
        text = text.strip()
        if not text:
            return ""

        text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        if len(text) > max_chars:
            text = text[:max_chars].rstrip()

        return text

    def call(self, prompt: str, max_tokens: int = 1024, temperature: Optional[float] = None, max_retries: int = 5) -> str:
        if self._is_glm_model():
            return self._call_glm_anthropic(prompt, max_tokens, temperature, max_retries)
        else:
            return self._call_minimax(prompt, max_tokens, temperature, max_retries)

    def _call_glm_anthropic(self, prompt: str, max_tokens: int, temperature: Optional[float], max_retries: int) -> str:
        safe_prompt = self._preprocess_glm_prompt(prompt, max_chars=6000)
        if not safe_prompt:
            return ""

        safe_max_tokens = max(1, min(int(max_tokens), 128))

        kwargs = {
            "model": self._normalize_glm_model_name(self.model),
            "max_tokens": safe_max_tokens,
            "messages": [{"role": "user", "content": safe_prompt}]
        }
        bad_request_downgraded = False

        for attempt in range(max_retries):
            try:
                response = self.glm_client.messages.create(**kwargs)
                return response.content[0].text
            except Exception as e:
                error_str = str(e)
                detail = self._extract_error_detail(e)
                status_code = detail["status"]
                is_retryable_server = status_code in (429, 500, 502, 503, 504)
                is_retryable_bad_request = status_code == 400

                if (is_retryable_server or is_retryable_bad_request) and attempt < max_retries - 1:
                    if is_retryable_bad_request:
                        kwargs["model"] = self._normalize_glm_model_name(kwargs.get("model", self.model))
                        kwargs["max_tokens"] = max(1, min(int(kwargs.get("max_tokens", safe_max_tokens)), 128))
                        kwargs.pop("temperature", None)
                        if detail.get("code") == "1210":
                            prompt_max_chars = 3000 if not bad_request_downgraded else 2000
                            kwargs["messages"] = [{
                                "role": "user",
                                "content": self._preprocess_glm_prompt(safe_prompt, max_chars=prompt_max_chars),
                            }]
                            bad_request_downgraded = True

                    wait_time = min(60, (2 ** attempt) * 3)
                    print(
                        f"GLM API error (attempt {attempt + 1}/{max_retries}, status={status_code}): "
                        f"code={detail['code']}, message={detail['message']}, request_id={detail['request_id']}. "
                        f"Waiting {wait_time}s...",
                        file=sys.stderr,
                    )
                    time.sleep(wait_time)
                    continue
                print(
                    f"Error calling GLM API: status={status_code}, code={detail['code']}, "
                    f"message={detail['message']}, request_id={detail['request_id']}, raw={error_str}",
                    file=sys.stderr,
                )
                return ""
        return ""

    def _call_minimax(self, prompt: str, max_tokens: int, temperature: Optional[float], max_retries: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.minimax_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        if temperature is not None:
            payload["temperature"] = temperature

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.minimax_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                elif response.status_code in (429, 500) and attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"Rate limited. Waiting {wait_time}s before retry... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error calling MiniMax API: {response.status_code} - {response.text}")
                    return ""
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                print(f"Error calling MiniMax API: {e}")
                return ""
        return ""
