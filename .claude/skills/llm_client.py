#!/usr/bin/env python3
import requests
import time
import anthropic


class LLMClient:
    def __init__(self, model: str = "MiniMax-M2.5-highspeed"):
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

    def call(self, prompt: str, max_tokens: int = 1024, temperature: float = None, max_retries: int = 5) -> str:
        if self._is_glm_model():
            return self._call_glm_anthropic(prompt, max_tokens, temperature, max_retries)
        else:
            return self._call_minimax(prompt, max_tokens, temperature, max_retries)

    def _call_glm_anthropic(self, prompt: str, max_tokens: int, temperature: float, max_retries: int) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        for attempt in range(max_retries):
            try:
                response = self.glm_client.messages.create(**kwargs)
                return response.content[0].text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str and attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"Rate limited. Waiting {wait_time}s before retry... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                print(f"Error calling GLM API: {e}")
                return ""
        return ""

    def _call_minimax(self, prompt: str, max_tokens: int, temperature: float, max_retries: int) -> str:
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