#!/usr/bin/env python3
"""
Shared LLM Client for skill scripts.
Provides a unified interface to call the GLM API via Anthropic SDK.
"""
import anthropic
import time


class LLMClient:
    """Client for interacting with the LLM API."""

    def __init__(self, model: str = "GLM-4.5V"):
        self.client = anthropic.Anthropic(
            base_url="https://api.z.ai/api/anthropic",
            api_key="db2682f8a0024278a672f762ce36d7cd.RC8PtxIy5xdlh8Uj"
        )
        self.model = model

    def call(self, prompt: str, max_tokens: int = 4096, temperature: float = None, max_retries: int = 5) -> str:
        """
        Call the LLM API with a prompt and return the response text.

        Args:
            prompt: The user prompt to send to the LLM
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0-1.0). Higher values = more random/creative.
                        None uses the model's default temperature.
            max_retries: Maximum number of retries for rate limiting errors (default: 5)

        Returns:
            The response text from the LLM, or empty string on error
        """
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        if temperature is not None:
            kwargs["temperature"] = temperature

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(**kwargs)
                return response.content[0].text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str and attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"Rate limited. Waiting {wait_time}s before retry... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                print(f"Error calling LLM API: {e}")
                return ""

        return ""
# #!/usr/bin/env python3
# """
# Shared LLM Client for skill scripts.
# Provides a unified interface to call the MiniMax API via requests.
# """
# import requests
# import time
# import json


# class LLMClient:
#     """Client for interacting with the LLM API."""

#     def __init__(self):
#         self.api_key = "sk-cp-jqg2XWIob99HfZTveS5CqjO1h8BAQguTCcHG0p_vZlQ_rNqJgQLqNMwJ7AHMMwRhogi2I8A7o9FZ-f1dR2jsVNfwUsdLzicgrXm9tM8bqodav3ZhtQ0Ig-Y"
#         self.base_url = "https://api.minimaxi.com/v1"
#         self.model = "MiniMax-M2.5-highspeed"

#     def call(self, prompt: str, max_tokens: int = 4096, temperature: float = None, max_retries: int = 5) -> str:
#         """
#         Call the LLM API with a prompt and return the response text.

#         Args:
#             prompt: The user prompt to send to the LLM
#             max_tokens: Maximum tokens in the response
#             temperature: Sampling temperature (0.0-1.0). Higher values = more random/creative.
#                         None uses the model's default temperature.
#             max_retries: Maximum number of retries for rate limiting errors (default: 5)

#         Returns:
#             The response text from the LLM, or empty string on error
#         """
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }

#         payload = {
#             "model": self.model,
#             "max_tokens": max_tokens,
#             "messages": [{"role": "user", "content": prompt}]
#         }

#         if temperature is not None:
#             payload["temperature"] = temperature

#         # Retry logic with exponential backoff for rate limiting
#         for attempt in range(max_retries):
#             try:
#                 response = requests.post(
#                     f"{self.base_url}/chat/completions",
#                     headers=headers,
#                     json=payload,
#                     timeout=120
#                 )

#                 if response.status_code == 200:
#                     result = response.json()
#                     return result["choices"][0]["message"]["content"]
#                 elif response.status_code == 429 and attempt < max_retries - 1:
#                     wait_time = min(60, (2 ** attempt) * 3)
#                     print(f"Rate limited. Waiting {wait_time}s before retry... (attempt {attempt + 1}/{max_retries})")
#                     time.sleep(wait_time)
#                     continue
#                 else:
#                     print(f"Error calling LLM API: {response.status_code} - {response.text}")
#                     return ""
#             except Exception as e:
#                 error_str = str(e)
#                 if "429" in error_str and attempt < max_retries - 1:
#                     wait_time = min(60, (2 ** attempt) * 3)
#                     print(f"Rate limited. Waiting {wait_time}s before retry... (attempt {attempt + 1}/{max_retries})")
#                     time.sleep(wait_time)
#                     continue
#                 print(f"Error calling LLM API: {e}")
#                 return ""

#         return ""
