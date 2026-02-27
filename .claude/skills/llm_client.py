#!/usr/bin/env python3
"""
Shared LLM Client for skill scripts.
Provides a unified interface to call the GLM API via Anthropic SDK.
"""
import anthropic


class LLMClient:
    """Client for interacting with the LLM API."""

    def __init__(self):
        self.client = anthropic.Anthropic(
            base_url="https://api.z.ai/api/anthropic",
            api_key="db2682f8a0024278a672f762ce36d7cd.RC8PtxIy5xdlh8Uj"
        )
        # self.model = "GLM-4.5-Air"  # Original model
        self.model = "GLM-5"  # Testing GLM-5

    def call(self, prompt: str, max_tokens: int = 4096, temperature: float = None) -> str:
        """
        Call the LLM API with a prompt and return the response text.

        Args:
            prompt: The user prompt to send to the LLM
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0-1.0). Higher values = more random/creative.
                        None uses the model's default temperature.

        Returns:
            The response text from the LLM, or empty string on error
        """
        try:
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            # Only add temperature if specified
            if temperature is not None:
                kwargs["temperature"] = temperature

            response = self.client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return ""
