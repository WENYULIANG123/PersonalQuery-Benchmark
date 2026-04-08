#!/usr/bin/env python3
"""LLM Clients (MiniMax only, using Anthropic SDK)."""

import time
from typing import Optional


class MiniMaxAnthropicClient:
    """MiniMax LLM Client using Anthropic SDK.

    Uses the Anthropic-compatible API endpoint at https://api.minimaxi.com/anthropic
    Supports MiniMax-M2.7 and other models with thinking/text content blocks.
    """
    def __init__(self, model: str = "MiniMax-M2.7"):
        self.model = model
        # 设置环境变量
        import os
        os.environ["ANTHROPIC_BASE_URL"] = "https://api.minimaxi.com/anthropic"
        os.environ["ANTHROPIC_API_KEY"] = "sk-cp-jqg2XWIob99HfZTveS5CqjO1h8BAQguTCcHG0p_vZlQ_rNqJgQLqNMwJ7AHMMwRhogi2I8A7o9FZ-f1dR2jsVNfwUsdLzicgrXm9tM8bqodav3ZhtQ0Ig-Y"

        import anthropic
        self.client = anthropic.Anthropic()

    def call_with_thinking(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_retries: int = 5,
    ) -> tuple:
        """Call MiniMax API and return both thinking and text.

        Returns:
            tuple: (thinking_text, text_content)
        """
        import anthropic

        safe_prompt = prompt.strip() if isinstance(prompt, str) else str(prompt)
        if not safe_prompt:
            return "", ""

        safe_max_tokens = max(128, min(int(max_tokens), 4096))
        safe_temp = temperature if temperature is not None else 0.7

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

                return thinking_text, text_content.strip()

            except Exception as e:
                error_str = str(e)
                if "429" in error_str and attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"[MiniMax-Anthropic] Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                print(f"[MiniMax-Anthropic] Error calling API: {e}")
                return "", ""

        return "", ""
