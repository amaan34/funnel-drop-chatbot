import os
from typing import List, Optional

from openai import OpenAI


class LLMClient:
    """
    Thin wrapper around OpenAI Chat Completions to allow easy swapping of models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 512,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM calls.")
        self.model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(api_key=self.api_key)

    def chat(self, messages: List[dict], temperature: Optional[float] = None) -> str:
        temp = temperature if temperature is not None else self.temperature
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            max_tokens=self.max_output_tokens,
        )
        return response.choices[0].message.content

