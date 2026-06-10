from __future__ import annotations

import logging

import httpx

from firewall.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin client for victim LLM inference via Groq or mock mode."""

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.provider = provider or settings.victim_llm_provider
        self.model = model or settings.victim_llm_model
        self.api_key = settings.groq_api_key

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        if self.provider == "mock" or not self.api_key:
            return self._mock_response(prompt)
        if self.provider == "groq":
            return await self._groq_generate(prompt, system_prompt)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    async def _groq_generate(self, prompt: str, system_prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 512,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def _mock_response(self, prompt: str) -> str:
        lowered = prompt.lower()
        if any(k in lowered for k in ("ignore", "jailbreak", "system prompt", "dan")):
            return (
                "I cannot comply with requests that attempt to override my instructions "
                "or extract internal configuration."
            )
        return f"This is a mock response to: {prompt[:120]}"
