"""Provider factory: build an LLMProvider from environment configuration.

Centralizes provider selection so both the Chatbot baseline and the ReAct agent
can switch between OpenAI-compatible, Gemini and local models via DEFAULT_PROVIDER.
"""

import os
from typing import Optional

from src.core.llm_provider import LLMProvider


def get_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """Return an LLMProvider instance based on env / argument.

    Args:
        provider_name: one of 'openai', 'google', 'local'. Falls back to the
            DEFAULT_PROVIDER env var, then 'openai'.
    """
    provider_name = (provider_name or os.getenv("DEFAULT_PROVIDER", "openai")).lower()
    model = os.getenv("DEFAULT_MODEL")

    if provider_name == "openai":
        from src.core.openai_provider import OpenAIProvider

        # OpenAIProvider reads OPENAI_API_KEY / OPENAI_BASE_URL / DEFAULT_MODEL from env.
        return OpenAIProvider()

    if provider_name in ("google", "gemini"):
        from src.core.gemini_provider import GeminiProvider

        return GeminiProvider(
            model_name=model or "gemini-1.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
        )

    if provider_name == "local":
        from src.core.local_provider import LocalProvider

        model_path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")
        return LocalProvider(model_path=model_path)

    raise ValueError(
        f"Unknown provider '{provider_name}'. Use one of: openai, google, local."
    )
