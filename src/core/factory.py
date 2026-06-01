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
    default_model = os.getenv("DEFAULT_MODEL", "")

    if provider_name == "openai":
        from src.core.openai_provider import OpenAIProvider

        # OPENAI_MODEL overrides; otherwise reuse DEFAULT_MODEL. OpenAIProvider
        # falls back to its own default when model_name is None.
        model = os.getenv("OPENAI_MODEL") or default_model or None
        return OpenAIProvider(model_name=model)

    if provider_name in ("google", "gemini"):
        from src.core.gemini_provider import GeminiProvider

        # Model names are NOT interchangeable across providers. Only reuse
        # DEFAULT_MODEL for Gemini when it actually looks like a Gemini model,
        # otherwise fall back to a sane Gemini default (override via GEMINI_MODEL).
        model = os.getenv("GEMINI_MODEL")
        if not model:
            model = default_model if default_model.startswith("gemini") else "gemini-2.0-flash"
        return GeminiProvider(
            model_name=model,
            api_key=os.getenv("GEMINI_API_KEY"),
        )

    if provider_name == "local":
        from src.core.local_provider import LocalProvider

        model_path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")
        return LocalProvider(model_path=model_path)

    raise ValueError(
        f"Unknown provider '{provider_name}'. Use one of: openai, google, local."
    )
