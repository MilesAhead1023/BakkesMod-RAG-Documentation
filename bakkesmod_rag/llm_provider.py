"""
LLM Provider
=============
Single LLM fallback chain with live verification.
Eliminates the 3x duplication across interactive_rag, rag_gui, and code_generator.
"""

import os
import socket
import logging
from typing import Any, Optional, Sequence

from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)

from bakkesmod_rag.config import RAGConfig

logger = logging.getLogger("bakkesmod_rag.llm_provider")

_NULL_LLM_MESSAGE = (
    "No LLM provider is available. "
    "Add an API key in Settings, or install Ollama for offline use."
)


class NullLLM(CustomLLM):
    """A no-op LLM that returns a helpful message instead of crashing.

    Used when all real providers are unavailable and allow_null=True.
    """

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name="null",
            num_output=256,
        )

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(text=_NULL_LLM_MESSAGE)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        def gen():
            yield CompletionResponse(text=_NULL_LLM_MESSAGE, delta=_NULL_LLM_MESSAGE)
        return gen()

    @classmethod
    def class_name(cls) -> str:
        return "null_llm"


def _try_llm(llm, name: str) -> bool:
    """Verify an LLM works by making a tiny test call.

    Args:
        llm: LlamaIndex LLM instance.
        name: Human-readable provider name for logging.

    Returns:
        True if the LLM responds successfully.
    """
    try:
        response = llm.complete("Say OK")
        if response and response.text:
            return True
    except Exception as e:
        logger.warning("%s test call failed: %s", name, str(e)[:120])
    return False


def _is_google_auth_error(error: Exception) -> bool:
    """Check if a Google API error is an auth/key error (not quota).

    Args:
        error: The exception to check.

    Returns:
        True if the error indicates an authentication or API key problem.
    """
    msg = str(error).lower()
    return any(keyword in msg for keyword in ("invalid", "unauthorized", "api key"))


def get_llm(config: Optional[RAGConfig] = None, allow_null: bool = False):
    """Get a verified LLM using the fallback chain.

    Priority (quality-first):
        1. Anthropic Claude Sonnet (premium)
        2. OpenAI GPT-4o (best non-Anthropic)
        3. Google Gemini 2.5 Pro (high quality paid)
        4. OpenRouter / DeepSeek V3 (FREE, strong quality)
        5. Google Gemini 2.5 Flash (FREE tier, fast)
        6. Ollama local model (offline, no config needed)

    Each provider is tested with a real API call to catch expired credits.

    Args:
        config: RAGConfig instance (optional, uses defaults if None).
        allow_null: If True, return NullLLM instead of raising when all
            providers fail.

    Returns:
        A verified LlamaIndex LLM instance, or NullLLM if allow_null=True
        and no provider is available.

    Raises:
        RuntimeError: If no LLM provider is available and allow_null=False.
    """
    if config is None:
        from bakkesmod_rag.config import get_config
        config = get_config()

    providers = []
    google_auth_failed = False

    # 1. Anthropic
    if config.anthropic_api_key:
        try:
            from llama_index.llms.anthropic import Anthropic

            providers.append((
                "Anthropic Claude Sonnet (primary)",
                Anthropic(
                    model=config.llm.fallback_models.get(
                        "anthropic", "claude-sonnet-4-5"
                    ),
                    max_retries=1,
                    temperature=config.llm.temperature,
                ),
            ))
        except Exception as e:
            logger.warning("Could not initialize Anthropic: %s", e)

    # 2. OpenAI GPT-4o (upgraded from mini)
    if config.openai_api_key:
        try:
            from llama_index.llms.openai import OpenAI as OpenAILLM

            providers.append((
                "OpenAI GPT-4o (high quality)",
                OpenAILLM(
                    model=config.llm.fallback_models.get("openai", "gpt-4o"),
                    temperature=config.llm.temperature,
                ),
            ))
        except Exception as e:
            logger.warning("Could not initialize OpenAI LLM: %s", e)

    # 3. Google Gemini 2.5 Pro (paid, high quality)
    if config.google_api_key:
        try:
            from llama_index.llms.google_genai import GoogleGenAI

            providers.append((
                "Google Gemini 2.5 Pro (paid)",
                GoogleGenAI(
                    model=config.llm.fallback_models.get(
                        "gemini_pro", "gemini-2.5-pro"
                    ),
                    temperature=config.llm.temperature,
                ),
            ))
        except Exception as e:
            logger.warning("Could not initialize Gemini Pro: %s", e)

    # 4. OpenRouter / DeepSeek V3
    if config.openrouter_api_key:
        try:
            from llama_index.llms.openrouter import OpenRouter

            providers.append((
                "OpenRouter / DeepSeek V3 (FREE)",
                OpenRouter(
                    model=config.llm.fallback_models.get(
                        "openrouter", "deepseek/deepseek-chat-v3-0324"
                    ),
                    api_key=config.openrouter_api_key,
                    temperature=config.llm.temperature,
                ),
            ))
        except Exception as e:
            logger.warning("Could not initialize OpenRouter: %s", e)

    # 5. Google Gemini 2.5 Flash (FREE tier)
    if config.google_api_key:
        try:
            from llama_index.llms.google_genai import GoogleGenAI

            providers.append((
                "Google Gemini 2.5 Flash (FREE)",
                GoogleGenAI(
                    model=config.llm.fallback_models.get(
                        "gemini", "gemini-2.5-flash"
                    ),
                    temperature=config.llm.temperature,
                ),
            ))
        except Exception as e:
            logger.warning("Could not initialize Gemini Flash: %s", e)

    # Try each provider with a live verification call
    for name, llm in providers:
        # Gemini auth short-circuit: skip Flash if Pro already had auth failure
        if google_auth_failed and "Gemini 2.5 Flash" in name:
            logger.info(
                "Skipping %s — Google auth already failed on Gemini Pro", name
            )
            continue

        logger.info("Trying %s...", name)
        print(f"[LLM] Trying {name}...")

        try:
            response = llm.complete("Say OK")
            if response and response.text:
                print(f"[LLM] Using {name}")
                logger.info("Using %s", name)
                return llm
        except Exception as e:
            logger.warning("%s test call failed: %s", name, str(e)[:120])
            # Detect Google auth errors to short-circuit Flash later
            if "Gemini 2.5 Pro" in name and _is_google_auth_error(e):
                google_auth_failed = True
                logger.info(
                    "Google auth error detected — will skip Gemini Flash"
                )

    # 6. Ollama local model (auto-detected, no config needed)
    try:
        logger.info("Trying Ollama (local)...")
        print("[LLM] Trying Ollama (local)...")
        with socket.create_connection(("localhost", 11434), timeout=2.0):
            pass
        from llama_index.llms.ollama import Ollama

        llm = Ollama(
            model="llama3.2",
            base_url="http://localhost:11434",
            request_timeout=10.0,
        )
        llm.complete("Say OK")
        print("[LLM] Using Ollama llama3.2 (local, offline)")
        logger.info("Using Ollama llama3.2 (local, offline)")
        return llm
    except Exception:
        logger.info("Ollama not available — skipping")

    # No provider available
    if allow_null:
        logger.warning("No LLM provider available — returning NullLLM")
        print("[LLM] No provider available — using NullLLM (limited mode)")
        return NullLLM()

    raise RuntimeError(
        "No LLM provider available! "
        "Set ANTHROPIC_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY, "
        "or OPENAI_API_KEY, or install Ollama for offline use."
    )


def get_embed_model(config: Optional[RAGConfig] = None):
    """Get the embedding model.

    Args:
        config: RAGConfig instance (optional, uses defaults if None).

    Returns:
        A LlamaIndex embedding model instance.
    """
    if config is None:
        from bakkesmod_rag.config import get_config
        config = get_config()

    from llama_index.embeddings.openai import OpenAIEmbedding

    return OpenAIEmbedding(
        model=config.embedding.model,
        max_retries=config.embedding.max_retries,
    )
