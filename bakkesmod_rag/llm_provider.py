"""
LLM Provider
=============
Single LLM fallback chain with live verification.
Eliminates the 3x duplication across interactive_rag, rag_gui, and code_generator.
"""

import os
import logging
from typing import Optional

from bakkesmod_rag.config import RAGConfig

logger = logging.getLogger("bakkesmod_rag.llm_provider")


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


def get_llm(config: Optional[RAGConfig] = None):
    """Get a verified LLM using the fallback chain.

    Priority: Anthropic (premium) -> OpenRouter/DeepSeek (FREE)
              -> Gemini (FREE) -> OpenAI (cheap)

    Each provider is tested with a real API call to catch expired credits.

    Args:
        config: RAGConfig instance (optional, uses defaults if None).

    Returns:
        A verified LlamaIndex LLM instance.

    Raises:
        RuntimeError: If no LLM provider is available.
    """
    if config is None:
        from bakkesmod_rag.config import get_config
        config = get_config()

    providers = []

    # Anthropic
    if config.anthropic_api_key:
        try:
            from llama_index.llms.anthropic import Anthropic

            providers.append((
                "Anthropic Claude Sonnet (primary)",
                Anthropic(
                    model=config.llm.fallback_models.get("anthropic", "claude-sonnet-4-5"),
                    max_retries=1,
                    temperature=config.llm.temperature,
                ),
            ))
        except Exception as e:
            logger.warning("Could not initialize Anthropic: %s", e)

    # OpenRouter / DeepSeek V3
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

    # Google Gemini
    if config.google_api_key:
        try:
            from llama_index.llms.google_genai import GoogleGenAI

            providers.append((
                "Google Gemini 2.5 Flash (FREE)",
                GoogleGenAI(
                    model=config.llm.fallback_models.get("gemini", "gemini-2.5-flash"),
                    temperature=config.llm.temperature,
                ),
            ))
        except Exception as e:
            logger.warning("Could not initialize Gemini: %s", e)

    # OpenAI
    if config.openai_api_key:
        try:
            from llama_index.llms.openai import OpenAI as OpenAILLM

            providers.append((
                "OpenAI GPT-4o-mini (cheap)",
                OpenAILLM(
                    model=config.llm.fallback_models.get("openai", "gpt-4o-mini"),
                    temperature=config.llm.temperature,
                ),
            ))
        except Exception as e:
            logger.warning("Could not initialize OpenAI LLM: %s", e)

    # Try each provider with a live verification call
    for name, llm in providers:
        logger.info("Trying %s...", name)
        print(f"[LLM] Trying {name}...")
        if _try_llm(llm, name):
            print(f"[LLM] Using {name}")
            logger.info("Using %s", name)
            return llm

    raise RuntimeError(
        "No LLM provider available! "
        "Set ANTHROPIC_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY"
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
