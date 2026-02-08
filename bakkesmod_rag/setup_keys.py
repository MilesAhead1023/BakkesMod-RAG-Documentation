"""
API Key Onboarding
==================
Interactive prompts for missing API keys at startup.
Saves to .env so it's one-time setup only.

Uses getpass for masked input and python-dotenv set_key for persistence.
Verified against:
  - Python docs: https://docs.python.org/3/library/getpass.html
  - python-dotenv: https://github.com/theskumar/python-dotenv
"""

import os
import getpass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv, set_key

logger = logging.getLogger("bakkesmod_rag.setup_keys")

# Each key: (required, format_prefix, description)
KEY_DEFINITIONS: Dict[str, Tuple[bool, Optional[str], str]] = {
    "OPENAI_API_KEY": (
        True,
        "sk-",
        "Required for embeddings (text-embedding-3-small). No alternative.",
    ),
    "ANTHROPIC_API_KEY": (
        False,
        "sk-ant-",
        "Claude Sonnet — premium LLM for highest quality answers.",
    ),
    "OPENROUTER_API_KEY": (
        False,
        "sk-or-",
        "DeepSeek V3 via OpenRouter — FREE LLM fallback.",
    ),
    "GOOGLE_API_KEY": (
        False,
        "AI",
        "Gemini 2.5 Flash — FREE LLM fallback.",
    ),
    "COHERE_API_KEY": (
        False,
        None,
        "Neural reranking (fallback after free BAAI/FlashRank rerankers).",
    ),
}


def mask_key(key: str) -> str:
    """Mask an API key, showing only the last 4 characters.

    Args:
        key: The full API key string.

    Returns:
        Masked string like ``sk-...AbCd``.
    """
    if len(key) <= 4:
        return "****"
    prefix = key[:3]
    suffix = key[-4:]
    return f"{prefix}...{suffix}"


def validate_key_format(name: str, value: str) -> bool:
    """Check if a key matches its expected format prefix.

    Warns but does not block — key formats may change over time.

    Args:
        name: Key name (e.g. ``OPENAI_API_KEY``).
        value: The key value to validate.

    Returns:
        True if format matches or no format defined, False otherwise.
    """
    _, prefix, _ = KEY_DEFINITIONS.get(name, (False, None, ""))
    if prefix is None:
        return True
    if value.startswith(prefix):
        return True
    logger.warning(
        "Key %s doesn't match expected prefix '%s' — accepting anyway", name, prefix
    )
    return False


def check_missing_keys() -> List[str]:
    """Return names of API keys not set in the environment.

    Returns:
        List of missing key names in definition order.
    """
    missing = []
    for name in KEY_DEFINITIONS:
        if not os.environ.get(name):
            missing.append(name)
    return missing


def prompt_for_key(name: str) -> Optional[str]:
    """Interactively prompt the user for a single API key.

    Args:
        name: The environment variable name.

    Returns:
        The entered key string, or None if the user skipped (optional keys).
    """
    required, _, description = KEY_DEFINITIONS[name]
    label = "REQUIRED" if required else "optional"

    print(f"\n  {name} [{label}]")
    print(f"  → {description}")

    if required:
        prompt_text = f"  Enter {name}: "
    else:
        prompt_text = f"  Enter {name} (or press Enter to skip): "

    try:
        value = getpass.getpass(prompt=prompt_text)
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    value = value.strip()
    if not value:
        if required:
            print("  ⚠ This key is required. Please provide it.")
            return prompt_for_key(name)
        return None

    valid = validate_key_format(name, value)
    if not valid:
        print(f"  ⚠ Format warning: expected prefix for {name} not found (saved anyway)")

    print(f"  ✓ Saved: {mask_key(value)}")
    return value


def save_keys(keys: Dict[str, str], dotenv_path: Optional[str] = None) -> None:
    """Persist keys to .env file using python-dotenv set_key.

    Creates the .env file if it doesn't exist. Existing keys are updated,
    new keys are appended.

    Args:
        keys: Dict of {KEY_NAME: value} to save.
        dotenv_path: Path to .env file. Defaults to .env in cwd.
    """
    if dotenv_path is None:
        dotenv_path = str(Path.cwd() / ".env")

    # Create file if missing
    env_file = Path(dotenv_path)
    if not env_file.exists():
        env_file.touch()
        logger.info("Created new .env file at %s", dotenv_path)

    for name, value in keys.items():
        set_key(dotenv_path, name, value)
        os.environ[name] = value
        logger.info("Saved %s to .env", name)


def ensure_api_keys(dotenv_path: Optional[str] = None) -> bool:
    """Main entry point: check for missing keys, prompt if needed, save to .env.

    Call this before RAGEngine initialization. If all keys are present,
    this is a no-op (zero friction for returning users).

    Args:
        dotenv_path: Path to .env file. Defaults to .env in cwd.

    Returns:
        True if all required keys are available after prompting,
        False if a required key is still missing (user ctrl+C'd).
    """
    if dotenv_path is None:
        dotenv_path = str(Path.cwd() / ".env")

    # Load existing .env first
    load_dotenv(dotenv_path, override=False)

    missing = check_missing_keys()
    if not missing:
        logger.info("All API keys present — skipping onboarding")
        return True

    # Check if any required keys are missing
    has_required_missing = any(
        KEY_DEFINITIONS[name][0] for name in missing
    )
    has_optional_missing = any(
        not KEY_DEFINITIONS[name][0] for name in missing
    )

    print("\n" + "=" * 60)
    print("  API KEY SETUP")
    print("=" * 60)

    if has_required_missing:
        print("\n  Some required API keys are missing.")
    else:
        print("\n  Optional API keys can improve functionality.")
    print("  Keys are saved to .env and only needed once.\n")

    collected: Dict[str, str] = {}
    for name in missing:
        value = prompt_for_key(name)
        if value:
            collected[name] = value

    if collected:
        save_keys(collected, dotenv_path)
        print(f"\n  ✓ {len(collected)} key(s) saved to .env")

        # Reload config to pick up new keys
        from bakkesmod_rag.config import reload_config
        reload_config()
        logger.info("Config reloaded after key onboarding")

    # Check if required keys are now available
    still_missing = [
        name for name in check_missing_keys()
        if KEY_DEFINITIONS[name][0]  # required only
    ]
    if still_missing:
        print(f"\n  ⚠ Required key(s) still missing: {', '.join(still_missing)}")
        print("  The system may not function correctly.\n")
        return False

    print("\n" + "=" * 60)
    print("  ✓ API key setup complete!")
    print("=" * 60 + "\n")
    return True
