"""
Test LLM Fallback Chain
========================
Quick test to verify automatic LLM fallback works.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_interactive_rag_fallback():
    """Test that interactive_rag.py initializes with fallback."""
    print("\n=== Testing interactive_rag.py LLM Fallback ===\n")

    # Simulate Anthropic being unavailable
    original_key = os.environ.get("ANTHROPIC_API_KEY")
    try:
        # Remove Anthropic key temporarily
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        print("[TEST] Removed ANTHROPIC_API_KEY to simulate API failure")
        print("[TEST] Expected: Should fallback to Google Gemini\n")

        # Import and test
        from llama_index.core import Settings
        from interactive_rag import build_rag_system

        # This should fallback to Gemini
        rag = build_rag_system()

        print(f"\n[RESULT] LLM configured: {Settings.llm}")
        print(f"[RESULT] Model class: {type(Settings.llm).__name__}")

        # Verify it's not Anthropic
        assert "Anthropic" not in type(Settings.llm).__name__, "Should have fallen back from Anthropic"
        print("\n✅ Fallback test PASSED - System works without Anthropic!")

    finally:
        # Restore original key
        if original_key:
            os.environ["ANTHROPIC_API_KEY"] = original_key

def test_code_generator_fallback():
    """Test that code_generator.py initializes with fallback."""
    print("\n=== Testing code_generator.py LLM Fallback ===\n")

    original_key = os.environ.get("ANTHROPIC_API_KEY")
    try:
        # Remove Anthropic key
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        print("[TEST] Removed ANTHROPIC_API_KEY to simulate API failure")
        print("[TEST] Expected: Should fallback to Google Gemini\n")

        from code_generator import CodeGenerator
        from llama_index.core import Settings

        gen = CodeGenerator()

        print(f"\n[RESULT] LLM configured: {Settings.llm}")
        print(f"[RESULT] Model class: {type(Settings.llm).__name__}")

        assert "Anthropic" not in type(Settings.llm).__name__, "Should have fallen back from Anthropic"
        print("\n✅ Fallback test PASSED - Code generator works without Anthropic!")

    finally:
        if original_key:
            os.environ["ANTHROPIC_API_KEY"] = original_key

if __name__ == "__main__":
    print("Testing LLM automatic fallback chain...")
    print("=" * 60)

    try:
        test_interactive_rag_fallback()
    except Exception as e:
        print(f"\n❌ interactive_rag fallback test FAILED: {e}")

    try:
        test_code_generator_fallback()
    except Exception as e:
        print(f"\n❌ code_generator fallback test FAILED: {e}")

    print("\n" + "=" * 60)
    print("Fallback testing complete!")
