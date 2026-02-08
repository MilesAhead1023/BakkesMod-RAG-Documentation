"""
Test Code Generator
===================
Tests RAG-enhanced code generation.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from code_generator import CodeGenerator


def test_generate_simple_plugin():
    """Test generating a simple plugin."""
    print("\n=== Test: Simple Plugin Generation ===\n")

    # Skip if no API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[SKIP] No API key - test requires LLM")
        return

    generator = CodeGenerator()

    requirements = "Create a plugin that logs a message when the match starts"

    result = generator.generate_plugin(requirements)

    # Should have header and implementation
    assert "header" in result
    assert "implementation" in result

    # Should contain basic plugin structure
    assert "class" in result["header"]
    assert "onLoad" in result["header"]
    assert "HookEvent" in result["implementation"]

    print("[OK] Plugin generated")
    print(f"  Header: {len(result['header'])} chars")
    print(f"  Implementation: {len(result['implementation'])} chars")


def test_generate_with_rag_context():
    """Test code generation uses RAG context."""
    print("\n=== Test: RAG Context Integration ===\n")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[SKIP] No API key")
        return

    generator = CodeGenerator()

    requirements = "Create a plugin that hooks the goal scored event"

    result = generator.generate_plugin_with_rag(requirements)

    # Should use correct event name from RAG docs
    assert "Function TAGame.Ball_TA.OnHitGoal" in result["implementation"]

    print("[OK] RAG context used correctly")


if __name__ == "__main__":
    try:
        test_generate_simple_plugin()
        test_generate_with_rag_context()

        print("\n" + "=" * 80)
        print("  ALL CODE GENERATOR TESTS PASSED!")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
