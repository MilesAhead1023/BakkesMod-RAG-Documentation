"""
Code Generation Integration Test
=================================
End-to-end test of code generation mode.
"""

import os
from dotenv import load_dotenv
load_dotenv()


def test_full_code_generation_workflow():
    """Test complete code generation workflow."""
    print("\n=== Integration Test: Code Generation ===\n")

    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("[SKIP] Missing API keys (ANTHROPIC_API_KEY and/or OPENAI_API_KEY)")
        print("\n" + "=" * 80)
        print("  INTEGRATION TEST SKIPPED (No API keys)")
        print("=" * 80)
        return

    from code_generator import CodeGenerator

    generator = CodeGenerator()

    # Test 1: Simple plugin
    print("Test 1: Simple plugin generation...")
    result1 = generator.generate_plugin_with_rag(
        "Create a plugin that logs when the match starts"
    )
    assert "header" in result1
    assert "implementation" in result1
    print("[OK] Simple plugin generated\n")

    # Test 2: Complex plugin with ImGui
    print("Test 2: Complex plugin with UI...")
    result2 = generator.generate_complete_project(
        "Create a plugin that tracks player boost and shows it in a window"
    )
    assert "cmake" in result2
    assert "readme" in result2
    print("[OK] Complete project generated\n")

    # Test 3: Validate syntax
    print("Test 3: Code validation...")
    from code_validator import CodeValidator
    validator = CodeValidator()

    validation = validator.validate_syntax(result2["implementation"])
    print(f"  Syntax valid: {validation['valid']}")
    if not validation['valid']:
        print(f"  Errors: {validation['errors']}")

    api_check = validator.validate_bakkesmod_api(result2["implementation"])
    print(f"  Uses gameWrapper: {api_check['uses_gamewrapper']}")
    print(f"  Hooks events: {api_check['hooks_events']}")
    print("[OK] Validation complete\n")

    print("=" * 80)
    print("  INTEGRATION TEST COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_full_code_generation_workflow()
    except Exception as e:
        print(f"\n[ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
