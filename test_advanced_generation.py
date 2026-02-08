"""
Test Advanced Code Generation
==============================
Tests multi-file project generation and advanced features.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from code_generator import CodeGenerator


def test_generate_complete_project():
    """Test generating complete plugin project."""
    print("\n=== Test: Complete Project Generation ===\n")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[SKIP] No API key")
        return

    generator = CodeGenerator()

    requirements = """
    Create a plugin that:
    1. Tracks boost usage per player
    2. Shows boost stats in an ImGui window
    3. Saves stats to a config file
    """

    result = generator.generate_complete_project(requirements)

    # Should generate multiple files
    assert "header" in result
    assert "implementation" in result
    assert "cmake" in result

    # Should have ImGui code
    assert "ImGui::Begin" in result["implementation"]

    print("[OK] Complete project generated")
    print(f"  Files: {list(result.keys())}")


def test_generate_with_imgui():
    """Test ImGui UI generation."""
    print("\n=== Test: ImGui UI Generation ===\n")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[SKIP] No API key")
        return

    generator = CodeGenerator()

    requirements = "Create a settings window with a checkbox to enable/disable the plugin"

    result = generator.generate_imgui_window(requirements)

    assert "ImGui::Begin" in result
    assert "ImGui::Checkbox" in result
    assert "ImGui::End" in result

    print("[OK] ImGui window generated")


if __name__ == "__main__":
    try:
        test_generate_complete_project()
        test_generate_with_imgui()

        print("\n" + "=" * 80)
        print("  ALL ADVANCED GENERATION TESTS PASSED!")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
