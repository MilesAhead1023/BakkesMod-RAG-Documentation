"""
Test Code Template System
==========================
Tests template generation for BakkesMod plugins.
"""

from code_templates import PluginTemplateEngine


def test_basic_plugin_template():
    """Test generating basic plugin structure."""
    print("\n=== Test: Basic Plugin Template ===\n")

    engine = PluginTemplateEngine()

    result = engine.generate_basic_plugin(
        plugin_name="TestPlugin",
        description="A test plugin"
    )

    # Should generate .h and .cpp files
    assert "header" in result
    assert "implementation" in result
    assert "TestPlugin" in result["header"]
    assert "class TestPlugin" in result["header"]
    assert "void onLoad()" in result["header"]

    print("[OK] Basic template generated")


def test_hook_event_template():
    """Test generating event hook code."""
    print("\n=== Test: Event Hook Template ===\n")

    engine = PluginTemplateEngine()

    code = engine.generate_event_hook(
        event_name="Function TAGame.Ball_TA.OnHitGoal",
        callback_name="onGoalScored"
    )

    assert "HookEvent" in code
    assert "Function TAGame.Ball_TA.OnHitGoal" in code
    assert "onGoalScored" in code

    print("[OK] Event hook template generated")


def test_imgui_window_template():
    """Test generating ImGui window code."""
    print("\n=== Test: ImGui Window Template ===\n")

    engine = PluginTemplateEngine()

    code = engine.generate_imgui_window(
        window_title="Settings",
        elements=["checkbox", "slider"]
    )

    assert "ImGui::Begin" in code
    assert "Settings" in code
    assert "ImGui::End" in code

    print("[OK] ImGui window template generated")


if __name__ == "__main__":
    try:
        test_basic_plugin_template()
        test_hook_event_template()
        test_imgui_window_template()

        print("\n" + "=" * 80)
        print("  ALL TEMPLATE TESTS PASSED!")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
