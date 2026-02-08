"""
Test Code Validator
===================
Tests C++ code validation and syntax checking.
"""

from code_validator import CodeValidator


def test_valid_cpp_syntax():
    """Test validator accepts valid C++ code."""
    print("\n=== Test: Valid C++ Syntax ===\n")

    validator = CodeValidator()

    valid_code = """
void MyPlugin::onLoad() {
    LOG("Plugin loaded");
}
"""

    result = validator.validate_syntax(valid_code)

    assert result["valid"] == True
    assert len(result["errors"]) == 0

    print("[OK] Valid code accepted")


def test_invalid_cpp_syntax():
    """Test validator catches syntax errors."""
    print("\n=== Test: Invalid C++ Syntax ===\n")

    validator = CodeValidator()

    invalid_code = """
void MyPlugin::onLoad() {
    LOG("Unclosed string
}
"""

    result = validator.validate_syntax(invalid_code)

    assert result["valid"] == False
    assert len(result["errors"]) > 0

    print(f"[OK] Caught {len(result['errors'])} syntax errors")


def test_bakkesmod_api_usage():
    """Test validator checks BakkesMod API usage."""
    print("\n=== Test: BakkesMod API Validation ===\n")

    validator = CodeValidator()

    code_with_api = """
void MyPlugin::onLoad() {
    gameWrapper->HookEvent("Function TAGame.Ball_TA.OnHitGoal", callback);
}
"""

    result = validator.validate_bakkesmod_api(code_with_api)

    assert result["uses_gamewrapper"] == True
    assert result["hooks_events"] == True

    print("[OK] API usage validated")


if __name__ == "__main__":
    try:
        test_valid_cpp_syntax()
        test_invalid_cpp_syntax()
        test_bakkesmod_api_usage()

        print("\n" + "=" * 80)
        print("  ALL VALIDATOR TESTS PASSED!")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
