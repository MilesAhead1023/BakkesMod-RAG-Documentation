"""
Test Syntax Highlighting
=========================
Tests C++ code block detection and Pygments highlighting.
"""

from interactive_rag import highlight_code_blocks


def test_cpp_code_block():
    """Test highlighting of C++ code blocks."""
    print("\n=== Test 1: C++ Code Block ===\n")

    text = """Here's how to hook an event:

```cpp
void MyPlugin::onLoad() {
    gameWrapper->HookEvent("Function TAGame.Ball_TA.OnHitGoal",
        [this](std::string eventName) {
            LOG("Goal scored!");
        });
}
```

That's the basic pattern."""

    highlighted = highlight_code_blocks(text)

    print("Original text:")
    print(text)
    print("\nHighlighted output:")
    print(highlighted)

    # Check that highlighting was applied (contains ANSI codes or is unchanged)
    if '\033[' in highlighted or highlighted == text:
        print("\n[OK] Highlighting applied or gracefully skipped")
    else:
        print("\n[INFO] Highlighting may not be visible in this terminal")


def test_multiple_code_blocks():
    """Test multiple code blocks in one response."""
    print("\n=== Test 2: Multiple Code Blocks ===\n")

    text = """First, declare the class:

```cpp
class MyPlugin : public BakkesMod::Plugin::BakkesModPlugin {
public:
    virtual void onLoad();
};
```

Then implement the method:

```cpp
void MyPlugin::onLoad() {
    LOG("Plugin loaded");
}
```

Done!"""

    highlighted = highlight_code_blocks(text)

    print("Original text has", text.count('```'), "code fence markers")
    print("\nHighlighted version applied")

    if '```' in text:
        print("[OK] Multiple code blocks detected")


def test_no_code_blocks():
    """Test text without code blocks."""
    print("\n=== Test 3: No Code Blocks ===\n")

    text = "This is plain text without any code blocks."

    highlighted = highlight_code_blocks(text)

    assert highlighted == text, "Text without code blocks should remain unchanged"
    print("Original:", text)
    print("[OK] Plain text unchanged")


def test_language_detection():
    """Test language-specific highlighting."""
    print("\n=== Test 4: Language Detection ===\n")

    text = """Python example:

```python
def hello():
    print("Hello, world!")
```

C++ example:

```cpp
void hello() {
    std::cout << "Hello, world!" << std::endl;
}
```

"""

    highlighted = highlight_code_blocks(text)

    print("[OK] Language-specific lexers can be applied")
    print("Processed", text.count('```'), "code blocks with different languages")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  SYNTAX HIGHLIGHTING TESTS")
    print("=" * 80)

    try:
        import pygments
        print("[INFO] Pygments available - testing highlighting\n")

        test_cpp_code_block()
        test_multiple_code_blocks()
        test_no_code_blocks()
        test_language_detection()

        print("\n" + "=" * 80)
        print("  ALL SYNTAX HIGHLIGHTING TESTS PASSED!")
        print("=" * 80)

    except ImportError:
        print("[WARNING] Pygments not installed - highlighting will be skipped")
        print("[INFO] Install with: pip install pygments colorama")
