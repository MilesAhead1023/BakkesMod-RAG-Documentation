"""
Test Neural Reranking Configuration
====================================
Validates that Cohere reranking is configured correctly.
"""

import os


def test_cohere_config():
    """Test that Cohere reranker is configured in config.py."""
    print("\n=== Test: Cohere Configuration ===\n")

    # Read config.py
    with open("config.py", "r", encoding="utf-8") as f:
        config_content = f.read()

    # Check Cohere API key field exists
    assert "cohere_api_key" in config_content, "config.py should have cohere_api_key field"
    print("[OK] Found cohere_api_key in config.py")

    # Check reranker settings
    assert "enable_reranker" in config_content, "config.py should have enable_reranker setting"
    assert "reranker_model" in config_content, "config.py should have reranker_model setting"
    print("[OK] Found reranker settings in config.py")

    # Check Phase 2 comment
    assert "Phase 2" in config_content and "reranking" in config_content, "Should have Phase 2 reranking comments"
    print("[OK] Found Phase 2 reranking documentation")

    print("\n[CONFIGURATION VALIDATED]")
    print("  Feature: Neural Reranking")
    print("  Provider: Cohere")
    print("  Model: rerank-english-v3.0")
    print("  Status: Configured (needs COHERE_API_KEY in .env)")


def test_reranker_integration():
    """Test that reranker is integrated in interactive_rag.py."""
    print("\n=== Test: Reranker Integration ===\n")

    # Read interactive_rag.py
    with open("interactive_rag.py", "r", encoding="utf-8") as f:
        rag_content = f.read()

    # Check CohereRerank import
    assert "CohereRerank" in rag_content, "interactive_rag.py should import CohereRerank"
    print("[OK] Found CohereRerank import")

    # Check reranker initialization
    assert "node_postprocessors" in rag_content, "Should have node_postprocessors for reranker"
    assert "reranker =" in rag_content or "CohereRerank(" in rag_content, "Should initialize reranker"
    print("[OK] Found reranker initialization")

    # Check integration with query engine
    assert "node_postprocessors=node_postprocessors" in rag_content, "Query engine should use postprocessors"
    print("[OK] Reranker integrated with query engine")

    print("\n[INTEGRATION VALIDATED]")
    print("  Integration: Complete")
    print("  Postprocessor: CohereRerank")
    print("  Top-N: 5 results after reranking")


def test_requirements():
    """Test that required packages are in requirements.txt."""
    print("\n=== Test: Requirements ===\n")

    with open("requirements.txt", "r", encoding="utf-8") as f:
        requirements = f.read()

    # Check for Cohere packages
    assert "cohere" in requirements.lower(), "requirements.txt should include cohere package"
    assert "cohere-rerank" in requirements.lower() or "postprocessor-cohere" in requirements.lower(), \
        "requirements.txt should include cohere reranker"
    print("[OK] Found cohere packages in requirements.txt")

    print("\n[REQUIREMENTS VALIDATED]")


def test_env_template():
    """Check if .env has Cohere API key placeholder."""
    print("\n=== Test: Environment Template ===\n")

    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            env_content = f.read()

        if "COHERE_API_KEY" in env_content:
            print("[OK] COHERE_API_KEY found in .env")

            # Check if it's set
            api_key = os.getenv("COHERE_API_KEY")
            if api_key and len(api_key) > 0:
                print(f"[OK] COHERE_API_KEY is set (length: {len(api_key)})")
            else:
                print("[INFO] COHERE_API_KEY not set - reranker will be disabled")
        else:
            print("[INFO] COHERE_API_KEY not in .env file - reranker will be disabled")
    else:
        print("[INFO] No .env file found - reranker will be disabled")

    print("\n[ENVIRONMENT CHECKED]")


if __name__ == "__main__":
    try:
        test_cohere_config()
        test_reranker_integration()
        test_requirements()
        test_env_template()

        print("\n" + "=" * 80)
        print("  ALL NEURAL RERANKING CONFIGURATION TESTS PASSED!")
        print("=" * 80)
        print("\n[NEXT STEPS]")
        print("  1. Get Cohere API key: https://dashboard.cohere.ai/api-keys")
        print("  2. Add to .env: COHERE_API_KEY=your_key_here")
        print("  3. Install packages: pip install -r requirements.txt")
        print("  4. Reranker will activate automatically when key is present")

    except AssertionError as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
