"""
Quick RAG System Test
=====================
Tests the 2026 Gold Standard RAG with a simple query.
This verifies your setup is working correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_env():
    """Check if at least one API key is configured."""
    print("=" * 60)
    print("STEP 1: Environment Check")
    print("=" * 60)

    keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY"),
    }

    configured = []
    for provider, key in keys.items():
        if key and key != f"your_{provider.lower()}_key_here":
            configured.append(provider)
            print(f"[OK] {provider}: Configured")
        else:
            print(f"[X] {provider}: Not configured")

    if not configured:
        print("\n[ERROR] ERROR: No API keys configured!")
        print("Please edit .env and add at least one API key.")
        return False

    print(f"\n[OK] Found {len(configured)} configured provider(s): {', '.join(configured)}")
    return True


def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "=" * 60)
    print("STEP 2: Import Test")
    print("=" * 60)

    try:
        print("Importing config...")
        from config import get_config

        print("Importing observability...")
        from observability import initialize_observability, get_logger

        print("Importing cost tracker...")
        from cost_tracker import get_tracker

        print("Importing RAG system...")
        from rag_2026 import build_gold_standard_rag

        print("\n[OK] All imports successful!")
        return True

    except ImportError as e:
        print(f"\n[ERROR] Import failed: {e}")
        print("\nTry running: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\n" + "=" * 60)
    print("STEP 3: Configuration Test")
    print("=" * 60)

    try:
        from config import get_config
        config = get_config()

        print(f"[OK] Embedding Provider: {config.embedding.provider}")
        print(f"[OK] Embedding Model: {config.embedding.model}")
        print(f"[OK] Primary LLM Provider: {config.llm.primary_provider}")
        print(f"[OK] Primary LLM Model: {config.llm.primary_model}")
        print(f"[OK] Storage Directory: {config.storage.storage_dir}")
        print(f"[OK] Docs Directory: {config.storage.docs_dir}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Configuration error: {e}")
        return False


def test_rag_build():
    """Test RAG system initialization."""
    print("\n" + "=" * 60)
    print("STEP 4: RAG System Build Test")
    print("=" * 60)
    print("This may take a while on first run (building indices)...")
    print("Subsequent runs will be faster (loading cached indices).\n")

    try:
        from rag_2026 import build_gold_standard_rag

        print("Building RAG system...")
        rag = build_gold_standard_rag(incremental=True)

        print("\n[OK] RAG system built successfully!")
        return rag

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("\nMake sure the 'docs' directory exists with markdown files.")
        return None
    except Exception as e:
        print(f"\n[ERROR] Build failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_query(rag):
    """Test a simple query."""
    print("\n" + "=" * 60)
    print("STEP 5: Query Test")
    print("=" * 60)

    # Use a simple, common query about BakkesMod
    test_query = "What is BakkesMod?"

    print(f"Query: {test_query}\n")

    try:
        response = rag.query(test_query)

        print("Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)

        print(f"\n[OK] Query completed!")
        print(f"  Sources used: {len(response.source_nodes)}")

        # Show cost estimate
        print("\n" + "=" * 60)
        print("Cost Report")
        print("=" * 60)
        print(rag.cost_tracker.get_report())

        return True

    except Exception as e:
        print(f"\n[ERROR] Query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("    2026 Gold Standard RAG - Quick Test")
    print("=" * 60 + "\n")

    # Step 1: Check environment
    if not check_env():
        sys.exit(1)

    # Step 2: Test imports
    if not test_imports():
        sys.exit(1)

    # Step 3: Test configuration
    if not test_config():
        sys.exit(1)

    # Step 4: Build RAG system
    rag = test_rag_build()
    if not rag:
        sys.exit(1)

    # Step 5: Test query
    if not test_query(rag):
        sys.exit(1)

    # Success!
    print("\n" + "=" * 60)
    print("          ALL TESTS PASSED!")
    print("=" * 60 + "\n")
    print("Your RAG system is working correctly!")
    print("\nNext steps:")
    print("  • Run: python rag_2026.py (interactive mode)")
    print("  • Run: python rag_2026.py 'your question here' (single query)")
    print("  • Check logs in ./logs/ directory")
    print("  • View metrics at http://localhost:8000/metrics (if enabled)")


if __name__ == "__main__":
    main()
