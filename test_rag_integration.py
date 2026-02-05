"""Integration tests for RAG stack - validates API connectivity and basic functionality."""

import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


class TestAPIConnectivity:
    """Test that all required API keys are present and APIs are reachable."""

    def test_openai_api_key_exists(self):
        """Verify OpenAI API key is in environment."""
        assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY not found in environment"
        assert len(os.environ["OPENAI_API_KEY"]) > 20, "OPENAI_API_KEY seems invalid"

    def test_anthropic_api_key_exists(self):
        """Verify Anthropic API key is in environment."""
        assert "ANTHROPIC_API_KEY" in os.environ, "ANTHROPIC_API_KEY not found"
        assert len(os.environ["ANTHROPIC_API_KEY"]) > 20, "ANTHROPIC_API_KEY seems invalid"

    def test_google_api_key_exists(self):
        """Verify Google API key is in environment."""
        assert "GOOGLE_API_KEY" in os.environ, "GOOGLE_API_KEY not found"
        assert len(os.environ["GOOGLE_API_KEY"]) > 20, "GOOGLE_API_KEY seems invalid"

    @pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OpenAI API key not set")
    def test_openai_embedding_api(self):
        """Test OpenAI embeddings API is accessible."""
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(
            input=["test"],
            model="text-embedding-3-small"
        )

        assert response.data[0].embedding is not None
        assert len(response.data[0].embedding) > 0

    @pytest.mark.skipif("ANTHROPIC_API_KEY" not in os.environ, reason="Anthropic API key not set")
    def test_anthropic_api(self):
        """Test Anthropic API is accessible."""
        from anthropic import Anthropic

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert message.content is not None
        assert len(message.content) > 0

    @pytest.mark.skipif("GOOGLE_API_KEY" not in os.environ, reason="Google API key not set")
    def test_gemini_api(self):
        """Test Gemini API is accessible."""
        from google import genai

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents='Hi'
        )

        assert response.text is not None
        assert len(response.text) > 0


class TestRAGBuilder:
    """Test RAG builder functionality."""

    def test_docs_directory_exists(self):
        """Verify docs directory exists."""
        assert os.path.exists("./docs"), "docs/ directory not found"

    def test_markdown_files_exist(self):
        """Verify markdown files exist in docs."""
        docs_path = Path("./docs")
        md_files = list(docs_path.rglob("*.md"))
        assert len(md_files) > 0, "No markdown files found in docs/"

    @pytest.mark.skipif(not os.path.exists("./rag_storage"), reason="RAG storage not built yet")
    def test_rag_storage_exists(self):
        """Verify RAG storage directory was created."""
        assert os.path.exists("./rag_storage/docstore.json")
        assert os.path.exists("./rag_storage/default__vector_store.json")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
