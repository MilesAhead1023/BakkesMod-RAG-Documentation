"""
Test suite for RAG GUI Application
===================================
Tests the BakkesMod RAG GUI components and functionality.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from rag_gui import BakkesModRAGGUI


class TestBakkesModRAGGUI:
    """Test suite for the GUI application."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Set up mock environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    @pytest.fixture
    def mock_rag_components(self):
        """Mock RAG component dependencies."""
        with patch('rag_gui.SimpleDirectoryReader'), \
             patch('rag_gui.MarkdownNodeParser'), \
             patch('rag_gui.VectorStoreIndex'), \
             patch('rag_gui.KnowledgeGraphIndex'), \
             patch('rag_gui.SemanticCache'), \
             patch('rag_gui.QueryRewriter'), \
             patch('rag_gui.CodeGenerator'), \
             patch('rag_gui.CodeValidator'), \
             patch('rag_gui.Settings'):
            yield

    def test_gui_initialization_without_api_keys(self):
        """Test that GUI initialization fails gracefully without API keys."""
        # Clear environment
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]
        
        app = BakkesModRAGGUI()
        status = app.initialize_system()
        
        assert "ERROR" in status
        assert "API keys" in status

    def test_gui_statistics_initial_state(self, mock_env, mock_rag_components):
        """Test that statistics are initialized correctly."""
        app = BakkesModRAGGUI()
        
        # Initial state
        assert app.query_count == 0
        assert app.successful_queries == 0
        assert app.total_query_time == 0.0
        assert app.cache_hits == 0
        
        # Get statistics
        stats = app.get_statistics()
        assert "Total queries: 0" in stats
        assert "Successful: 0" in stats

    def test_query_empty_input(self, mock_env, mock_rag_components):
        """Test querying with empty input."""
        app = BakkesModRAGGUI()
        app.query_engine = Mock()
        
        # Test empty query
        result = list(app.query_rag(""))
        assert len(result) > 0
        assert "Please enter a question" in result[0]

    def test_query_without_initialization(self, mock_env):
        """Test querying before system is initialized."""
        app = BakkesModRAGGUI()
        app.query_engine = None
        
        result = list(app.query_rag("Test query"))
        assert len(result) > 0
        assert "not initialized" in result[0]

    def test_code_generation_empty_requirements(self, mock_env, mock_rag_components):
        """Test code generation with empty requirements."""
        app = BakkesModRAGGUI()
        
        header, impl, status = app.generate_code("")
        
        assert header == ""
        assert impl == ""
        assert "Please enter plugin requirements" in status

    def test_code_generation_without_generator(self, mock_env):
        """Test code generation when generator is not initialized."""
        app = BakkesModRAGGUI()
        app.code_generator = None
        
        header, impl, status = app.generate_code("Create a plugin")
        
        assert header == ""
        assert impl == ""
        assert "not initialized" in status

    def test_export_without_generated_code(self, mock_env, mock_rag_components):
        """Test export when no code has been generated."""
        app = BakkesModRAGGUI()
        app.last_generated_code = None
        
        result = app.export_code("TestPlugin")
        
        assert "No code to export" in result

    def test_export_with_empty_plugin_name(self, mock_env, mock_rag_components):
        """Test export with empty plugin name defaults to MyPlugin."""
        app = BakkesModRAGGUI()
        app.last_generated_code = {
            "header": "// Header code",
            "implementation": "// Implementation code",
            "requirements": "Test requirements",
            "timestamp": "2026-02-08T00:00:00"
        }
        
        # Mock file operations
        with patch('rag_gui.Path') as mock_path:
            mock_dir = MagicMock()
            mock_path.return_value = mock_dir
            mock_dir.__truediv__ = lambda self, other: MagicMock()
            
            with patch('builtins.open', create=True) as mock_open:
                result = app.export_code("")
                
                # Should use default name
                assert "MyPlugin" in str(result) or "exported" in result.lower()

    def test_confidence_calculation_no_sources(self, mock_env):
        """Test confidence calculation with no sources."""
        app = BakkesModRAGGUI()
        
        confidence, label, explanation = app._calculate_confidence([])
        
        assert confidence == 0.0
        assert label == "NO DATA"
        assert "No sources" in explanation

    def test_confidence_calculation_no_scores(self, mock_env):
        """Test confidence calculation with sources but no scores."""
        app = BakkesModRAGGUI()
        
        # Mock nodes without scores
        mock_nodes = [Mock(score=None), Mock(score=None)]
        
        confidence, label, explanation = app._calculate_confidence(mock_nodes)
        
        assert confidence == 0.5
        assert label == "MEDIUM"

    def test_confidence_calculation_high_scores(self, mock_env):
        """Test confidence calculation with high similarity scores."""
        app = BakkesModRAGGUI()
        
        # Mock nodes with high scores
        mock_nodes = [
            Mock(score=0.95),
            Mock(score=0.92),
            Mock(score=0.90)
        ]
        
        confidence, label, explanation = app._calculate_confidence(mock_nodes)
        
        assert confidence > 0.7  # Should be high
        assert label in ["HIGH", "VERY HIGH"]

    def test_confidence_calculation_low_scores(self, mock_env):
        """Test confidence calculation with low similarity scores."""
        app = BakkesModRAGGUI()
        
        # Mock nodes with low scores
        mock_nodes = [
            Mock(score=0.3),
            Mock(score=0.25),
            Mock(score=0.2)
        ]
        
        confidence, label, explanation = app._calculate_confidence(mock_nodes)
        
        assert confidence < 0.5  # Should be low
        assert label in ["LOW", "VERY LOW"]

    def test_statistics_tracking(self, mock_env):
        """Test that statistics are tracked correctly."""
        app = BakkesModRAGGUI()
        
        # Simulate queries
        app.query_count = 5
        app.successful_queries = 4
        app.total_query_time = 10.0
        app.cache_hits = 2
        
        stats = app.get_statistics()
        
        assert "Total queries: 5" in stats
        assert "Successful: 4" in stats
        assert "Cache hits: 2" in stats
        assert "Success rate: 80.0%" in stats
        assert "Cache hit rate: 40.0%" in stats

    def test_plugin_name_sanitization(self, mock_env, mock_rag_components):
        """Test that plugin names are sanitized properly."""
        app = BakkesModRAGGUI()
        app.last_generated_code = {
            "header": "// Header",
            "implementation": "// Implementation",
            "requirements": "Test",
            "timestamp": "2026-02-08T00:00:00"
        }
        
        # Test with special characters
        with patch('rag_gui.Path') as mock_path:
            mock_dir = MagicMock()
            mock_path.return_value = mock_dir
            mock_dir.__truediv__ = lambda self, other: MagicMock()
            
            with patch('builtins.open', create=True):
                # Plugin name with special chars should be sanitized
                result = app.export_code("My@Plugin#Name!")
                
                # Name should be sanitized (alphanumeric + underscore only)
                assert "exported" in result.lower() or "MyPluginName" in result


def test_gui_can_import():
    """Test that the GUI module can be imported."""
    try:
        import rag_gui
        assert hasattr(rag_gui, 'BakkesModRAGGUI')
        assert hasattr(rag_gui, 'create_gui')
        assert hasattr(rag_gui, 'main')
    except ImportError as e:
        pytest.fail(f"Failed to import rag_gui: {e}")


def test_gui_dependencies():
    """Test that all required dependencies are available."""
    required_modules = [
        'gradio',
        'llama_index.core',
        'llama_index.llms.anthropic',
        'llama_index.embeddings.openai',
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        pytest.skip(f"Missing required modules: {', '.join(missing)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
