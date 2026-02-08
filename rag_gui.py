"""
BakkesMod RAG - Comprehensive GUI Application
==============================================
A professional Gradio-based interface for querying BakkesMod SDK documentation,
generating plugin code, and managing the RAG system.

Features:
- Real-time streaming responses
- Code syntax highlighting
- Source citation viewer
- Confidence scores
- Plugin code generation
- Session statistics
- Export functionality
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Generator
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Gradio
try:
    import gradio as gr
except ImportError:
    print("ERROR: Gradio not installed. Installing...")
    os.system("pip install gradio")
    import gradio as gr

# Import RAG components
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
    Document
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex

# Import additional components
from cache_manager import SemanticCache
from query_rewriter import QueryRewriter
from code_generator import CodeGenerator
from code_validator import CodeValidator


class BakkesModRAGGUI:
    """Main GUI application for BakkesMod RAG system."""

    def __init__(self):
        """Initialize the GUI application."""
        self.query_engine = None
        self.cache = None
        self.query_rewriter = None
        self.code_generator = None
        self.code_validator = None
        self.nodes = []
        self.documents = []
        
        # Session statistics
        self.query_count = 0
        self.successful_queries = 0
        self.total_query_time = 0.0
        self.cache_hits = 0
        
        # Last generated code
        self.last_generated_code = None
        
        # Initialize system
        self.initialize_system()

    def initialize_system(self) -> str:
        """
        Initialize the RAG system.
        
        Returns:
            Status message
        """
        try:
            # Check API keys
            if not os.getenv("OPENAI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
                return "❌ ERROR: Missing API keys! Check your .env file."
            
            # Configure LLM and embeddings
            Settings.embed_model = OpenAIEmbedding(
                model="text-embedding-3-small", 
                max_retries=3
            )
            Settings.llm = Anthropic(
                model="claude-sonnet-4-5", 
                max_retries=3, 
                temperature=0
            )
            
            # Load documents
            docs_dir = "docs_bakkesmod_only"
            if not Path(docs_dir).exists():
                return f"❌ ERROR: Documentation directory '{docs_dir}' not found!"
            
            reader = SimpleDirectoryReader(
                input_dir=docs_dir,
                required_exts=[".md"],
                recursive=True,
                filename_as_id=True
            )
            self.documents = reader.load_data()
            
            # Clean documents
            cleaned_docs = []
            for doc in self.documents:
                clean_text = "".join(
                    filter(lambda x: x.isprintable() or x in "\n\r\t", doc.text)
                )
                cleaned_docs.append(Document(text=clean_text, metadata=doc.metadata))
            self.documents = cleaned_docs
            
            # Parse into nodes
            parser = MarkdownNodeParser()
            self.nodes = parser.get_nodes_from_documents(self.documents)
            
            # Build or load indices
            storage_dir = "rag_storage_bakkesmod"
            storage_path = Path(storage_dir)
            
            if storage_path.exists():
                # Load from cache
                storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
                vector_index = load_index_from_storage(storage_context, index_id="vector")
                
                # Try to load KG index
                try:
                    kg_index = load_index_from_storage(
                        storage_context, 
                        index_id="knowledge_graph"
                    )
                except:
                    # Build KG index if not found
                    kg_index = KnowledgeGraphIndex.from_documents(
                        self.documents,
                        max_triplets_per_chunk=2,
                        show_progress=False
                    )
                    kg_index.set_index_id("knowledge_graph")
                    kg_index.storage_context.persist(persist_dir=storage_dir)
            else:
                # Build new indices
                vector_index = VectorStoreIndex(self.nodes, show_progress=False)
                vector_index.set_index_id("vector")
                storage_path.mkdir(parents=True, exist_ok=True)
                vector_index.storage_context.persist(persist_dir=storage_dir)
                
                # Build KG index
                kg_index = KnowledgeGraphIndex.from_documents(
                    self.documents,
                    max_triplets_per_chunk=2,
                    show_progress=False
                )
                kg_index.set_index_id("knowledge_graph")
                kg_index.storage_context.persist(persist_dir=storage_dir)
            
            # Create retrievers
            vector_retriever = vector_index.as_retriever(similarity_top_k=5)
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=self.nodes, 
                similarity_top_k=5
            )
            kg_retriever = kg_index.as_retriever(similarity_top_k=3)
            
            # Create fusion retriever
            fusion_retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever, kg_retriever],
                num_queries=4,
                mode="reciprocal_rerank",
                use_async=True
            )
            
            # Create query engine
            self.query_engine = RetrieverQueryEngine.from_args(
                fusion_retriever,
                streaming=True
            )
            
            # Initialize cache
            self.cache = SemanticCache(
                cache_dir=".cache/semantic",
                similarity_threshold=0.92,
                ttl_seconds=86400 * 7,
                embed_model=Settings.embed_model
            )
            
            # Initialize query rewriter
            self.query_rewriter = QueryRewriter(
                llm=Settings.llm, 
                use_llm=False
            )
            
            # Initialize code generator and validator
            self.code_generator = CodeGenerator()
            self.code_validator = CodeValidator()
            
            return (f"✅ System initialized successfully!\n"
                   f"📚 Loaded {len(self.documents)} documents\n"
                   f"📝 Processed {len(self.nodes)} searchable chunks\n"
                   f"🔍 Using 3-way fusion: Vector + BM25 + Knowledge Graph")
            
        except Exception as e:
            return f"❌ ERROR during initialization: {str(e)}"

    def query_rag(
        self, 
        query: str, 
        use_cache: bool = True
    ) -> Generator[str, None, None]:
        """
        Query the RAG system with streaming response.
        
        Args:
            query: User question
            use_cache: Whether to use semantic cache
            
        Yields:
            Response chunks
        """
        if not query or not query.strip():
            yield "⚠️ Please enter a question."
            return
        
        if not self.query_engine:
            yield "❌ ERROR: System not initialized. Please restart the application."
            return
        
        start_time = time.time()
        self.query_count += 1
        
        try:
            # Expand query with synonyms
            expanded_query = self.query_rewriter.expand_with_synonyms(query)
            
            # Check cache first
            if use_cache:
                cache_result = self.cache.get(query)
                if cache_result:
                    response_text, similarity, metadata = cache_result
                    query_time = time.time() - start_time
                    self.successful_queries += 1
                    self.total_query_time += query_time
                    self.cache_hits += 1
                    
                    # Format cached response
                    yield f"💾 **CACHED RESPONSE** (similarity: {similarity:.1%})\n\n"
                    yield response_text
                    yield f"\n\n---\n"
                    yield f"⏱️ Query time: {query_time:.2f}s\n"
                    yield f"💰 Cost savings: ~$0.02-0.04\n"
                    yield f"🕐 Cache age: {metadata['cache_age_seconds']/3600:.1f} hours"
                    return
            
            # Execute query
            response = self.query_engine.query(expanded_query)
            
            # Stream the response
            full_text = ""
            yield "🤖 **ANSWER:**\n\n"
            
            for token in response.response_gen:
                full_text += token
                yield token
            
            query_time = time.time() - start_time
            self.successful_queries += 1
            self.total_query_time += query_time
            
            # Cache the response
            if use_cache:
                self.cache.set(query, full_text, response.source_nodes)
            
            # Calculate confidence
            confidence, conf_label, conf_explanation = self._calculate_confidence(
                response.source_nodes
            )
            
            # Add metadata
            yield f"\n\n---\n"
            yield f"⏱️ Query time: {query_time:.2f}s\n"
            yield f"📊 Confidence: {confidence:.0%} ({conf_label}) - {conf_explanation}\n"
            yield f"📚 Sources: {len(response.source_nodes)}\n"
            
            # Add source files
            if response.source_nodes:
                yield f"\n**Source Files:**\n"
                seen_files = set()
                for node in response.source_nodes:
                    filename = node.node.metadata.get("file_name", "unknown")
                    if filename not in seen_files:
                        seen_files.add(filename)
                        yield f"- {filename}\n"
                        
        except Exception as e:
            query_time = time.time() - start_time
            self.total_query_time += query_time
            yield f"\n\n❌ **ERROR:** {str(e)}\n"
            yield "Please try rephrasing your question or check the system status."

    def generate_code(
        self, 
        requirements: str
    ) -> Tuple[str, str, str]:
        """
        Generate plugin code from requirements.
        
        Args:
            requirements: Plugin requirements
            
        Returns:
            Tuple of (header_code, implementation_code, status_message)
        """
        if not requirements or not requirements.strip():
            return "", "", "⚠️ Please enter plugin requirements."
        
        if not self.code_generator:
            return "", "", "❌ ERROR: Code generator not initialized."
        
        try:
            # Generate code
            result = self.code_generator.generate_plugin_with_rag(requirements)
            
            header = result.get("header", "")
            implementation = result.get("implementation", "")
            
            # Validate generated code
            header_valid, header_errors = self.code_validator.validate_syntax(header)
            impl_valid, impl_errors = self.code_validator.validate_syntax(implementation)
            
            # Store for export
            self.last_generated_code = {
                "header": header,
                "implementation": implementation,
                "requirements": requirements,
                "timestamp": datetime.now().isoformat()
            }
            
            # Build status message
            status = "✅ Code generated successfully!\n\n"
            
            if not header_valid:
                status += f"⚠️ Header validation warnings:\n"
                for error in header_errors:
                    status += f"  - {error}\n"
            
            if not impl_valid:
                status += f"⚠️ Implementation validation warnings:\n"
                for error in impl_errors:
                    status += f"  - {error}\n"
            
            if header_valid and impl_valid:
                status += "✅ All code validation checks passed!"
            
            return header, implementation, status
            
        except Exception as e:
            return "", "", f"❌ ERROR: {str(e)}"

    def export_code(self, plugin_name: str = "MyPlugin") -> str:
        """
        Export generated code to files.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Status message
        """
        if not self.last_generated_code:
            return "⚠️ No code to export. Generate code first."
        
        if not plugin_name or not plugin_name.strip():
            plugin_name = "MyPlugin"
        
        # Clean plugin name
        plugin_name = "".join(c for c in plugin_name if c.isalnum() or c in "_")
        
        try:
            # Create output directory
            output_dir = Path("generated_plugins") / plugin_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write header file
            header_path = output_dir / f"{plugin_name}.h"
            with open(header_path, "w", encoding="utf-8") as f:
                f.write(self.last_generated_code["header"])
            
            # Write implementation file
            impl_path = output_dir / f"{plugin_name}.cpp"
            with open(impl_path, "w", encoding="utf-8") as f:
                f.write(self.last_generated_code["implementation"])
            
            # Write README
            readme_path = output_dir / "README.md"
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(f"# {plugin_name}\n\n")
                f.write(f"**Generated:** {self.last_generated_code['timestamp']}\n\n")
                f.write(f"## Requirements\n\n{self.last_generated_code['requirements']}\n\n")
                f.write(f"## Files\n\n")
                f.write(f"- `{plugin_name}.h` - Header file\n")
                f.write(f"- `{plugin_name}.cpp` - Implementation file\n\n")
                f.write(f"## Build Instructions\n\n")
                f.write(f"1. Add these files to your BakkesMod plugin project\n")
                f.write(f"2. Update CMakeLists.txt to include these files\n")
                f.write(f"3. Build using Visual Studio or CMake\n")
            
            return (f"✅ Code exported successfully!\n\n"
                   f"📁 Location: {output_dir.absolute()}\n\n"
                   f"Files created:\n"
                   f"- {plugin_name}.h\n"
                   f"- {plugin_name}.cpp\n"
                   f"- README.md")
            
        except Exception as e:
            return f"❌ ERROR during export: {str(e)}"

    def get_statistics(self) -> str:
        """
        Get session statistics.
        
        Returns:
            Formatted statistics
        """
        avg_time = (self.total_query_time / self.query_count 
                   if self.query_count > 0 else 0)
        success_rate = ((self.successful_queries / self.query_count * 100) 
                       if self.query_count > 0 else 0)
        cache_hit_rate = ((self.cache_hits / self.query_count * 100) 
                         if self.query_count > 0 else 0)
        
        stats = f"""
📊 **SESSION STATISTICS**

**Queries:**
- Total queries: {self.query_count}
- Successful: {self.successful_queries}
- Success rate: {success_rate:.1f}%

**Performance:**
- Average query time: {avg_time:.2f}s
- Cache hits: {self.cache_hits}
- Cache hit rate: {cache_hit_rate:.1f}%

**Cache Savings:**
- Estimated cost savings: ${self.cache_hits * 0.03:.2f}

**System Info:**
- Documents loaded: {len(self.documents)}
- Searchable chunks: {len(self.nodes)}
- Retrieval: Vector + BM25 + Knowledge Graph
"""
        return stats

    def _calculate_confidence(self, source_nodes: list) -> Tuple[float, str, str]:
        """Calculate confidence score from retrieval quality."""
        if not source_nodes:
            return 0.0, "NO DATA", "No sources retrieved"
        
        scores = [
            node.score for node in source_nodes 
            if hasattr(node, 'score') and node.score is not None
        ]
        
        if not scores:
            return 0.5, "MEDIUM", "Sources retrieved but no similarity scores"
        
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        num_sources = len(source_nodes)
        
        # Calculate variance
        if len(scores) > 1:
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            std_dev = variance ** 0.5
        else:
            std_dev = 0.0
        
        # Calculate confidence
        score_component = avg_score * 50
        max_component = max_score * 20
        source_bonus = min(num_sources / 5.0, 1.0) * 10
        consistency_component = max(0, (1 - std_dev) * 20)
        
        confidence = (
            score_component + max_component + 
            source_bonus + consistency_component
        ) / 100.0
        confidence = max(0.0, min(1.0, confidence))
        
        # Assign label
        if confidence >= 0.85:
            return confidence, "VERY HIGH", "Excellent source match"
        elif confidence >= 0.70:
            return confidence, "HIGH", "Strong source match"
        elif confidence >= 0.50:
            return confidence, "MEDIUM", "Moderate source match"
        elif confidence >= 0.30:
            return confidence, "LOW", "Weak source match"
        else:
            return confidence, "VERY LOW", "Poor source match"


def create_gui():
    """Create and configure the Gradio interface."""
    
    # Initialize the RAG system
    app = BakkesModRAGGUI()
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .source-box {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .code-block {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Consolas', 'Monaco', monospace;
        overflow-x: auto;
    }
    """
    
    # Create Gradio interface
    with gr.Blocks(
        title="BakkesMod RAG System",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🚀 BakkesMod RAG System
        ### Professional Documentation Assistant & Plugin Code Generator
        
        Ask questions about BakkesMod SDK, generate plugin code, and explore the documentation.
        """)
        
        # Tabs for different features
        with gr.Tabs():
            
            # Tab 1: Query Documentation
            with gr.Tab("📚 Query Documentation"):
                gr.Markdown("""
                Ask questions about the BakkesMod SDK, plugin development, ImGui integration,
                event hooking, and more.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="How do I hook the goal scored event?",
                            lines=3
                        )
                        use_cache_checkbox = gr.Checkbox(
                            label="Use semantic cache (faster, cheaper)",
                            value=True
                        )
                        query_btn = gr.Button("🔍 Search", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        **Example Questions:**
                        - What is BakkesMod?
                        - How do I create a plugin?
                        - How do I access player velocity?
                        - How do I use ImGui?
                        - What events can I hook?
                        """)
                
                query_output = gr.Markdown(label="Response")
                
                query_btn.click(
                    fn=app.query_rag,
                    inputs=[query_input, use_cache_checkbox],
                    outputs=query_output
                )
            
            # Tab 2: Generate Plugin Code
            with gr.Tab("⚙️ Generate Plugin Code"):
                gr.Markdown("""
                Describe your plugin requirements and get complete, validated C++ code
                ready to compile.
                """)
                
                with gr.Row():
                    with gr.Column():
                        code_requirements = gr.Textbox(
                            label="Plugin Requirements",
                            placeholder="Create a plugin that hooks goal events and logs scorer info",
                            lines=5
                        )
                        generate_btn = gr.Button("🔨 Generate Code", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        header_output = gr.Code(
                            label="Header File (.h)",
                            language="cpp",
                            lines=15
                        )
                    
                    with gr.Column():
                        impl_output = gr.Code(
                            label="Implementation File (.cpp)",
                            language="cpp",
                            lines=15
                        )
                
                code_status = gr.Markdown(label="Status")
                
                with gr.Row():
                    plugin_name_input = gr.Textbox(
                        label="Plugin Name",
                        placeholder="MyPlugin",
                        value="MyPlugin"
                    )
                    export_btn = gr.Button("💾 Export to Files")
                
                export_status = gr.Markdown(label="Export Status")
                
                generate_btn.click(
                    fn=app.generate_code,
                    inputs=code_requirements,
                    outputs=[header_output, impl_output, code_status]
                )
                
                export_btn.click(
                    fn=app.export_code,
                    inputs=plugin_name_input,
                    outputs=export_status
                )
            
            # Tab 3: Statistics & Info
            with gr.Tab("📊 Statistics"):
                gr.Markdown("View session statistics and system information.")
                
                stats_output = gr.Markdown()
                stats_btn = gr.Button("🔄 Refresh Statistics")
                
                stats_btn.click(
                    fn=app.get_statistics,
                    outputs=stats_output
                )
                
                # Auto-load stats on open
                demo.load(
                    fn=app.get_statistics,
                    outputs=stats_output
                )
            
            # Tab 4: Help & Examples
            with gr.Tab("❓ Help"):
                gr.Markdown("""
                ## About BakkesMod RAG System
                
                This system uses Retrieval-Augmented Generation (RAG) to provide accurate
                answers about BakkesMod plugin development.
                
                ### Features:
                - **Hybrid Retrieval**: Combines Vector search, BM25 keyword search, and Knowledge Graph
                - **Semantic Caching**: Saves costs by caching similar queries (30-40% savings)
                - **Code Generation**: Creates complete, validated plugin code from descriptions
                - **Streaming Responses**: See answers as they're generated
                - **Confidence Scores**: Transparent quality indicators
                
                ### Architecture:
                - **Documents**: {num_docs} BakkesMod SDK documentation files
                - **Chunks**: {num_nodes} searchable text chunks
                - **Retrieval**: 3-way fusion (Vector + BM25 + Knowledge Graph)
                - **LLM**: Claude 3.5 Sonnet for responses
                - **Embeddings**: OpenAI text-embedding-3-small
                
                ### Cost Optimization:
                - Semantic cache reduces API calls by 30-40%
                - Typical query cost: $0.01-0.05
                - Cache hit: ~$0 (instant response)
                
                ### Example Workflows:
                
                **1. Learning BakkesMod:**
                - Ask "What is BakkesMod?" to get an overview
                - Ask "How do I create my first plugin?" for getting started
                
                **2. Developing a Plugin:**
                - Query specific API questions in the Documentation tab
                - Use Generate Code tab to create boilerplate
                - Export code and add to your Visual Studio project
                
                **3. Debugging:**
                - Ask about specific classes, methods, or events
                - Get code examples and best practices
                
                ### Tips:
                - Use specific questions for better results
                - Enable semantic cache to save costs on similar queries
                - Check confidence scores - higher is more reliable
                - Generated code is validated but may need minor adjustments
                """.format(
                    num_docs=len(app.documents),
                    num_nodes=len(app.nodes)
                ))
        
        # Footer
        gr.Markdown("""
        ---
        **BakkesMod RAG System** | Built with LlamaIndex, Anthropic Claude, and OpenAI
        """)
    
    return demo


def main():
    """Main entry point."""
    print("=" * 80)
    print("BakkesMod RAG - GUI Application")
    print("=" * 80)
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment!")
        print("Please create a .env file with your API keys.")
        sys.exit(1)
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not found in environment!")
        print("Please create a .env file with your API keys.")
        sys.exit(1)
    
    print("\nInitializing GUI...")
    
    # Create and launch GUI
    demo = create_gui()
    
    print("\n" + "=" * 80)
    print("GUI Ready!")
    print("=" * 80)
    print("\nLaunching web interface...")
    print("The GUI will open in your default browser.")
    print("Press Ctrl+C to stop the server.\n")
    
    # Launch with share=False for local use, share=True for public link
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
