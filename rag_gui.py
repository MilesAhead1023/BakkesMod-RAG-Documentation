"""BakkesMod RAG -- Gradio GUI entry point.

Thin wrapper around ``bakkesmod_rag.RAGEngine``.  Builds a four-tab
Gradio interface (Query, Generate Code, Statistics, Help) while
delegating all RAG logic to the unified engine.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Generator
from dotenv import load_dotenv

load_dotenv()

try:
    import gradio as gr
except ImportError:
    print("ERROR: Gradio not installed.  Run:  pip install gradio")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
engine = None
query_count = 0
successful_queries = 0
total_query_time = 0.0
cache_hits = 0
last_generated_code = None


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialize():
    """Build the RAG engine.  Called once at startup.

    Returns:
        Status message displayed in the GUI.
    """
    global engine
    from bakkesmod_rag import RAGEngine
    from bakkesmod_rag.setup_keys import ensure_api_keys

    ensure_api_keys()

    try:
        engine = RAGEngine()
        return (
            f"System initialized successfully!\n"
            f"Loaded {engine.num_documents} documents\n"
            f"Processed {engine.num_nodes} searchable chunks\n"
            f"Using 3-way fusion: Vector + BM25 + Knowledge Graph"
        )
    except Exception as e:
        return f"ERROR during initialization: {e}"


# ---------------------------------------------------------------------------
# Query tab
# ---------------------------------------------------------------------------

def query_rag(query: str, use_cache: bool = True) -> Generator[str, None, None]:
    """Stream an answer from the RAG engine.

    Args:
        query: User question.
        use_cache: Whether to use the semantic cache.

    Yields:
        Chunks of the response (including trailing metadata).
    """
    global query_count, successful_queries, total_query_time, cache_hits

    if not query or not query.strip():
        yield "Please enter a question."
        return

    if engine is None:
        yield "ERROR: System not initialized. Please restart the application."
        return

    query_count += 1

    try:
        gen, get_meta = engine.query_streaming(query, use_cache=use_cache)

        # Stream the answer tokens
        full_text = ""
        yield "**ANSWER:**\n\n"
        for token in gen:
            full_text += token
            yield token

        # Fetch metadata after the generator is exhausted
        meta = get_meta()
        successful_queries += 1
        total_query_time += meta.query_time

        if meta.cached:
            cache_hits += 1
            yield (
                f"\n\n---\n"
                f"**CACHED RESPONSE** (similarity: {meta.confidence:.1%})\n"
                f"Query time: {meta.query_time:.2f}s\n"
                f"Cost savings: ~$0.02-0.04"
            )
        else:
            yield f"\n\n---\n"
            yield f"Query time: {meta.query_time:.2f}s\n"
            yield (
                f"Confidence: {meta.confidence:.0%} "
                f"({meta.confidence_label}) - "
                f"{meta.confidence_explanation}\n"
            )
            yield f"Sources: {len(meta.sources)}\n"

            if meta.sources:
                yield "\n**Source Files:**\n"
                seen = set()
                for src in meta.sources:
                    name = src.get("file_name", "unknown") if isinstance(src, dict) else src
                    if name not in seen:
                        seen.add(name)
                        yield f"- {name}\n"

    except Exception as e:
        total_query_time += 0.0
        yield f"\n\n**ERROR:** {e}\n"
        yield "Please try rephrasing your question or check the system status."


# ---------------------------------------------------------------------------
# Code generation tab
# ---------------------------------------------------------------------------

def generate_code(requirements: str):
    """Generate plugin code from natural-language requirements.

    Args:
        requirements: Plugin description.

    Returns:
        Tuple of (header_code, implementation_code, status_message).
    """
    global last_generated_code

    if not requirements or not requirements.strip():
        return "", "", "Please enter plugin requirements."

    if engine is None:
        return "", "", "ERROR: System not initialized."

    try:
        result = engine.generate_code(requirements)

        last_generated_code = {
            "header": result.header,
            "implementation": result.implementation,
            "project_files": result.project_files,
            "requirements": requirements,
            "timestamp": datetime.now().isoformat(),
        }

        status = "**Code generated successfully!**\n\n"

        # Show features detected
        if result.features_used:
            status += f"**Features detected:** {', '.join(result.features_used)}\n\n"

        # Show project file list
        if result.project_files:
            status += f"**Project files:** {len(result.project_files)} files generated\n"
            for fname in sorted(result.project_files.keys()):
                size = len(result.project_files[fname])
                status += f"- `{fname}` ({size} bytes)\n"
            status += "\n"

        validation = result.validation
        if validation and not validation.get("valid", True):
            status += "**Validation warnings:**\n"
            for err in validation.get("errors", []):
                status += f"- {err}\n"
            for warn in validation.get("warnings", []):
                status += f"- (warn) {warn}\n"
        else:
            status += "**All code validation checks passed!**"

        return result.header, result.implementation, status

    except Exception as e:
        return "", "", f"ERROR: {e}"


def export_code(plugin_name: str = "MyPlugin") -> str:
    """Export the last generated code to files on disk.

    Exports all project files (header, implementation, pch, version,
    logging, GuiBase, resource, .rc, .props, .vcxproj) when available.

    Args:
        plugin_name: Name used for the output directory.

    Returns:
        Status message.
    """
    if not last_generated_code:
        return "No code to export. Generate code first."

    if not plugin_name or not plugin_name.strip():
        plugin_name = "MyPlugin"

    # Sanitize plugin name
    plugin_name = "".join(c for c in plugin_name if c.isalnum() or c in "_")

    try:
        output_dir = Path("generated_plugins") / plugin_name
        output_dir.mkdir(parents=True, exist_ok=True)

        project_files = last_generated_code.get("project_files", {})
        files_written = []

        if project_files:
            # Export all project files
            for fname, content in project_files.items():
                file_path = output_dir / fname
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                files_written.append(fname)
        else:
            # Fallback: just header + implementation
            header_path = output_dir / f"{plugin_name}.h"
            with open(header_path, "w", encoding="utf-8") as f:
                f.write(last_generated_code["header"])
            files_written.append(f"{plugin_name}.h")

            impl_path = output_dir / f"{plugin_name}.cpp"
            with open(impl_path, "w", encoding="utf-8") as f:
                f.write(last_generated_code["implementation"])
            files_written.append(f"{plugin_name}.cpp")

        file_list = "\n".join(f"- {f}" for f in sorted(files_written))
        return (
            f"Code exported successfully!\n\n"
            f"Location: {output_dir.absolute()}\n\n"
            f"Files created ({len(files_written)}):\n{file_list}"
        )

    except Exception as e:
        return f"ERROR during export: {e}"


# ---------------------------------------------------------------------------
# Statistics tab
# ---------------------------------------------------------------------------

def get_statistics() -> str:
    """Return formatted session statistics as a Markdown string."""
    avg_time = (total_query_time / query_count) if query_count > 0 else 0
    success_rate = (successful_queries / query_count * 100) if query_count > 0 else 0
    cache_hit_rate = (cache_hits / query_count * 100) if query_count > 0 else 0

    num_docs = engine.num_documents if engine else 0
    num_nodes = engine.num_nodes if engine else 0

    return (
        f"**SESSION STATISTICS**\n\n"
        f"**Queries:**\n"
        f"- Total queries: {query_count}\n"
        f"- Successful: {successful_queries}\n"
        f"- Success rate: {success_rate:.1f}%\n\n"
        f"**Performance:**\n"
        f"- Average query time: {avg_time:.2f}s\n"
        f"- Cache hits: {cache_hits}\n"
        f"- Cache hit rate: {cache_hit_rate:.1f}%\n\n"
        f"**Cache Savings:**\n"
        f"- Estimated cost savings: ${cache_hits * 0.03:.2f}\n\n"
        f"**System Info:**\n"
        f"- Documents loaded: {num_docs}\n"
        f"- Searchable chunks: {num_nodes}\n"
        f"- Retrieval: Vector + BM25 + Knowledge Graph"
    )


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

def create_gui():
    """Create and return the Gradio Blocks application."""

    # Initialize the engine at import time so the GUI is ready immediately
    init_status = initialize()
    print(init_status)

    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    """

    num_docs = engine.num_documents if engine else "?"
    num_nodes = engine.num_nodes if engine else "?"

    with gr.Blocks(
        title="BakkesMod RAG System",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as demo:

        gr.Markdown(
            "# BakkesMod RAG System\n"
            "### Documentation Assistant & Plugin Code Generator\n\n"
            "Ask questions about BakkesMod SDK, generate plugin code, "
            "and explore the documentation."
        )

        with gr.Tabs():

            # ---- Tab 1: Query Documentation ----
            with gr.Tab("Query Documentation"):
                gr.Markdown(
                    "Ask questions about the BakkesMod SDK, plugin "
                    "development, ImGui integration, event hooking, "
                    "and more."
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="How do I hook the goal scored event?",
                            lines=3,
                        )
                        use_cache_checkbox = gr.Checkbox(
                            label="Use semantic cache (faster, cheaper)",
                            value=True,
                        )
                        query_btn = gr.Button("Search", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**Example Questions:**\n"
                            "- What is BakkesMod?\n"
                            "- How do I create a plugin?\n"
                            "- How do I access player velocity?\n"
                            "- How do I use ImGui?\n"
                            "- What events can I hook?"
                        )

                query_output = gr.Markdown(label="Response")

                query_btn.click(
                    fn=query_rag,
                    inputs=[query_input, use_cache_checkbox],
                    outputs=query_output,
                )

            # ---- Tab 2: Generate Plugin Code ----
            with gr.Tab("Generate Plugin Code"):
                gr.Markdown(
                    "Describe your plugin requirements and get complete, "
                    "validated C++ code ready to compile."
                )

                with gr.Row():
                    with gr.Column():
                        code_requirements = gr.Textbox(
                            label="Plugin Requirements",
                            placeholder=(
                                "Create a plugin that hooks goal events "
                                "and logs scorer info"
                            ),
                            lines=5,
                        )
                        generate_btn = gr.Button(
                            "Generate Code", variant="primary"
                        )

                with gr.Row():
                    with gr.Column():
                        header_output = gr.Code(
                            label="Header File (.h)",
                            language="cpp",
                            lines=15,
                        )
                    with gr.Column():
                        impl_output = gr.Code(
                            label="Implementation File (.cpp)",
                            language="cpp",
                            lines=15,
                        )

                code_status = gr.Markdown(label="Status")

                with gr.Row():
                    plugin_name_input = gr.Textbox(
                        label="Plugin Name",
                        placeholder="MyPlugin",
                        value="MyPlugin",
                    )
                    export_btn = gr.Button("Export to Files")

                export_status = gr.Markdown(label="Export Status")

                generate_btn.click(
                    fn=generate_code,
                    inputs=code_requirements,
                    outputs=[header_output, impl_output, code_status],
                )

                export_btn.click(
                    fn=export_code,
                    inputs=plugin_name_input,
                    outputs=export_status,
                )

            # ---- Tab 3: Statistics ----
            with gr.Tab("Statistics"):
                gr.Markdown("View session statistics and system information.")

                stats_output = gr.Markdown()
                stats_btn = gr.Button("Refresh Statistics")

                stats_btn.click(
                    fn=get_statistics,
                    outputs=stats_output,
                )

                demo.load(
                    fn=get_statistics,
                    outputs=stats_output,
                )

            # ---- Tab 4: Help ----
            with gr.Tab("Help"):
                gr.Markdown(
                    "## About BakkesMod RAG System\n\n"
                    "This system uses Retrieval-Augmented Generation (RAG) "
                    "to provide accurate answers about BakkesMod plugin "
                    "development.\n\n"
                    "### Features\n"
                    "- **Hybrid Retrieval**: Vector search + BM25 keyword "
                    "search + Knowledge Graph\n"
                    "- **Semantic Caching**: Saves costs by caching similar "
                    "queries (30-40% savings)\n"
                    "- **Code Generation**: Creates complete, validated "
                    "plugin code from descriptions\n"
                    "- **Streaming Responses**: See answers as they are "
                    "generated\n"
                    "- **Confidence Scores**: Transparent quality "
                    "indicators\n\n"
                    "### Architecture\n"
                    f"- **Documents**: {num_docs} BakkesMod SDK "
                    "documentation files\n"
                    f"- **Chunks**: {num_nodes} searchable text chunks\n"
                    "- **Retrieval**: 3-way fusion "
                    "(Vector + BM25 + Knowledge Graph)\n"
                    "- **LLM**: Automatic fallback chain "
                    "(Anthropic / OpenRouter / Gemini / OpenAI)\n"
                    "- **Embeddings**: OpenAI text-embedding-3-small\n\n"
                    "### Cost Optimisation\n"
                    "- Semantic cache reduces API calls by 30-40%\n"
                    "- Typical query cost: $0.01-0.05\n"
                    "- Cache hit: ~$0 (instant response)\n\n"
                    "### Example Workflows\n\n"
                    "**1. Learning BakkesMod:**\n"
                    '- Ask "What is BakkesMod?" to get an overview\n'
                    '- Ask "How do I create my first plugin?" to get '
                    "started\n\n"
                    "**2. Developing a Plugin:**\n"
                    "- Query specific API questions in the Documentation "
                    "tab\n"
                    "- Use Generate Code tab to create boilerplate\n"
                    "- Export code and add to your Visual Studio project\n\n"
                    "**3. Debugging:**\n"
                    "- Ask about specific classes, methods, or events\n"
                    "- Get code examples and best practices\n\n"
                    "### Tips\n"
                    "- Use specific questions for better results\n"
                    "- Enable semantic cache to save costs on similar "
                    "queries\n"
                    "- Check confidence scores - higher is more reliable\n"
                    "- Generated code is validated but may need minor "
                    "adjustments"
                )

        gr.Markdown(
            "---\n"
            "**BakkesMod RAG System** | Built with LlamaIndex, "
            "Anthropic Claude, and OpenAI"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Launch the Gradio GUI."""
    print("=" * 80)
    print("BakkesMod RAG - GUI Application")
    print("=" * 80)

    demo = create_gui()

    print("\n" + "=" * 80)
    print("GUI Ready!")
    print("=" * 80)
    print("\nLaunching web interface...")
    print("The GUI will open in your default browser.")
    print("Press Ctrl+C to stop the server.\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
