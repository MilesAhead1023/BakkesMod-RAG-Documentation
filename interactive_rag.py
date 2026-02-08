"""Interactive RAG Query Interface -- CLI entry point.

Thin wrapper around ``bakkesmod_rag.RAGEngine``.  Handles user I/O,
syntax highlighting, and session statistics while delegating all RAG
logic to the unified engine.
"""

import os
import sys
import time
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Initialize colorama for Windows terminal colors
try:
    import colorama
    colorama.init()
except ImportError:
    pass


def log(message, level="INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level:5s}] {message}")


def highlight_code_blocks(text):
    """Apply syntax highlighting to C++ code blocks in *text*.

    Uses Pygments if available; returns *text* unchanged otherwise.
    """
    try:
        from pygments import highlight
        from pygments.lexers import CppLexer, get_lexer_by_name
        from pygments.formatters import TerminalFormatter

        code_block_pattern = r'```(\w+)?\n(.*?)```'

        def _highlight_match(match):
            language = match.group(1) or 'cpp'
            code = match.group(2)
            try:
                if language.lower() in ('cpp', 'c++', 'c'):
                    lexer = CppLexer()
                else:
                    lexer = get_lexer_by_name(language, stripall=True)
                return '\n' + highlight(code, lexer, TerminalFormatter())
            except Exception:
                return match.group(0)

        return re.sub(code_block_pattern, _highlight_match, text, flags=re.DOTALL)

    except ImportError:
        return text


def display_help():
    """Display help information."""
    print("\n" + "=" * 80)
    print("  HELP - BakkesMod RAG System")
    print("=" * 80)
    print("\nThis system can answer questions about:")
    print("  - BakkesMod SDK documentation and API reference")
    print("  - Plugin development and architecture")
    print("  - ImGui UI integration")
    print("  - Event hooking and game events")
    print("  - Car physics and player data access")
    print("\nExample questions:")
    print("  - What is BakkesMod?")
    print("  - How do I create a plugin?")
    print("  - How do I hook the goal scored event?")
    print("  - What are the main classes in the SDK?")
    print("  - How do I access player car velocity?")
    print("  - How do I use ImGui to create a settings window?")
    print("\nCommands:")
    print("  help   - Show this help message")
    print("  stats  - Show session statistics")
    print("  /generate <requirements> - Generate plugin code using RAG + LLM")
    print("  /code <requirements>     - Alias for /generate")
    print("  quit   - Exit the program (or Ctrl+C)")
    print("=" * 80)


def print_session_summary(query_count, successful, total_time):
    """Print a summary of session statistics before exiting."""
    if query_count > 0:
        print(f"\nSession summary:")
        print(f"  Total queries: {query_count}")
        print(f"  Successful: {successful}")
        print(f"  Average time: {total_time / query_count:.2f}s")
    print("\nThank you for using BakkesMod RAG!\n")


def main():
    """Main interactive loop."""
    print("=" * 80)
    print("  BAKKESMOD RAG - INTERACTIVE MODE")
    print("=" * 80)

    from bakkesmod_rag import RAGEngine

    log("Building RAG system...")
    try:
        engine = RAGEngine()
    except Exception as e:
        log(f"Failed to build system: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\nWelcome! {engine.num_documents} docs, {engine.num_nodes} nodes.")
    print("Type 'help' for examples, 'quit' to exit.\n")

    query_count = 0
    total_time = 0.0
    successful = 0

    while True:
        try:
            query = input("[QUERY] > ").strip()
            if not query:
                continue

            # --- quit ---
            if query.lower() in ('quit', 'exit', 'q'):
                print("\nExiting...")
                print_session_summary(query_count, successful, total_time)
                break

            # --- help ---
            if query.lower() == 'help':
                display_help()
                continue

            # --- code generation ---
            if query.lower().startswith(('/generate ', '/code ')):
                requirements = query.split(' ', 1)[1] if ' ' in query else ""
                if not requirements:
                    print("[ERROR] Usage: /generate <plugin requirements>")
                    continue

                log(f"Generating code for: {requirements[:60]}...")
                try:
                    result = engine.generate_code(requirements)

                    print("\n" + "=" * 80)
                    print("[GENERATED PLUGIN PROJECT]")
                    print("=" * 80)

                    # Show detected features
                    if result.features_used:
                        print(f"\n[FEATURES DETECTED] {', '.join(result.features_used)}")

                    # Show all project files
                    if result.project_files:
                        print(f"\n[PROJECT FILES] {len(result.project_files)} files generated:")
                        for fname in sorted(result.project_files.keys()):
                            size = len(result.project_files[fname])
                            print(f"  - {fname} ({size} bytes)")

                        # Show main plugin files in detail
                        for fname in sorted(result.project_files.keys()):
                            if fname.endswith((".h", ".cpp")) and fname not in (
                                "pch.h", "pch.cpp", "logging.h", "version.h",
                                "resource.h", "GuiBase.h", "GuiBase.cpp",
                            ):
                                print(f"\n--- {fname} ---")
                                print(result.project_files[fname])
                    else:
                        # Fallback: show header/implementation directly
                        print("\n--- HEADER FILE (.h) ---")
                        print(result.header)
                        print("\n--- IMPLEMENTATION FILE (.cpp) ---")
                        print(result.implementation)

                    # Validation results
                    print("\n" + "=" * 80)
                    if result.validation and not result.validation.get("valid", True):
                        print("[VALIDATION WARNINGS]")
                        for err in result.validation.get("errors", []):
                            print(f"  - {err}")
                        for warn in result.validation.get("warnings", []):
                            print(f"  - (warn) {warn}")
                    else:
                        print("[VALIDATION] All checks passed")
                    print("=" * 80)

                except Exception as e:
                    log(f"Code generation failed: {e}", "ERROR")

                continue

            # --- stats ---
            if query.lower() == 'stats':
                print(f"\n[SESSION STATISTICS]")
                print(f"  Total queries: {query_count}")
                print(f"  Successful: {successful}")
                if query_count > 0:
                    print(f"  Success rate: {(successful / query_count * 100):.1f}%")
                    print(f"  Average time: {total_time / query_count:.2f}s")
                else:
                    print(f"  No queries yet!")
                continue

            # --- streaming query ---
            log(f"Processing: {query[:60]}{'...' if len(query) > 60 else ''}")

            try:
                gen, get_meta = engine.query_streaming(query)

                print("\n" + "-" * 80)
                print("[ANSWER]")
                print("-" * 80)

                full_tokens = []
                for token in gen:
                    print(token, end="", flush=True)
                    full_tokens.append(token)

                print()  # newline after streaming
                print("-" * 80)

                # Apply syntax highlighting if code blocks present
                full_text = "".join(full_tokens)
                if "```" in full_text:
                    highlighted = highlight_code_blocks(full_text)
                    # Reprint with highlighting
                    line_count = full_text.count('\n') + 3
                    print(f"\033[F" * line_count)  # move cursor up
                    print("\n" + "-" * 80)
                    print("[ANSWER]")
                    print("-" * 80)
                    print(highlighted)
                    print("-" * 80)

                meta = get_meta()
                query_count += 1
                total_time += meta.query_time
                successful += 1

                # Display metadata
                if meta.cached:
                    print(f"\n[METADATA] Cache hit! "
                          f"(similarity: {meta.confidence:.1%}) "
                          f"Time: {meta.query_time:.2f}s")
                else:
                    confidence = meta.confidence
                    print(f"\n[METADATA]")
                    print(f"  Query time: {meta.query_time:.2f}s")
                    print(f"  Sources: {len(meta.sources)}")
                    print(f"  Confidence: {confidence:.0%} "
                          f"({meta.confidence_label}) - "
                          f"{meta.confidence_explanation}")
                    print(f"  Cached for future queries")

                    if meta.sources:
                        print(f"\n[SOURCE FILES]")
                        for src in meta.sources:
                            name = src.get("file_name", "unknown") if isinstance(src, dict) else src
                            print(f"  - {name}")

            except Exception as e:
                query_count += 1
                log(f"Query failed: {e}", "ERROR")
                print(f"\n[ERROR] {e}")
                print("Please try rephrasing your question or check the logs.")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            print_session_summary(query_count, successful, total_time)
            break
        except Exception as e:
            log(f"Unexpected error: {e}", "ERROR")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
