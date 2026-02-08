"""
Code Generator
==============
Generates BakkesMod plugin code using RAG + LLM.
"""

import os
from typing import Dict, Optional
from llama_index.core import Settings, VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding
from code_templates import PluginTemplateEngine
from code_validator import CodeValidator


class CodeGenerator:
    """Generates BakkesMod plugin code with RAG context."""

    def __init__(self, rag_storage_dir: str = "rag_storage_bakkesmod"):
        """
        Initialize code generator.

        Args:
            rag_storage_dir: Path to RAG index storage
        """
        # Initialize LLM for code generation
        Settings.llm = Anthropic(model="claude-sonnet-4-5", temperature=0, max_retries=3)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        # Load RAG index
        self.index = self._load_index(rag_storage_dir)
        self.query_engine = self.index.as_query_engine(similarity_top_k=5) if self.index else None

        # Initialize helpers
        self.template_engine = PluginTemplateEngine()
        self.validator = CodeValidator()

    def _load_index(self, storage_dir: str) -> Optional[VectorStoreIndex]:
        """Load RAG index from storage."""
        try:
            from pathlib import Path
            if not Path(storage_dir).exists():
                print(f"[WARNING] RAG index not found at {storage_dir}")
                return None

            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            return load_index_from_storage(storage_context, index_id="vector")
        except Exception as e:
            print(f"[WARNING] Could not load RAG index: {e}")
            return None

    def generate_plugin(self, requirements: str) -> Dict[str, str]:
        """
        Generate plugin code from requirements (without RAG).

        Args:
            requirements: Natural language description of plugin

        Returns:
            Dict with 'header' and 'implementation' keys
        """
        prompt = f"""Generate a complete BakkesMod plugin based on these requirements:

{requirements}

Generate:
1. Header file (.h) with class declaration
2. Implementation file (.cpp) with full implementation

Use proper BakkesMod plugin structure:
- Inherit from BakkesModPlugin
- Implement onLoad() and onUnload()
- Use gameWrapper for game access
- Use HookEvent for event handling
- Use LOG() for logging

Return ONLY the code, no explanations.

HEADER FILE:
```cpp
[header code here]
```

IMPLEMENTATION FILE:
```cpp
[implementation code here]
```
"""

        response = Settings.llm.complete(prompt)
        return self._parse_code_response(response.text)

    def generate_plugin_with_rag(self, requirements: str) -> Dict[str, str]:
        """
        Generate plugin code using RAG context.

        Args:
            requirements: Natural language description

        Returns:
            Dict with 'header' and 'implementation' keys
        """
        if not self.query_engine:
            print("[WARNING] No RAG index, falling back to template-based generation")
            return self.generate_plugin(requirements)

        # Query RAG for relevant documentation
        rag_query = f"How to implement: {requirements}"
        rag_response = self.query_engine.query(rag_query)
        sdk_context = str(rag_response)

        # Generate code with SDK context
        prompt = f"""You are a BakkesMod plugin code generator.

USER REQUIREMENTS:
{requirements}

RELEVANT SDK DOCUMENTATION:
{sdk_context}

Generate a complete BakkesMod plugin that implements these requirements using the SDK documentation as reference.

Plugin structure:
- Header file (.h): Class declaration inheriting from BakkesModPlugin
- Implementation file (.cpp): Full implementation with proper BakkesMod API usage

CRITICAL: Use the EXACT event names and API calls from the documentation above.

Return format:

HEADER FILE:
```cpp
[complete header code]
```

IMPLEMENTATION FILE:
```cpp
[complete implementation code]
```
"""

        response = Settings.llm.complete(prompt)
        code = self._parse_code_response(response.text)

        # Validate generated code
        validation = self.validator.validate_syntax(code.get("implementation", ""))
        if not validation["valid"]:
            print(f"[WARNING] Generated code has syntax issues: {validation['errors']}")

        return code

    def generate_complete_project(self, requirements: str) -> Dict[str, str]:
        """
        Generate a complete plugin project with all files.

        Args:
            requirements: Detailed plugin requirements

        Returns:
            Dict with 'header', 'implementation', 'cmake', 'readme' keys
        """
        # Generate main plugin code
        plugin_code = self.generate_plugin_with_rag(requirements)

        # Generate CMakeLists.txt
        cmake_content = self._generate_cmake_file()

        # Generate README
        readme_content = self._generate_readme(requirements)

        return {
            **plugin_code,
            "cmake": cmake_content,
            "readme": readme_content
        }

    def generate_imgui_window(self, requirements: str) -> str:
        """
        Generate ImGui window code.

        Args:
            requirements: UI requirements

        Returns:
            C++ code for ImGui window
        """
        # Query RAG for ImGui documentation
        if self.query_engine:
            rag_query = f"ImGui window implementation: {requirements}"
            rag_response = self.query_engine.query(rag_query)
            imgui_context = str(rag_response)
        else:
            imgui_context = "Basic ImGui window structure"

        prompt = f"""Generate ImGui window code for these requirements:

{requirements}

SDK CONTEXT:
{imgui_context}

Generate complete C++ function that:
- Uses ImGui::Begin() and ImGui::End()
- Implements the requested UI elements
- Follows BakkesMod ImGui patterns

Return ONLY the C++ function code.
"""

        response = Settings.llm.complete(prompt)
        return response.text.strip()

    def _generate_cmake_file(self) -> str:
        """Generate CMakeLists.txt for the plugin."""
        return """cmake_minimum_required(VERSION 3.15)
project(MyPlugin)

set(CMAKE_CXX_STANDARD 17)

# BakkesMod SDK
find_package(BakkesModSDK REQUIRED)

# Plugin source files
add_library(MyPlugin SHARED
    MyPlugin.h
    MyPlugin.cpp
)

target_link_libraries(MyPlugin
    BakkesModSDK::BakkesModSDK
)

# Install
install(TARGETS MyPlugin
    DESTINATION plugins
)
"""

    def _generate_readme(self, requirements: str) -> str:
        """Generate README.md for the plugin."""
        return f"""# MyPlugin

## Description

{requirements}

## Installation

1. Copy `MyPlugin.dll` to `bakkesmod/plugins/`
2. Enable in BakkesMod plugin manager

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Usage

[Plugin usage instructions here]
"""

    def _parse_code_response(self, response_text: str) -> Dict[str, str]:
        """Parse LLM response to extract header and implementation code."""
        import re

        # Extract code blocks
        header_match = re.search(r'HEADER FILE:.*?```cpp\n(.*?)```', response_text, re.DOTALL)
        impl_match = re.search(r'IMPLEMENTATION FILE:.*?```cpp\n(.*?)```', response_text, re.DOTALL)

        return {
            "header": header_match.group(1).strip() if header_match else "",
            "implementation": impl_match.group(1).strip() if impl_match else ""
        }
