"""
Code Generator
==============
Merged module for BakkesMod plugin code generation, templating, and validation.

Contains three classes:
- BakkesModCodeGenerator: LLM-powered plugin generation (with optional RAG context)
- PluginTemplateEngine: Deterministic code scaffolding from templates
- CodeValidator: C++ syntax and BakkesMod API pattern validation

Consolidates the former root-level code_generator.py, code_templates.py,
and code_validator.py into the unified bakkesmod_rag package.
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger("bakkesmod_rag.code_generator")


# ---------------------------------------------------------------------------
# CodeValidator
# ---------------------------------------------------------------------------

class CodeValidator:
    """Validates C++ code for BakkesMod plugins.

    Performs lightweight static checks on generated C++ code:
    bracket matching, string literal closure, and BakkesMod API pattern
    detection. This is not a full compiler -- it catches the most common
    generation mistakes before the user tries to build.
    """

    def __init__(self):
        """Initialize validator with bracket pairs and API patterns."""
        self.bracket_pairs = {"{": "}", "(": ")", "[": "]"}

        self.api_patterns = {
            "gamewrapper": r"gameWrapper->",
            "hook_event": r"HookEvent\(",
            "server_wrapper": r"ServerWrapper",
            "car_wrapper": r"CarWrapper",
        }

    def validate_syntax(self, code: str) -> Dict:
        """Validate C++ syntax (brackets and string literals).

        Args:
            code: C++ source code to validate.

        Returns:
            Dict with ``valid`` (bool) and ``errors`` (list of strings).
        """
        errors: List[str] = []
        errors.extend(self._check_brackets(code))
        errors.extend(self._check_strings(code))

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }

    def validate_bakkesmod_api(self, code: str) -> Dict:
        """Check which BakkesMod API patterns appear in the code.

        Args:
            code: C++ source code to inspect.

        Returns:
            Dict with boolean flags for each API pattern found.
        """
        return {
            "uses_gamewrapper": bool(re.search(self.api_patterns["gamewrapper"], code)),
            "hooks_events": bool(re.search(self.api_patterns["hook_event"], code)),
            "uses_server": bool(re.search(self.api_patterns["server_wrapper"], code)),
            "uses_car": bool(re.search(self.api_patterns["car_wrapper"], code)),
        }

    def _check_brackets(self, code: str) -> List[str]:
        """Check for unmatched or mismatched brackets.

        Args:
            code: C++ source code.

        Returns:
            List of error descriptions (empty if all brackets match).
        """
        errors: List[str] = []
        stack: List[tuple] = []

        for i, char in enumerate(code):
            if char in self.bracket_pairs:
                stack.append((char, i))
            elif char in self.bracket_pairs.values():
                if not stack:
                    errors.append(
                        f"Unmatched closing bracket '{char}' at position {i}"
                    )
                else:
                    open_char, _ = stack.pop()
                    if self.bracket_pairs[open_char] != char:
                        errors.append(
                            f"Mismatched bracket: expected "
                            f"'{self.bracket_pairs[open_char]}' but got '{char}'"
                        )

        for open_char, pos in stack:
            errors.append(f"Unclosed bracket '{open_char}' at position {pos}")

        return errors

    def _check_strings(self, code: str) -> List[str]:
        """Check for unclosed string literals.

        Args:
            code: C++ source code.

        Returns:
            List of error descriptions (empty if strings are balanced).
        """
        errors: List[str] = []
        in_string = False
        escape_next = False

        for char in code:
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string

        if in_string:
            errors.append("Unclosed string literal")

        return errors


# ---------------------------------------------------------------------------
# PluginTemplateEngine
# ---------------------------------------------------------------------------

class PluginTemplateEngine:
    """Generates deterministic code templates for BakkesMod plugins.

    These templates require no LLM calls -- they produce syntactically
    correct scaffolding that a developer (or an LLM) can fill in.
    """

    def generate_basic_plugin(
        self, plugin_name: str, description: str
    ) -> Dict[str, str]:
        """Generate a minimal plugin with header and implementation.

        Args:
            plugin_name: C++ class name for the plugin.
            description: One-line description used in the header comment.

        Returns:
            Dict with ``header`` and ``implementation`` keys containing
            ready-to-compile C++ source strings.
        """
        header = f"""#pragma once
#include "bakkesmod/plugin/bakkesmodplugin.h"

/**
 * {description}
 */
class {plugin_name} : public BakkesMod::Plugin::BakkesModPlugin
{{
public:
    // Plugin lifecycle
    virtual void onLoad() override;
    virtual void onUnload() override;

private:
    // Plugin implementation
}};
"""

        implementation = f"""#include "{plugin_name}.h"

BAKKESMOD_PLUGIN({plugin_name}, "{plugin_name}", "1.0", PLUGINTYPE_FREEPLAY)

void {plugin_name}::onLoad()
{{
    // Plugin initialization
    LOG("{{}} loaded!", GetNameSafe());
}}

void {plugin_name}::onUnload()
{{
    // Plugin cleanup
    LOG("{{}} unloaded!", GetNameSafe());
}}
"""

        return {
            "header": header,
            "implementation": implementation,
        }

    def generate_event_hook(
        self, event_name: str, callback_name: str
    ) -> str:
        """Generate an event hook snippet.

        Args:
            event_name: Full event path
                (e.g. ``"Function TAGame.Ball_TA.OnHitGoal"``).
            callback_name: Name for the callback method.

        Returns:
            C++ code snippet that hooks the event.
        """
        code = f"""    // Hook {event_name}
    gameWrapper->HookEvent("{event_name}",
        [this](std::string eventName) {{
            {callback_name}(eventName);
        }});
"""
        return code

    def generate_imgui_window(
        self,
        window_title: str,
        elements: Optional[List[str]] = None,
    ) -> str:
        """Generate an ImGui window function.

        Args:
            window_title: Title shown in the ImGui title bar.
            elements: Optional list of UI element types to include.
                Supported values: ``"checkbox"``, ``"slider"``.

        Returns:
            Complete C++ function definition for the ImGui window.
        """
        code = f"""void Render{window_title}Window()
{{
    if (!ImGui::Begin("{window_title}"))
    {{
        ImGui::End();
        return;
    }}

    // UI elements
"""

        if elements:
            for element in elements:
                if element == "checkbox":
                    code += """    bool enabled = false;
    ImGui::Checkbox("Enabled", &enabled);

"""
                elif element == "slider":
                    code += """    float value = 0.0f;
    ImGui::SliderFloat("Value", &value, 0.0f, 100.0f);

"""

        code += """    ImGui::End();
}
"""
        return code


# ---------------------------------------------------------------------------
# BakkesModCodeGenerator
# ---------------------------------------------------------------------------

class BakkesModCodeGenerator:
    """LLM-powered BakkesMod plugin code generator.

    Unlike the old root-level ``CodeGenerator``, this class does **not**
    build its own LLM fallback chain or load indexes.  Instead it receives
    a pre-verified LLM and an optional query engine from the caller (typically
    the ``RAGEngine``).

    Args:
        llm: A verified LlamaIndex LLM instance (from ``llm_provider.get_llm``).
        query_engine: Optional LlamaIndex query engine for RAG-augmented
            generation.  When ``None``, ``generate_plugin_with_rag`` falls
            back to direct LLM generation.
    """

    def __init__(self, llm, query_engine=None):
        """Initialize the code generator.

        Args:
            llm: A verified LlamaIndex LLM instance.
            query_engine: Optional query engine for RAG context lookups.
        """
        self.llm = llm
        self.query_engine = query_engine
        self.template_engine = PluginTemplateEngine()
        self.validator = CodeValidator()

    # ------------------------------------------------------------------
    # Public generation methods
    # ------------------------------------------------------------------

    def generate_plugin(self, requirements: str) -> Dict[str, str]:
        """Generate plugin code from requirements using direct LLM prompting.

        No RAG context is used. Good for simple plugins or when no index
        is available.

        Args:
            requirements: Natural language description of the desired plugin.

        Returns:
            Dict with ``header`` and ``implementation`` C++ source strings.
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

        try:
            response = self.llm.complete(prompt)
            return self._parse_code_response(response.text)
        except Exception as e:
            logger.error("Direct LLM generation failed: %s", e)
            return {"header": "", "implementation": ""}

    def generate_plugin_with_rag(self, requirements: str) -> Dict[str, str]:
        """Generate plugin code using RAG context from SDK documentation.

        If no query engine is available, falls back to ``generate_plugin``.

        Args:
            requirements: Natural language description of the desired plugin.

        Returns:
            Dict with ``header`` and ``implementation`` C++ source strings.
        """
        if not self.query_engine:
            logger.warning("No query engine available, falling back to direct generation")
            return self.generate_plugin(requirements)

        # Query RAG for relevant SDK documentation
        try:
            rag_query = f"How to implement: {requirements}"
            rag_response = self.query_engine.query(rag_query)
            sdk_context = str(rag_response)
        except Exception as e:
            logger.warning("RAG query failed (%s), falling back to direct generation", e)
            return self.generate_plugin(requirements)

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

        try:
            response = self.llm.complete(prompt)
            code = self._parse_code_response(response.text)
        except Exception as e:
            logger.error("RAG-augmented generation failed: %s", e)
            return {"header": "", "implementation": ""}

        # Validate the generated implementation
        impl = code.get("implementation", "")
        if impl:
            validation = self.validator.validate_syntax(impl)
            if not validation["valid"]:
                logger.warning(
                    "Generated code has syntax issues: %s", validation["errors"]
                )

        return code

    def generate_complete_project(self, requirements: str) -> Dict[str, str]:
        """Generate a full plugin project: header, implementation, CMake, and README.

        Args:
            requirements: Detailed plugin requirements.

        Returns:
            Dict with ``header``, ``implementation``, ``cmake``, and
            ``readme`` keys.
        """
        plugin_code = self.generate_plugin_with_rag(requirements)

        cmake_content = self._generate_cmake_file()
        readme_content = self._generate_readme(requirements)

        return {
            **plugin_code,
            "cmake": cmake_content,
            "readme": readme_content,
        }

    def generate_imgui_window(self, requirements: str) -> str:
        """Generate ImGui window code for BakkesMod settings UI.

        Args:
            requirements: Natural language description of the UI.

        Returns:
            C++ code string for the ImGui window function.
        """
        # Optionally enrich with RAG context
        if self.query_engine:
            try:
                rag_query = f"ImGui window implementation: {requirements}"
                rag_response = self.query_engine.query(rag_query)
                imgui_context = str(rag_response)
            except Exception as e:
                logger.warning("ImGui RAG query failed (%s), using basic context", e)
                imgui_context = "Basic ImGui window structure"
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

        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error("ImGui generation failed: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_code_response(self, response_text: str) -> Dict[str, str]:
        """Extract header and implementation code blocks from LLM output.

        Expects the LLM to return code in fenced blocks labelled
        ``HEADER FILE:`` and ``IMPLEMENTATION FILE:``.

        Args:
            response_text: Raw text returned by the LLM.

        Returns:
            Dict with ``header`` and ``implementation`` strings
            (empty strings if parsing fails).
        """
        header_match = re.search(
            r"HEADER FILE:.*?```cpp\n(.*?)```", response_text, re.DOTALL
        )
        impl_match = re.search(
            r"IMPLEMENTATION FILE:.*?```cpp\n(.*?)```", response_text, re.DOTALL
        )

        return {
            "header": header_match.group(1).strip() if header_match else "",
            "implementation": impl_match.group(1).strip() if impl_match else "",
        }

    @staticmethod
    def _generate_cmake_file() -> str:
        """Generate a minimal CMakeLists.txt for the plugin project.

        Returns:
            CMake configuration string.
        """
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

    @staticmethod
    def _generate_readme(requirements: str) -> str:
        """Generate a basic README for the plugin project.

        Args:
            requirements: The original requirements used for generation.

        Returns:
            Markdown README string.
        """
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
