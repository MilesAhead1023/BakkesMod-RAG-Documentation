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
from typing import Dict, List, Optional, Any

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

    def validate_semantic(self, files: Dict[str, str]) -> Dict:
        """Validate BakkesMod-specific semantic patterns across project files.

        Checks for essential BakkesMod plugin patterns like the BAKKESMOD_PLUGIN
        macro, proper includes, onLoad/onUnload implementations, and
        _globalCvarManager declaration.

        Args:
            files: Dict mapping filename to file content (e.g.
                ``{"plugin.h": "...", "plugin.cpp": "..."}``).

        Returns:
            Dict with ``valid`` (bool), ``errors`` (list), and ``warnings``
            (list) describing what's missing.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Combine all content for cross-file checks
        all_content = "\n".join(files.values())

        # Find the main .cpp file (implementation)
        impl_files = {k: v for k, v in files.items() if k.endswith(".cpp") and k != "pch.cpp" and k != "GuiBase.cpp"}
        header_files = {k: v for k, v in files.items() if k.endswith(".h") and k not in ("pch.h", "version.h", "logging.h", "resource.h", "GuiBase.h")}

        # 1. Check pch.h include as first include in .cpp files
        for fname, content in impl_files.items():
            lines = [l.strip() for l in content.split("\n") if l.strip() and not l.strip().startswith("//")]
            includes = [l for l in lines if l.startswith("#include")]
            if includes and includes[0] != '#include "pch.h"':
                warnings.append(
                    f'{fname}: first #include should be "pch.h" '
                    f'(found: {includes[0] if includes else "none"})'
                )

        # 2. Check BAKKESMOD_PLUGIN() macro present
        if not re.search(r"BAKKESMOD_PLUGIN\s*\(", all_content):
            errors.append("Missing BAKKESMOD_PLUGIN() macro in implementation")

        # 3. Check onLoad() implementation
        if not re.search(r"::onLoad\s*\(\s*\)", all_content):
            errors.append("Missing onLoad() implementation")

        # 4. Check _globalCvarManager declaration
        if not re.search(r"_globalCvarManager", all_content):
            warnings.append(
                "Missing _globalCvarManager declaration "
                "(needed for LOG/DEBUGLOG macros)"
            )

        # 5. Check #pragma once or include guards in headers
        for fname, content in header_files.items():
            has_pragma = "#pragma once" in content
            has_guard = bool(re.search(r"#ifndef\s+\w+_H", content, re.IGNORECASE))
            if not has_pragma and not has_guard:
                warnings.append(f"{fname}: missing #pragma once or include guard")

        # 6. Check plugin class inherits from BakkesModPlugin
        if not re.search(r":\s*public\s+BakkesMod::Plugin::BakkesModPlugin", all_content):
            errors.append(
                "Plugin class must inherit from "
                "BakkesMod::Plugin::BakkesModPlugin"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def validate_project(self, project_files: Dict[str, str]) -> Dict:
        """Full project validation: syntax + semantic + cross-file consistency.

        Runs syntax validation on each code file, then semantic validation
        across all files, and finally checks cross-file consistency (e.g.
        header/implementation matching).

        Args:
            project_files: Dict mapping filename to file content for the
                entire project.

        Returns:
            Dict with ``valid`` (bool), ``errors`` (list), ``warnings``
            (list), and ``syntax_results`` (per-file syntax validation).
        """
        all_errors: List[str] = []
        all_warnings: List[str] = []
        syntax_results: Dict[str, Dict] = {}

        # 1. Syntax check each code file
        for fname, content in project_files.items():
            if fname.endswith((".cpp", ".h")):
                result = self.validate_syntax(content)
                syntax_results[fname] = result
                if not result["valid"]:
                    for err in result["errors"]:
                        all_errors.append(f"{fname}: {err}")

        # 2. Semantic validation
        semantic = self.validate_semantic(project_files)
        all_errors.extend(semantic["errors"])
        all_warnings.extend(semantic["warnings"])

        # 3. Cross-file consistency: check .h/.cpp pairs exist
        cpp_files = [f for f in project_files if f.endswith(".cpp")]
        h_files = [f for f in project_files if f.endswith(".h")]

        # Main plugin should have both .h and .cpp
        plugin_cpps = [f for f in cpp_files if f not in ("pch.cpp", "GuiBase.cpp")]
        plugin_hs = [f for f in h_files if f not in ("pch.h", "version.h", "logging.h", "resource.h", "GuiBase.h")]

        if plugin_cpps and not plugin_hs:
            all_warnings.append("Plugin has .cpp but no .h header file")
        if plugin_hs and not plugin_cpps:
            all_warnings.append("Plugin has .h header but no .cpp implementation")

        # 4. Check essential support files
        if "pch.h" not in project_files:
            all_warnings.append("Missing pch.h (precompiled header)")
        if "version.h" not in project_files:
            all_warnings.append("Missing version.h")

        return {
            "valid": len(all_errors) == 0,
            "errors": all_errors,
            "warnings": all_warnings,
            "syntax_results": syntax_results,
        }


# ---------------------------------------------------------------------------
# PluginTemplateEngine
# ---------------------------------------------------------------------------

class PluginTemplateEngine:
    """Generates deterministic code templates for BakkesMod plugins.

    These templates require no LLM calls -- they produce syntactically
    correct scaffolding that a developer (or an LLM) can fill in.

    Supports two modes:
    - **Simple**: ``generate_basic_plugin()`` for a minimal .h/.cpp pair.
    - **Complete project**: ``generate_complete_project()`` for all 12 files
      matching the official BakkesMod plugin template structure.

    Feature flags (passed to ``generate_complete_project``):
    - ``settings_window`` — SettingsWindowBase inheritance + RenderSettings()
    - ``plugin_window`` — PluginWindowBase inheritance + RenderWindow()
    - ``cvars`` — registerCvar() boilerplate in onLoad()
    - ``event_hooks`` — HookEvent() boilerplate
    - ``drawable`` — RegisterDrawable() boilerplate
    - ``imgui`` — ImGui rendering boilerplate
    """

    # ------------------------------------------------------------------
    # Backward-compatible simple generators
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Complete project generation (all 12 files)
    # ------------------------------------------------------------------

    def generate_complete_project(
        self,
        plugin_name: str,
        description: str = "A BakkesMod plugin",
        features: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Generate a complete BakkesMod plugin project with all files.

        Produces a full Visual Studio project matching the official BakkesMod
        plugin template, with ``$projectname$`` substitution applied.

        Args:
            plugin_name: C++ class name / project name.
            description: Short plugin description for the BAKKESMOD_PLUGIN macro.
            features: Optional list of feature flags to enable. Supported:
                ``"settings_window"``, ``"plugin_window"``, ``"cvars"``,
                ``"event_hooks"``, ``"drawable"``, ``"imgui"``.

        Returns:
            Dict mapping filename to file content for all project files.
        """
        features = features or []
        feat = set(features)

        files: Dict[str, str] = {}

        files[f"{plugin_name}.h"] = self._gen_plugin_h(plugin_name, feat)
        files[f"{plugin_name}.cpp"] = self._gen_plugin_cpp(plugin_name, description, feat)
        files["pch.h"] = self._gen_pch_h()
        files["pch.cpp"] = self._gen_pch_cpp()
        files["version.h"] = self._gen_version_h()
        files["logging.h"] = self._gen_logging_h()
        files["GuiBase.h"] = self._gen_guibase_h(plugin_name)
        files["GuiBase.cpp"] = self._gen_guibase_cpp(plugin_name)
        files["resource.h"] = self._gen_resource_h()
        files[f"{plugin_name}.rc"] = self._gen_plugin_rc(plugin_name)
        files["BakkesMod.props"] = self._gen_bakkesmod_props()
        files[f"{plugin_name}.vcxproj"] = self._gen_vcxproj(plugin_name)

        return files

    # --- Individual file generators ---

    def _gen_plugin_h(self, name: str, feat: set) -> str:
        """Generate plugin header with conditional feature blocks."""
        # Build inheritance list
        bases = ["public BakkesMod::Plugin::BakkesModPlugin"]
        settings_inherit = ""
        window_inherit = ""
        if "settings_window" in feat:
            settings_inherit = "\n\t,public SettingsWindowBase"
        if "plugin_window" in feat:
            window_inherit = "\n\t,public PluginWindowBase"

        # Build public methods
        public_methods = ""
        if "settings_window" in feat:
            public_methods += "\n\tvoid RenderSettings() override;"
        if "plugin_window" in feat:
            public_methods += "\n\tvoid RenderWindow() override;"

        return f'''#pragma once

#include "GuiBase.h"
#include "bakkesmod/plugin/bakkesmodplugin.h"
#include "bakkesmod/plugin/pluginwindow.h"
#include "bakkesmod/plugin/PluginSettingsWindow.h"

#include "version.h"
constexpr auto plugin_version = stringify(VERSION_MAJOR) "." stringify(VERSION_MINOR) "." stringify(VERSION_PATCH) "." stringify(VERSION_BUILD);


class {name}: public BakkesMod::Plugin::BakkesModPlugin{settings_inherit}{window_inherit}
{{

\tvoid onLoad() override;
\tvoid onUnload() override;

public:{public_methods}
}};
'''

    def _gen_plugin_cpp(self, name: str, description: str, feat: set) -> str:
        """Generate plugin implementation with conditional feature blocks."""
        on_load_body = "\t_globalCvarManager = cvarManager;\n"

        if "cvars" in feat:
            on_load_body += (
                '\n\tauto cvar = cvarManager->registerCvar("'
                f'{name}_enabled", "0", "Enable {name}", '
                "true, true, 0, true, 1);\n"
                "\tcvar.addOnValueChanged([this](std::string cvarName, "
                "CVarWrapper newCvar) {\n"
                '\t\tLOG("CVar {} changed to: {}", cvarName, '
                "newCvar.getStringValue());\n"
                "\t});\n"
            )

        if "event_hooks" in feat:
            on_load_body += (
                '\n\tgameWrapper->HookEvent("Function TAGame.Ball_TA.Explode", '
                "[this](std::string eventName) {\n"
                '\t\tLOG("Ball exploded!");\n'
                "\t});\n"
            )

        if "drawable" in feat:
            on_load_body += (
                "\n\tgameWrapper->RegisterDrawable("
                f"std::bind(&{name}::Render, this, "
                "std::placeholders::_1));\n"
            )

        on_load_body += f'\n\tLOG("{name} loaded!");'

        # Build render/settings methods
        extra_methods = ""
        if "settings_window" in feat:
            extra_methods += f"""

void {name}::RenderSettings()
{{
\tImGui::TextUnformatted("{name} Settings");
\tImGui::Separator();

\t// Add your settings UI here
\tImGui::TextUnformatted("Configure {name} options below:");
}}"""

        if "plugin_window" in feat:
            extra_methods += f"""

void {name}::RenderWindow()
{{
\tImGui::TextUnformatted("{name}");
\tImGui::Separator();

\t// Add your window UI here
\tImGui::TextUnformatted("Plugin window content goes here.");
}}"""

        if "drawable" in feat:
            extra_methods += f"""

void {name}::Render(CanvasWrapper canvas)
{{
\t// Add your drawable rendering here
\tcanvas.SetColor(255, 255, 255, 255);
}}"""

        return f'''#include "pch.h"
#include "{name}.h"


BAKKESMOD_PLUGIN({name}, "{description}", plugin_version, PLUGINTYPE_FREEPLAY)

std::shared_ptr<CVarManagerWrapper> _globalCvarManager;

void {name}::onLoad()
{{
{on_load_body}
}}

void {name}::onUnload()
{{
\tLOG("{name} unloaded!");
}}{extra_methods}
'''

    @staticmethod
    def _gen_pch_h() -> str:
        return '''#pragma once

#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#include "bakkesmod/plugin/bakkesmodplugin.h"

#include <string>
#include <vector>
#include <functional>
#include <memory>

#include "IMGUI/imgui.h"
#include "IMGUI/imgui_stdlib.h"
#include "IMGUI/imgui_searchablecombo.h"
#include "IMGUI/imgui_rangeslider.h"

#include "logging.h"
'''

    @staticmethod
    def _gen_pch_cpp() -> str:
        return '#include "pch.h"\n'

    @staticmethod
    def _gen_version_h() -> str:
        return '''#pragma once
#define VERSION_MAJOR 1
#define VERSION_MINOR 0
#define VERSION_PATCH 0
#define VERSION_BUILD 0

#define stringify(a) stringify_(a)
#define stringify_(a) #a
'''

    @staticmethod
    def _gen_logging_h() -> str:
        return r'''// ReSharper disable CppNonExplicitConvertingConstructor
#pragma once
#include <string>
#include <source_location>
#include <format>
#include <memory>

#include "bakkesmod/wrappers/cvarmanagerwrapper.h"

extern std::shared_ptr<CVarManagerWrapper> _globalCvarManager;
constexpr bool DEBUG_LOG = false;


struct FormatString
{
	std::string_view str;
	std::source_location loc{};

	FormatString(const char* str, const std::source_location& loc = std::source_location::current()) : str(str), loc(loc)
	{
	}

	FormatString(const std::string&& str, const std::source_location& loc = std::source_location::current()) : str(str), loc(loc)
	{
	}

	[[nodiscard]] std::string GetLocation() const
	{
		return std::format("[{} ({}:{})]\n", loc.function_name(), loc.file_name(), loc.line());
	}
};

struct FormatWstring
{
	std::wstring_view str;
	std::source_location loc{};

	FormatWstring(const wchar_t* str, const std::source_location& loc = std::source_location::current()) : str(str), loc(loc)
	{
	}

	FormatWstring(const std::wstring&& str, const std::source_location& loc = std::source_location::current()) : str(str), loc(loc)
	{
	}

	[[nodiscard]] std::wstring GetLocation() const
	{
		auto basic_string = std::format("[{} ({}:{})]\n", loc.function_name(), loc.file_name(), loc.line());
		return std::wstring(basic_string.begin(), basic_string.end());
	}
};


template <typename... Args>
void LOG(std::string_view format_str, Args&&... args)
{
	_globalCvarManager->log(std::vformat(format_str, std::make_format_args(args...)));
}

template <typename... Args>
void LOG(std::wstring_view format_str, Args&&... args)
{
	_globalCvarManager->log(std::vformat(format_str, std::make_wformat_args(args...)));
}


template <typename... Args>
void DEBUGLOG(const FormatString& format_str, Args&&... args)
{
	if constexpr (DEBUG_LOG)
	{
		auto text = std::vformat(format_str.str, std::make_format_args(args...));
		auto location = format_str.GetLocation();
		_globalCvarManager->log(std::format("{} {}", text, location));
	}
}

template <typename... Args>
void DEBUGLOG(const FormatWstring& format_str, Args&&... args)
{
	if constexpr (DEBUG_LOG)
	{
		auto text = std::vformat(format_str.str, std::make_wformat_args(args...));
		auto location = format_str.GetLocation();
		_globalCvarManager->log(std::format(L"{} {}", text, location));
	}
}
'''

    @staticmethod
    def _gen_guibase_h(name: str) -> str:
        return f'''#pragma once
#include "bakkesmod/plugin/PluginSettingsWindow.h"
#include "bakkesmod/plugin/pluginwindow.h"

class SettingsWindowBase : public BakkesMod::Plugin::PluginSettingsWindow
{{
public:
\tstd::string GetPluginName() override;
\tvoid SetImGuiContext(uintptr_t ctx) override;
}};

class PluginWindowBase : public BakkesMod::Plugin::PluginWindow
{{
public:
\tvirtual ~PluginWindowBase() = default;

\tbool isWindowOpen_ = false;
\tstd::string menuTitle_ = "{name}";

\tstd::string GetMenuName() override;
\tstd::string GetMenuTitle() override;
\tvoid SetImGuiContext(uintptr_t ctx) override;
\tbool ShouldBlockInput() override;
\tbool IsActiveOverlay() override;
\tvoid OnOpen() override;
\tvoid OnClose() override;
\tvoid Render() override;

\tvirtual void RenderWindow() = 0;
}};
'''

    @staticmethod
    def _gen_guibase_cpp(name: str) -> str:
        return f'''#include "pch.h"
#include "GuiBase.h"

std::string SettingsWindowBase::GetPluginName()
{{
\treturn "{name}";
}}

void SettingsWindowBase::SetImGuiContext(uintptr_t ctx)
{{
\tImGui::SetCurrentContext(reinterpret_cast<ImGuiContext*>(ctx));
}}

std::string PluginWindowBase::GetMenuName()
{{
\treturn "{name}";
}}

std::string PluginWindowBase::GetMenuTitle()
{{
\treturn menuTitle_;
}}

void PluginWindowBase::SetImGuiContext(uintptr_t ctx)
{{
\tImGui::SetCurrentContext(reinterpret_cast<ImGuiContext*>(ctx));
}}

bool PluginWindowBase::ShouldBlockInput()
{{
\treturn ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard;
}}

bool PluginWindowBase::IsActiveOverlay()
{{
\treturn true;
}}

void PluginWindowBase::OnOpen()
{{
\tisWindowOpen_ = true;
}}

void PluginWindowBase::OnClose()
{{
\tisWindowOpen_ = false;
}}

void PluginWindowBase::Render()
{{
\tif (!ImGui::Begin(menuTitle_.c_str(), &isWindowOpen_, ImGuiWindowFlags_None))
\t{{
\t\tImGui::End();
\t\treturn;
\t}}

\tRenderWindow();

\tImGui::End();

\tif (!isWindowOpen_)
\t{{
\t\t_globalCvarManager->executeCommand("togglemenu " + GetMenuName());
\t}}
}}
'''

    @staticmethod
    def _gen_resource_h() -> str:
        return '''//{{NO_DEPENDENCIES}}
// Microsoft Visual C++ generated include file.

// Next default values for new objects
//
#ifdef APSTUDIO_INVOKED
#ifndef APSTUDIO_READONLY_SYMBOLS
#define _APS_NEXT_RESOURCE_VALUE        101
#define _APS_NEXT_COMMAND_VALUE         40001
#define _APS_NEXT_CONTROL_VALUE         1001
#define _APS_NEXT_SYMED_VALUE           101
#endif
#endif
'''

    @staticmethod
    def _gen_plugin_rc(name: str) -> str:
        return f'''// Microsoft Visual C++ generated resource script.
//
#include "resource.h"
#include "version.h"

#define APSTUDIO_READONLY_SYMBOLS
#include "winres.h"
#undef APSTUDIO_READONLY_SYMBOLS

/////////////////////////////////////////////////////////////////////////////
// Version
//

VS_VERSION_INFO VERSIONINFO
FILEVERSION VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, VERSION_BUILD
PRODUCTVERSION VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, VERSION_BUILD
 FILEFLAGSMASK 0x3fL
#ifdef _DEBUG
 FILEFLAGS 0x1L
#else
 FILEFLAGS 0x0L
#endif
 FILEOS 0x40004L
 FILETYPE 0x0L
 FILESUBTYPE 0x0L
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "040904b0"
        BEGIN
            VALUE "FileDescription", "{name} BakkesMod Plugin"
            VALUE "FileVersion", stringify(VERSION_MAJOR) "." stringify(VERSION_MINOR) "." stringify(VERSION_PATCH) "." stringify(VERSION_BUILD)
            VALUE "InternalName", "{name}.dll"
            VALUE "OriginalFilename", "{name}.dll"
            VALUE "ProductName", "{name}"
            VALUE "ProductVersion", stringify(VERSION_MAJOR) "." stringify(VERSION_MINOR) "." stringify(VERSION_PATCH) "." stringify(VERSION_BUILD)
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x409, 1200
    END
END
'''

    @staticmethod
    def _gen_bakkesmod_props() -> str:
        return '''<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="ShowBakkesInfo" ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ImportGroup Label="PropertySheets" />
  <PropertyGroup Label="UserMacros">
    <BakkesModPath>$(registry:HKEY_CURRENT_USER\\Software\\BakkesMod\\AppPath@BakkesModPath)</BakkesModPath>
  </PropertyGroup>
  <ItemDefinitionGroup>
    <ClCompile>
      <AdditionalIncludeDirectories>$(BakkesModPath)\\bakkesmodsdk\\include;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
    </ClCompile>
    <Link>
      <AdditionalLibraryDirectories>$(BakkesModPath)\\bakkesmodsdk\\lib;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>
      <AdditionalDependencies>pluginsdk.lib;%(AdditionalDependencies)</AdditionalDependencies>
    </Link>
    <PostBuildEvent>
      <Command> "$(BakkesModPath)\\bakkesmodsdk\\bakkesmod-patch.exe" "$(TargetPath)"</Command>
    </PostBuildEvent>
  </ItemDefinitionGroup>
  <ItemGroup />
  <Target Name="ShowBakkesInfo" BeforeTargets="PrepareForBuild">
    <Message Text="Using bakkes found at $(BakkesModPath)" Importance="normal" />
  </Target>
</Project>
'''

    @staticmethod
    def _gen_vcxproj(name: str) -> str:
        return f'''<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <VCProjectVersion>16.0</VCProjectVersion>
    <RootNamespace>{name}</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <ConfigurationType>DynamicLibrary</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.props" />
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <Import Project="$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
    <Import Project="BakkesMod.props" />
  </ImportGroup>
  <PropertyGroup Label="UserMacros" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <LinkIncremental>false</LinkIncremental>
    <OutDir>$(SolutionDir)plugins\\</OutDir>
    <IntDir>$(SolutionDir)build\\.intermediates\\$(Configuration)\\</IntDir>
  </PropertyGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <SDLCheck>true</SDLCheck>
      <PreprocessorDefinitions>NDEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
      <PrecompiledHeader>Use</PrecompiledHeader>
      <PrecompiledHeaderFile>pch.h</PrecompiledHeaderFile>
      <RuntimeLibrary>MultiThreaded</RuntimeLibrary>
      <LanguageStandard>stdcpp20</LanguageStandard>
      <MultiProcessorCompilation>true</MultiProcessorCompilation>
      <AdditionalIncludeDirectories>$(ProjectDir);%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <GenerateDebugInformation>true</GenerateDebugInformation>
    </Link>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ClCompile Include="pch.cpp">
      <PrecompiledHeader Condition="'$(Configuration)|$(Platform)'=='Release|x64'">Create</PrecompiledHeader>
    </ClCompile>
    <ClCompile Include="{name}.cpp" />
    <ClCompile Include="GuiBase.cpp" />
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="logging.h" />
    <ClInclude Include="pch.h" />
    <ClInclude Include="GuiBase.h" />
    <ClInclude Include="{name}.h" />
    <ClInclude Include="version.h" />
  </ItemGroup>
  <ItemGroup>
    <ResourceCompile Include="{name}.rc" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />
</Project>
'''


# ---------------------------------------------------------------------------
# BakkesModCodeGenerator
# ---------------------------------------------------------------------------

class BakkesModCodeGenerator:
    """LLM-powered BakkesMod plugin code generator.

    Unlike the old root-level ``CodeGenerator``, this class does **not**
    build its own LLM fallback chain or load indexes.  Instead it receives
    a pre-verified LLM and an optional query engine from the caller (typically
    the ``RAGEngine``).

    When self-improving mode is enabled (via ``CodeGenConfig``), generated
    code goes through an iterative validate → fix → compile → fix loop
    that feeds errors back to the LLM for correction.  A feedback store
    records generation history so that successful patterns can be reused
    as few-shot examples.

    Args:
        llm: A verified LlamaIndex LLM instance (from ``llm_provider.get_llm``).
        query_engine: Optional LlamaIndex query engine for RAG-augmented
            generation.  When ``None``, ``generate_plugin_with_rag`` falls
            back to direct LLM generation.
        config: Optional ``CodeGenConfig``.  Controls self-improving mode,
            compilation, and feedback persistence.
    """

    def __init__(self, llm, query_engine=None, config=None):
        """Initialize the code generator.

        Args:
            llm: A verified LlamaIndex LLM instance.
            query_engine: Optional query engine for RAG context lookups.
            config: Optional ``CodeGenConfig`` for self-improving settings.
        """
        self.llm = llm
        self.query_engine = query_engine
        self.template_engine = PluginTemplateEngine()
        self.validator = CodeValidator()

        # Import config type lazily to avoid circular imports
        from bakkesmod_rag.config import CodeGenConfig
        self._config = config or CodeGenConfig()

        # Compiler (optional — graceful if MSVC not found)
        self.compiler = None
        if self._config.enable_compilation:
            try:
                from bakkesmod_rag.compiler import PluginCompiler
                sdk_dirs = self._find_sdk_include_dirs()
                self.compiler = PluginCompiler(
                    msvc_path=self._config.msvc_path,
                    sdk_include_dirs=sdk_dirs,
                )
                if not self.compiler.available:
                    self.compiler = None
            except Exception as e:
                logger.warning("Compiler initialization failed: %s", e)
                self.compiler = None

        # Feedback store (optional)
        self.feedback = None
        if self._config.feedback_enabled:
            try:
                from bakkesmod_rag.feedback_store import FeedbackStore
                self.feedback = FeedbackStore(
                    feedback_dir=self._config.feedback_dir,
                )
            except Exception as e:
                logger.warning("Feedback store initialization failed: %s", e)
                self.feedback = None

    @staticmethod
    def _find_sdk_include_dirs() -> List[str]:
        """Find BakkesMod SDK header directories for compiler include paths."""
        import os
        dirs = []
        for candidate in ("docs_bakkesmod_only", "templates"):
            if os.path.isdir(candidate):
                dirs.append(os.path.abspath(candidate))
        return dirs

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

    def generate_full_plugin_with_rag(self, description: str) -> Dict:
        """Generate a complete BakkesMod plugin project using RAG context.

        When self-improving mode is enabled, the code goes through an
        iterative validate → fix → compile → fix loop:

        1. Retrieve RAG context from SDK documentation.
        2. Check feedback store for similar successful patterns (few-shot).
        3. Detect features and derive plugin name.
        4. Generate plugin.h and plugin.cpp with LLM + RAG context.
        5. Wrap in full project scaffold (all 12 files).
        6. **Validate** — if errors, feed them back to LLM for fix.
        7. **Compile** (if MSVC available) — if errors, feed them back.
        8. Repeat steps 6-7 up to ``max_fix_iterations`` times.
        9. Return the best version (fewest total errors).

        Args:
            description: Natural language description of the desired plugin.

        Returns:
            Dict with ``project_files``, ``header``, ``implementation``,
            ``features_used``, ``validation``, ``explanation``,
            ``fix_iterations``, ``fix_history``, ``compile_result``,
            and ``generation_id``.
        """
        # 1. Retrieve RAG context
        sdk_context = ""
        if self.query_engine:
            try:
                rag_query = f"BakkesMod SDK: How to implement: {description}"
                rag_response = self.query_engine.query(rag_query)
                sdk_context = str(rag_response)
            except Exception as e:
                logger.warning("RAG context retrieval failed: %s", e)

        # 2. Get few-shot examples from feedback store
        few_shot_prompt = ""
        if self.feedback:
            features_preview = self._detect_features(description)
            few_shot_prompt = self.feedback.format_few_shot_prompt(features_preview)

        # 3. Detect features from description
        features = self._detect_features(description)

        # 4. Derive a C++ class name from description
        plugin_name = self._derive_plugin_name(description)

        # 5. Generate plugin.h and plugin.cpp with LLM
        llm_code = self._generate_with_llm(
            plugin_name, description, features, sdk_context,
            few_shot_context=few_shot_prompt,
        )

        # 6. Generate full project scaffold
        project_files = self.template_engine.generate_complete_project(
            plugin_name=plugin_name,
            description=description,
            features=features,
        )

        # Override plugin.h and plugin.cpp with LLM-generated versions
        header_key = f"{plugin_name}.h"
        impl_key = f"{plugin_name}.cpp"
        if llm_code.get("header"):
            project_files[header_key] = llm_code["header"]
        if llm_code.get("implementation"):
            project_files[impl_key] = llm_code["implementation"]

        # 7. Validate (and optionally self-improve)
        fix_history: List[Dict[str, Any]] = []
        best_files = dict(project_files)
        best_error_count = float("inf")
        compile_result_dict: Optional[Dict] = None

        if self._config.self_improving:
            max_iters = self._config.max_fix_iterations
            prev_error_sig = None

            for iteration in range(max_iters):
                # Validate
                validation = self.validator.validate_project(project_files)
                val_errors = validation.get("errors", [])
                val_warnings = validation.get("warnings", [])

                # Compile (if compiler available)
                compile_errors: List[str] = []
                compile_warnings: List[str] = []
                compile_success = None
                if self.compiler:
                    from bakkesmod_rag.compiler import CompileResult
                    cr = self.compiler.compile_project(project_files)
                    compile_success = cr.success
                    compile_errors = [str(e) for e in cr.errors]
                    compile_warnings = [str(w) for w in cr.warnings]
                    compile_result_dict = {
                        "success": cr.success,
                        "errors": compile_errors,
                        "warnings": compile_warnings,
                        "output": cr.output[:500],
                    }

                all_errors = val_errors + compile_errors
                total_errors = len(all_errors)

                # Track best version (fewest errors)
                if total_errors < best_error_count:
                    best_error_count = total_errors
                    best_files = dict(project_files)

                fix_history.append({
                    "iteration": iteration,
                    "validation_errors": val_errors,
                    "validation_warnings": val_warnings,
                    "compile_errors": compile_errors,
                    "compile_warnings": compile_warnings,
                    "compile_success": compile_success,
                    "total_errors": total_errors,
                })

                logger.info(
                    "Self-improving iteration %d/%d: %d errors (%d validation, %d compile)",
                    iteration + 1, max_iters, total_errors,
                    len(val_errors), len(compile_errors),
                )

                # Stop if no errors
                if total_errors == 0:
                    logger.info("Code is clean after %d iterations", iteration + 1)
                    break

                # Stop if same errors as last iteration (no progress)
                error_sig = tuple(sorted(all_errors))
                if error_sig == prev_error_sig:
                    logger.info(
                        "No progress (same %d errors), stopping fix loop",
                        total_errors,
                    )
                    break
                prev_error_sig = error_sig

                # Don't fix on the last iteration — just report
                if iteration >= max_iters - 1:
                    break

                # Feed errors back to LLM for correction
                fixed_code = self._fix_with_llm(
                    project_files, all_errors, iteration, plugin_name,
                )
                if fixed_code.get("header"):
                    project_files[header_key] = fixed_code["header"]
                if fixed_code.get("implementation"):
                    project_files[impl_key] = fixed_code["implementation"]
        else:
            # Self-improving disabled — single validation pass
            validation = self.validator.validate_project(project_files)
            best_files = dict(project_files)
            if self.compiler:
                cr = self.compiler.compile_project(project_files)
                compile_result_dict = {
                    "success": cr.success,
                    "errors": [str(e) for e in cr.errors],
                    "warnings": [str(w) for w in cr.warnings],
                    "output": cr.output[:500],
                }

        # Use best version
        project_files = best_files
        final_validation = self.validator.validate_project(project_files)

        # Record in feedback store
        generation_id = None
        if self.feedback:
            generation_id = self.feedback.record_generation(
                description=description,
                features=features,
                generated_files=project_files,
                validation_errors=final_validation.get("errors", []),
                validation_warnings=final_validation.get("warnings", []),
                fix_iterations=len(fix_history),
                compile_attempted=self.compiler is not None,
                compile_success=(
                    compile_result_dict.get("success", False)
                    if compile_result_dict else False
                ),
                compiler_errors=(
                    compile_result_dict.get("errors", [])
                    if compile_result_dict else []
                ),
            )

        return {
            "project_files": project_files,
            "header": project_files.get(header_key, ""),
            "implementation": project_files.get(impl_key, ""),
            "features_used": features,
            "validation": final_validation,
            "explanation": description,
            "fix_iterations": len(fix_history),
            "fix_history": fix_history,
            "compile_result": compile_result_dict,
            "generation_id": generation_id,
        }

    def record_feedback(
        self, generation_id: str, accepted: bool, user_edits: Optional[str] = None
    ) -> bool:
        """Record user feedback for a generation.

        Args:
            generation_id: ID returned from generation.
            accepted: Whether the user accepted the code.
            user_edits: Description of user modifications (optional).

        Returns:
            True if feedback was recorded successfully.
        """
        if not self.feedback or not generation_id:
            return False
        return self.feedback.update_feedback(generation_id, accepted, user_edits)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _FEATURE_KEYWORDS = {
        "settings_window": [
            "settings", "configuration", "config tab", "settings window",
            "preferences", "options menu",
        ],
        "plugin_window": [
            "window", "overlay", "ui window", "plugin window", "gui",
            "hud", "display",
        ],
        "cvars": [
            "cvar", "console variable", "configuration variable",
            "toggle", "enable", "disable", "setting",
        ],
        "event_hooks": [
            "hook", "event", "goal", "score", "ball", "boost", "demo",
            "demolition", "match", "tick", "countdown",
        ],
        "drawable": [
            "draw", "canvas", "render on screen", "draw on screen",
            "overlay", "hud", "visual",
        ],
        "imgui": [
            "imgui", "ui", "gui", "interface", "button", "checkbox",
            "slider", "menu", "window",
        ],
    }

    def _detect_features(self, description: str) -> List[str]:
        """Detect which BakkesMod features are needed from a description.

        Scans the description for keywords associated with each feature
        flag and returns the list of detected features.

        Args:
            description: Plugin requirements in natural language.

        Returns:
            List of feature flag strings.
        """
        desc_lower = description.lower()
        detected = []
        for feature, keywords in self._FEATURE_KEYWORDS.items():
            if any(kw in desc_lower for kw in keywords):
                detected.append(feature)
        return detected

    @staticmethod
    def _derive_plugin_name(description: str) -> str:
        """Derive a C++ class name from a natural language description.

        Takes the first few significant words, strips non-alphanumeric
        characters, and converts to PascalCase.

        Args:
            description: Plugin description text.

        Returns:
            A valid C++ identifier (e.g. ``"GoalTracker"``).
        """
        # Strip common filler words
        stopwords = {
            "a", "an", "the", "that", "which", "for", "and", "or", "to",
            "in", "on", "with", "my", "create", "make", "build", "write",
            "plugin", "bakkesmod",
        }
        words = re.findall(r"[a-zA-Z]+", description)
        significant = [w for w in words if w.lower() not in stopwords][:4]

        if not significant:
            return "MyPlugin"

        name = "".join(w.capitalize() for w in significant)

        # Ensure starts with a letter
        if not name[0].isalpha():
            name = "Plugin" + name

        return name

    def _generate_with_llm(
        self,
        plugin_name: str,
        description: str,
        features: List[str],
        sdk_context: str,
        few_shot_context: str = "",
    ) -> Dict[str, str]:
        """Use LLM to generate plugin.h and plugin.cpp with SDK context.

        Args:
            plugin_name: C++ class name.
            description: User's requirements.
            features: Detected feature flags.
            sdk_context: RAG-retrieved SDK documentation.
            few_shot_context: Optional few-shot examples from feedback store.

        Returns:
            Dict with ``header`` and ``implementation`` (empty on failure).
        """
        features_desc = ", ".join(features) if features else "basic plugin"

        prompt = f"""You are a BakkesMod plugin code generator.

USER REQUIREMENTS:
{description}

FEATURES TO IMPLEMENT: {features_desc}

RELEVANT SDK DOCUMENTATION:
{sdk_context or "No SDK context available."}
{few_shot_context}
Generate a complete BakkesMod plugin with these EXACT patterns:

CRITICAL REQUIREMENTS:
- Class name: {plugin_name}
- First include in .cpp MUST be: #include "pch.h"
- Second include: #include "{plugin_name}.h"
- BAKKESMOD_PLUGIN({plugin_name}, "{description[:60]}", plugin_version, PLUGINTYPE_FREEPLAY)
- Declare: std::shared_ptr<CVarManagerWrapper> _globalCvarManager;
- In onLoad(): _globalCvarManager = cvarManager;
- Include "version.h" in .h and use: constexpr auto plugin_version = stringify(VERSION_MAJOR) "." ...
- Header must have #pragma once
- Class inherits from BakkesMod::Plugin::BakkesModPlugin
- Include "GuiBase.h" in the header
- Use LOG() for logging (NOT cvarManager->log)
- Use HookEvent / HookEventWithCallerPost for event hooks
- Use cvarManager->registerCvar for CVars
- Use gameWrapper->RegisterDrawable for canvas drawing

Return ONLY the code in this exact format:

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

            # Validate the generated code
            if code.get("implementation"):
                validation = self.validator.validate_syntax(code["implementation"])
                if not validation["valid"]:
                    logger.warning(
                        "LLM-generated code has syntax issues: %s",
                        validation["errors"],
                    )
            return code

        except Exception as e:
            logger.error("LLM code generation failed: %s", e)
            return {"header": "", "implementation": ""}

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

    def _format_error_feedback(
        self,
        errors: List[str],
        project_files: Dict[str, str],
        plugin_name: str,
    ) -> str:
        """Format validation/compiler errors as an LLM feedback prompt.

        Constructs a structured prompt that describes each error and
        includes the current code so the LLM can see exactly what needs
        fixing.

        Args:
            errors: List of error description strings.
            project_files: Current project files.
            plugin_name: Plugin class name (to find the right files).

        Returns:
            Formatted feedback prompt string.
        """
        header_key = f"{plugin_name}.h"
        impl_key = f"{plugin_name}.cpp"

        header_code = project_files.get(header_key, "")
        impl_code = project_files.get(impl_key, "")

        error_list = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(errors))

        return f"""The following errors were found in the generated code:

{error_list}

CURRENT HEADER FILE ({header_key}):
```cpp
{header_code}
```

CURRENT IMPLEMENTATION FILE ({impl_key}):
```cpp
{impl_code}
```

Fix ALL of the listed errors. Keep the same plugin structure and class name.
Return the corrected code in the same format:

HEADER FILE:
```cpp
[corrected header]
```

IMPLEMENTATION FILE:
```cpp
[corrected implementation]
```
"""

    def _fix_with_llm(
        self,
        project_files: Dict[str, str],
        errors: List[str],
        iteration: int,
        plugin_name: str,
    ) -> Dict[str, str]:
        """Send errors and current code back to the LLM for correction.

        Args:
            project_files: Current project files with errors.
            errors: List of error descriptions to fix.
            iteration: Current fix iteration number (for logging).
            plugin_name: Plugin class name.

        Returns:
            Dict with ``header`` and ``implementation`` (corrected code,
            or empty strings on failure).
        """
        feedback = self._format_error_feedback(errors, project_files, plugin_name)

        prompt = f"""You are a BakkesMod plugin code fixer. Fix iteration {iteration + 1}.

{feedback}"""

        try:
            response = self.llm.complete(prompt)
            fixed = self._parse_code_response(response.text)
            if fixed.get("header") or fixed.get("implementation"):
                logger.info(
                    "LLM fix iteration %d produced corrected code", iteration + 1
                )
            else:
                logger.warning(
                    "LLM fix iteration %d returned unparseable response",
                    iteration + 1,
                )
            return fixed
        except Exception as e:
            logger.error("LLM fix attempt %d failed: %s", iteration + 1, e)
            return {"header": "", "implementation": ""}
