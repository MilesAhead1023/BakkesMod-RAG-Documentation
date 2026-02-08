"""
Code Template Engine
====================
Generates BakkesMod plugin code from templates.
"""

from typing import Dict, List, Optional
from pathlib import Path


class PluginTemplateEngine:
    """Generates code templates for BakkesMod plugins."""

    def __init__(self):
        """Initialize template engine."""
        self.template_dir = Path("templates")

    def generate_basic_plugin(self, plugin_name: str, description: str) -> Dict[str, str]:
        """
        Generate basic plugin structure.

        Args:
            plugin_name: Name of the plugin class
            description: Plugin description

        Returns:
            Dict with 'header' and 'implementation' keys
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
            "implementation": implementation
        }

    def generate_event_hook(self, event_name: str, callback_name: str) -> str:
        """
        Generate event hook code.

        Args:
            event_name: Full event name (e.g., "Function TAGame.Ball_TA.OnHitGoal")
            callback_name: Name for the callback function

        Returns:
            C++ code for event hook
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
        elements: Optional[List[str]] = None
    ) -> str:
        """
        Generate ImGui window code.

        Args:
            window_title: Window title
            elements: List of UI element types

        Returns:
            C++ code for ImGui window
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
