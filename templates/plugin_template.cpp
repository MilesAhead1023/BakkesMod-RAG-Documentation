#include "{{PLUGIN_NAME}}.h"

BAKKESMOD_PLUGIN({{PLUGIN_NAME}}, "{{PLUGIN_DISPLAY_NAME}}", "{{VERSION}}", {{PLUGIN_TYPE}})

void {{PLUGIN_NAME}}::onLoad()
{
    LOG("{} loaded!", GetNameSafe());

    {{ON_LOAD_CODE}}
}

void {{PLUGIN_NAME}}::onUnload()
{
    LOG("{} unloaded!", GetNameSafe());
}

{{EVENT_HANDLER_IMPLEMENTATIONS}}
