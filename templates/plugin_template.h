#pragma once
#include "bakkesmod/plugin/bakkesmodplugin.h"
#include "bakkesmod/plugin/pluginwindow.h"

/**
 * {{PLUGIN_DESCRIPTION}}
 */
class {{PLUGIN_NAME}} : public BakkesMod::Plugin::BakkesModPlugin
{
public:
    // Plugin lifecycle
    virtual void onLoad() override;
    virtual void onUnload() override;

    // Event handlers
    {{EVENT_HANDLERS}}

private:
    // Members
    {{PRIVATE_MEMBERS}}
};
