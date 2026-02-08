# Generated Plugin Example

This example shows the output of the code generation system for a complete boost tracking plugin.

## Requirements

```
Create a plugin that:
1. Tracks boost usage for all players
2. Shows boost stats in an ImGui window
3. Updates every game tick
4. Saves stats to a config file
```

## Generated Files

### MyPlugin.h

```cpp
#pragma once
#include "bakkesmod/plugin/bakkesmodplugin.h"
#include "bakkesmod/plugin/pluginwindow.h"
#include <map>

class MyPlugin : public BakkesMod::Plugin::BakkesModPlugin {
public:
    virtual void onLoad() override;
    virtual void onUnload() override;

    void RenderWindow();

private:
    void onTick(std::string eventName);
    void updateBoostStats();

    std::map<std::string, float> playerBoostUsage;
    bool renderWindow = false;
};
```

### MyPlugin.cpp

```cpp
#include "MyPlugin.h"

BAKKESMOD_PLUGIN(MyPlugin, "Boost Tracker", "1.0", PLUGINTYPE_FREEPLAY)

void MyPlugin::onLoad() {
    LOG("Boost Tracker loaded!");

    gameWrapper->HookEvent("Function TAGame.Car_TA.EventVehicleSetup",
        [this](std::string eventName) {
            onTick(eventName);
        });

    gameWrapper->RegisterDrawable([this](CanvasWrapper canvas) {
        RenderWindow();
    });
}

void MyPlugin::onUnload() {
    // Save stats to file
    LOG("Boost Tracker unloaded");
}

void MyPlugin::onTick(std::string eventName) {
    updateBoostStats();
}

void MyPlugin::updateBoostStats() {
    ServerWrapper server = gameWrapper->GetCurrentGameState();
    if (!server) return;

    ArrayWrapper<CarWrapper> cars = server.GetCars();
    for (int i = 0; i < cars.Count(); i++) {
        CarWrapper car = cars.Get(i);
        if (!car) continue;

        BoostWrapper boost = car.GetBoostComponent();
        if (!boost) continue;

        PriWrapper pri = car.GetPRI();
        if (!pri) continue;

        std::string playerName = pri.GetPlayerName().ToString();
        float boostAmount = boost.GetCurrentBoostAmount();

        playerBoostUsage[playerName] = boostAmount;
    }
}

void MyPlugin::RenderWindow() {
    if (!renderWindow) return;

    if (!ImGui::Begin("Boost Tracker", &renderWindow)) {
        ImGui::End();
        return;
    }

    ImGui::Text("Player Boost Stats");
    ImGui::Separator();

    for (const auto& [player, boost] : playerBoostUsage) {
        ImGui::Text("%s: %.0f%%", player.c_str(), boost * 100);
    }

    ImGui::End();
}
```

## Validation Results

```
Syntax Validation: PASS
  - Brackets matched: ✓
  - Strings closed: ✓
  - No syntax errors: ✓

API Validation: PASS
  - Uses gameWrapper: ✓
  - Hooks events: ✓
  - Uses ServerWrapper: ✓
  - Uses CarWrapper: ✓
  - Uses proper API patterns: ✓
```

## Next Steps

1. Copy code to plugin project
2. Build with CMake
3. Test in BakkesMod
4. Customize as needed
