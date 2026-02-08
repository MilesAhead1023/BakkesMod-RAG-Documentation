# Code Generation Mode - User Guide

## Overview

The BakkesMod RAG system can now generate complete, working plugin code from natural language requirements. It combines RAG-retrieved SDK documentation with Claude Sonnet 4.5's code generation capabilities to produce production-ready C++ plugins.

## Quick Start

### 1. Start Interactive Mode

```bash
python interactive_rag.py
```

### 2. Use the `/generate` Command

```
[QUERY] > /generate Create a plugin that hooks the goal scored event and logs the scorer's name
```

### 3. Review Generated Code

The system will output:
- **Header file (.h)** - Class declaration
- **Implementation file (.cpp)** - Full implementation
- Syntax validation results
- API usage analysis

## Features

### Basic Plugin Generation

**Command:** `/generate <requirements>`

**Example:**
```
/generate Create a plugin that tracks demolitions and shows the count
```

**Generates:**
- Plugin class structure
- Event hooks
- Logging
- Proper BakkesMod API usage

### Complete Project Generation

**Command:** `/generate-project <detailed requirements>`

**Example:**
```
/generate-project Create a boost tracker plugin with:
- Track boost per player
- Show stats in ImGui window
- Save to config file
```

**Generates:**
- Plugin .h and .cpp files
- CMakeLists.txt
- README.md
- Build instructions

### ImGui Window Generation

**Command:** `/generate-ui <UI requirements>`

**Example:**
```
/generate-ui Settings window with checkbox to enable/disable and slider for update rate
```

**Generates:**
- Complete ImGui window function
- Proper window management
- UI element code

## How It Works

### 1. RAG Context Retrieval

When you request code generation, the system:

1. **Queries the RAG system** for relevant SDK documentation
2. **Retrieves examples** of similar implementations
3. **Extracts API patterns** from official docs

### 2. Code Generation

The system uses Claude Sonnet 4.5 to:

1. **Parse your requirements** into implementation steps
2. **Apply SDK best practices** from RAG context
3. **Generate syntactically correct** C++ code
4. **Follow BakkesMod conventions** (naming, structure, etc.)

### 3. Validation

Generated code is automatically:

1. **Syntax validated** - checks brackets, strings, etc.
2. **API validated** - ensures proper BakkesMod API usage
3. **Pattern validated** - verifies event names, wrapper usage

## Examples

### Example 1: Simple Event Hook

**Requirements:**
```
Create a plugin that detects when a player joins the match
```

**Generated Code:**
```cpp
// MyPlugin.h
#pragma once
#include "bakkesmod/plugin/bakkesmodplugin.h"

class MyPlugin : public BakkesMod::Plugin::BakkesModPlugin {
public:
    virtual void onLoad() override;
    virtual void onUnload() override;

private:
    void onPlayerJoin(std::string eventName);
};

// MyPlugin.cpp
#include "MyPlugin.h"

BAKKESMOD_PLUGIN(MyPlugin, "Player Join Detector", "1.0", PLUGINTYPE_FREEPLAY)

void MyPlugin::onLoad() {
    gameWrapper->HookEvent("Function TAGame.GFxData_MainMenu_TA.MainMenuAdded",
        [this](std::string eventName) {
            onPlayerJoin(eventName);
        });
}

void MyPlugin::onUnload() {
    // Cleanup
}

void MyPlugin::onPlayerJoin(std::string eventName) {
    LOG("Player joined the match!");
}
```

### Example 2: ImGui Settings Window

**Requirements:**
```
Create a settings window with options to toggle plugin on/off and adjust update frequency
```

**Generated Code:**
```cpp
void MyPlugin::RenderSettingsWindow() {
    if (!ImGui::Begin("Plugin Settings")) {
        ImGui::End();
        return;
    }

    ImGui::Checkbox("Enable Plugin", &isEnabled);
    ImGui::SliderFloat("Update Rate (Hz)", &updateRate, 1.0f, 60.0f);

    if (ImGui::Button("Reset to Defaults")) {
        isEnabled = true;
        updateRate = 30.0f;
    }

    ImGui::End();
}
```

## Best Practices

### 1. Be Specific

**Good:**
```
Create a plugin that hooks Function TAGame.Ball_TA.OnHitGoal and logs the scorer's PRI name
```

**Bad:**
```
Make a goal plugin
```

### 2. Mention Required Features

**Good:**
```
Create a plugin with:
- Event hook for goals
- ImGui window showing stats
- Config file persistence
```

**Bad:**
```
Plugin that does goal stuff
```

### 3. Reference SDK Concepts

**Good:**
```
Use ServerWrapper to get all players and track their CarWrappers
```

**Bad:**
```
Get all the cars
```

## Limitations

### Current Limitations

1. **No actual compilation** - Generated code is not compiled, only syntax-checked
2. **Single plugin only** - Cannot generate multi-plugin projects
3. **No dependency management** - Assumes standard BakkesMod SDK
4. **Limited error handling** - May not generate comprehensive error checks

### Known Issues

1. **Complex state management** - May not handle complex plugin state correctly
2. **Threading** - Does not generate thread-safe code automatically
3. **Memory management** - Basic RAII only, no advanced memory patterns

## Troubleshooting

### "Generated code has syntax errors"

**Solution:** Try rephrasing requirements to be more specific about implementation details.

### "API usage validation failed"

**Solution:** The generated code may not be using BakkesMod APIs correctly. Review and manually fix API calls.

### "No RAG context found"

**Solution:** Ensure RAG index is built (`rag_storage_bakkesmod/` exists). Rebuild if necessary.

## Advanced Usage

### Custom Templates

You can modify `templates/plugin_template.h` and `templates/plugin_template.cpp` to change the base plugin structure.

### Code Validation Rules

Edit `code_validator.py` to add custom validation rules for your team's coding standards.

## Future Features

Planned enhancements:

- [ ] Multi-file plugin generation
- [ ] Test file generation
- [ ] Actual compilation and testing
- [ ] GitHub Actions CI/CD generation
- [ ] Plugin marketplace integration
- [ ] Version migration helpers

## Support

For issues or questions:
- Check generated code carefully before using
- Review BakkesMod SDK documentation
- Test in a safe environment first
