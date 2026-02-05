# BakkesMod SDK Reference

> Complete documentation for the BakkesMod plugin SDK.
> Source: https://bakkesplugins.com/wiki/bakkesmod-sdk
> Generated: 2026-01-25

---

## Table of Contents

1. [Plugin Tutorial](#plugin-tutorial)
2. [Code Snippets](#code-snippets)
3. [ImGui](#imgui)
4. [Useful Info](#useful-info)
5. [Functions](#functions)
6. [Classes](#classes)
7. [Constants](#constants)
8. [Enums](#enums)
9. [Structs](#structs)

---


# Plugin Tutorial

# Plugin Tutorial

Documentation for plugin tutorial.

---

### 1. Setup your installs

Trying to get into making plugins for the first time? There's some super useful tools out there! These steps should get you started on the path towards development.

Note these instructions are designed for **Windows 10 (and presumably 11)**. Rocket League is Windows exclusive, so this is the easiest way to develop for it. If you really want to use linux, it will be harder but may still be possible
1. Make sure you have bakkesmod installed and run it at least once
[https://bakkesmod.com/](https://bakkesmod.com/)
2. Download Visual Studio 2022. Note that this is not the same as Visual Studio Code.
[https://visualstudio.microsoft.com/](https://visualstudio.microsoft.com/)
3. During Visual Studio's install, make sure you select and download `Desktop development with C++`  
To add it to an existing install open Visual Studio, hit `Tools`, then `Get Tools and Features...`  
![uploads/3822272a-4dc8-4d39-aada-4a10e716513d-1_cppsetupstep1.png](https://cdn.bakkesplugins.com/uploads/3822272a-4dc8-4d39-aada-4a10e716513d-1_cppsetupstep1.png)  
Either through the installer or with an existing install you'll eventually get to this menu. Ensure `Desktop development with C++` is checked. It may appear in a different location in different Visual Studio versions  
![uploads/f5b93ad3-988b-42b1-a3ee-437304abadf4-1_cppsetupstep2.png](https://cdn.bakkesplugins.com/uploads/f5b93ad3-988b-42b1-a3ee-437304abadf4-1_cppsetupstep2.png)  
If it's not checked, check it and then hit `Install` in the bottom right  
4. Get Martinn's template set up. [Add the template to your Visual Studio](https://github.com/Martinii89/BakkesmodPluginTemplate) with the instructions in its README
5. Start coding! When you build, your plugin will automatically be moved into your plugins folder, and it will load in Rocket League. If you're having trouble understanding how to start the code, try looking at [Starting A Plugin](/wiki/bakkesmod-sdk/plugin-tutorial/starting-a-plugin) and the code snippets for some examples on how Bakkesmod SDK is used

---
Need help with getting started? Join our programming discord!
[![Discord](https://img.shields.io/discord/862068148328857700?color=5865F2&label=Discord&logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/TgVst5QDru)

---

### 2. Starting a plugin

It will assume you're using the template as linked in [Setting Up](/wiki/bakkesmod-sdk/plugin-tutorial/getting-started). You can of course make a plugin without it, but it gives an easier basis to work off of. You can name it anything, but here we'll be using a demo plugin named CoolPlugin

The template has a lot of commented out code that is designed to help you learn how to use the syntax. We'll uncomment some of this as we go along.

First we'll look at your `CoolPlugin.h` file. It describes any functions your code will use. 
At the top are two lines that are required for plugins, and allow you to call BakkesMod SDK functions
```cpp
#pragma once
#include "bakkesmod/plugin/bakkesmodplugin.h"
```

Next is the declaration of your class. Here you describe any functions you will be using.
`onLoad()` is automatically called by BakkesMod when the plugin is loaded, and `onUnload()` is called when it is unloaded. The unLoad isn't totally necessary, but we'll uncomment it to demonstrate
```cpp
class CoolPlugin: public BakkesMod::Plugin::BakkesModPlugin
{
  void onLoad() override;
  void onUnload() override;
};
```

Now we will look at the `CoolPlugin.cpp` file. Here you write the code that will define the functions  
First it includes your header so it can define the functions described above

```cpp
#include "pch.h"
#include "CoolPlugin.h"
```

Next it declares the plugin. The string in "" will be used in the plugin manager to describe the plugin, but it needs to be a short description. Keep it to 1 or two words. You can also define a plugin version, although the template handles that automatically. Finally is the plugin type. These don't do anything, so just use `PLUGINTYPE_FREEPLAY`. Your plugin will still work in any playlist and mode.

```cpp
BAKKESMOD_PLUGIN(CoolPlugin, "Cool Plugin", plugin_version, PLUGINTYPE_FREEPLAY)
```

Next you define your functions, starting with `onLoad()`. The template already has an onload. As this is a demo, we'll just do a hello world.
```cpp
void CoolPlugin::onLoad() {
  // This line is required for LOG to work and must be before any use of LOG()
  _globalCvarManager = cvarManager;
  // do something when it loads
  LOG("Hello I'm CoolPlugin B)");
}
```

Next is `onUnload()`. Bakkesmod handles most of unloading, so only worry about this if your code is using some 3rd party library that needs unloading to be handled specially. For now this can just add the new function to the bottom of CoolPlugin.cpp and log that it unloaded
```cpp
void CoolPlugin::onUnload() {
  LOG("I was too cool for this world B'(");
}
```

Now you've got a super basic plugin. Hit Build -> Build CoolPlugin or press ctrl + B to finalize the plugin. It will create `CoolPlugin.dll` in a `plugins/` folder next to your plugin source code. It will also move it into your bakkesmod folder if you are using the template.

Now open Rocket League and open the BakkesMod console with f6. Type `plugin load CoolPlugin` and you should see it load and say hello! If you unload it with `plugin unload CoolPlugin` you should see it say goodbye to you 😢

Next we'll make the plugin actually do something. We'll reverse engineer the `ballontop` command, which puts the ball on top of your car in freeplay.
In `CoolPlugin.h` add a new function
```cpp
class CoolPlugin: public BakkesMod::Plugin::BakkesModPlugin
{
  virtual void onLoad();
  virtual void onUnload();
  void ballOnTop();
};
```

We need to define that function in `CoolPlugin.cpp`. Just jump to the end of the file and add
```cpp
void CoolPlugin::ballOnTop() {

}
```
⠀
The next lines of code are all going to be within `CoolPlugin::ballOnTop()` in `CoolPlugin.cpp`

First we need to make sure we should be running the plugin. We only want it to work in freeplay. The `gameWrapper` gives a ton of useful functions to figure out what context the code is being run from. The first line of `CoolPlugin::ballOnTop()` will be
```cpp
if (!gameWrapper->IsInFreeplay()) { return; }
```

There are also `gameWrapper->IsInGame()` and `gameWrapper->IsInOnlineGame()` if you'd rather your plugin run elsewhere

The next line will be getting the `ServerWrapper`.  This is what controls pretty much everything in the current game. You can get players, cars, the ball, the goals, and other things from it so it's incredibly useful. **[We also nullcheck it](/wiki/bakkesmod-sdk/plugin-tutorial/best-practices)**. If you call functions on a null server Rocket League will crash
 ```cpp
ServerWrapper server = gameWrapper->GetCurrentGameState();
if (!server) { return; }
```

Next we get the ball and nullcheck it
```cpp
BallWrapper ball = server.GetBall();
if (!ball) { return; }
```

And we get the car. As this is freeplay we only have one to worry about, but in any mode this will select your car. And we nullcheck it. If this seems redundant, it isn't. What if your car was demolished? Without this nullcheck, you'd crash your game if you ran the command before the respawn.
```cpp
CarWrapper car = gameWrapper->GetLocalCar();
if (!car) { return; }
```

Next we can start manipulating things. We can grab the car's velocity and assign it to the ball so they match. If the ball is going slower or faster than the car, it'll just fly off by itself
```cpp
Vector carVelocity = car.GetVelocity();
ball.SetVelocity(carVelocity);
```

Next we'll actually move the ball. A Vector is a 3 dimensional point in space. The Z axis is up and down in Rocket League, so we can put the ball above the car by using the car's location and adding distance on the Z axis.
```cpp
Vector carLocation = car.GetLocation();
float ballRadius = ball.GetRadius();
ball.SetLocation(carLocation + Vector{0, 0, ballRadius * 2});
```

We've now defined `CoolPlugin::ballOnTop()`
In full:
```cpp
void CoolPlugin::ballOnTop() {
  if (!gameWrapper->IsInFreeplay()) { return; }
  ServerWrapper server = gameWrapper->GetCurrentGameState();
  if (!server) { return; }

  BallWrapper ball = server.GetBall();
  if (!ball) { return; }
  CarWrapper car = gameWrapper->GetLocalCar();
  if (!car) { return; }

  Vector carVelocity = car.GetVelocity();
  ball.SetVelocity(carVelocity);

  Vector carLocation = car.GetLocation();
  float ballRadius = ball.GetRadius();
  ball.SetLocation(carLocation + Vector{0, 0, ballRadius * 2});
}
```

Now we need to call the function

This can be done with a notifier. These can be called with a console command to activate your code. Here we'll define it as `CoolerBallOnTop` because Bakkes' `ballontop` is for the birds.
When it's called, it takes a vector (basically a list) of command arguments that were provided by the command. Here we don't care about those, so they are unused.
The code is called in a lambda `[](){}`. You can just copy the syntax here.
Next is the notifier description, which does nothing, so it'll be `""`.
Finally is the permissions, which decides when the notifier works. It should usually be `PERMISSION_ALL`, as that works anywhere. If you want to restrict the use to specific modes or game states, you can use any of the [permissions listed here](/wiki/bakkesmod-sdk/enums/notifier/permission)
```cpp
cvarManager->registerNotifier("CoolerBallOnTop", [this](std::vector<std::string> args) {
    ballOnTop();
}, "", PERMISSION_ALL);
```

Now we'll put it all together
```cpp
// CoolPlugin.h
#pragma once

#include "bakkesmod/plugin/bakkesmodplugin.h"
#include "bakkesmod/plugin/pluginwindow.h"
#include "bakkesmod/plugin/PluginSettingsWindow.h"

#include "version.h"
constexpr auto plugin_version = stringify(VERSION_MAJOR) "." stringify(VERSION_MINOR) "." stringify(VERSION_PATCH) "." stringify(VERSION_BUILD);


class CoolPlugin: public BakkesMod::Plugin::BakkesModPlugin
  //,public SettingsWindowBase
  //,public PluginWindowBase
{

  //std::shared_ptr<bool> enabled;

  //Boilerplate
  void onLoad() override;
  void onUnload() override;
  void ballOnTop();

public:
  //void RenderSettings() override;
  //void RenderWindow() override;
};

```

```cpp
// CoolPlugin.cpp
#include "pch.h"
#include "CoolPlugin.h"


BAKKESMOD_PLUGIN(CoolPlugin, "Cool Plugin", plugin_version, PLUGINTYPE_FREEPLAY)

std::shared_ptr<CVarManagerWrapper> _globalCvarManager;

void CoolPlugin::onLoad()
{
  // This line is required for LOG to work and must be before any use of LOG()
  _globalCvarManager = cvarManager;
  // do something when it loads
  LOG("Hello I'm CoolPlugin B)");

  cvarManager->registerNotifier("CoolerBallOnTop", [this](std::vector<std::string> args) {
    ballOnTop();
    }, "", PERMISSION_ALL);
}

void CoolPlugin::onUnload() {
  LOG("I was too cool for this world B'(");
}

void CoolPlugin::ballOnTop() {
  if (!gameWrapper->IsInFreeplay()) { return; }
  ServerWrapper server = gameWrapper->GetCurrentGameState();
  if (!server) { return; }

  BallWrapper ball = server.GetBall();
  if (!ball) { return; }
  CarWrapper car = gameWrapper->GetLocalCar();
  if (!car) { return; }

  Vector carVelocity = car.GetVelocity();
  ball.SetVelocity(carVelocity);

  Vector carLocation = car.GetLocation();
  float ballRadius = ball.GetRadius();
  ball.SetLocation(carLocation + Vector{ 0, 0, ballRadius * 2 });
}
```

Finally build the plugin with ctrl + b. Then start freeplay, load the plugin with `plugin load CoolPlugin` and call `CoolerBallOnTop` from the f6 console. The ball should teleport above you!
  
If you want your plugin to load automatically on startup, you can use the plugin manager in game (under the "Plugins" tab in the BakkesMod menu) and toggle the "Loaded" checkbox for your plugin.

You can find the final code here!
[https://github.com/ubelhj/BakkesModStarterPlugin/tree/new-starter-tutorial](https://github.com/ubelhj/BakkesModStarterPlugin/tree/new-starter-tutorial)

Now what if you wanted a user to be able to modify plugin behavior on the fly? [Next are Plugin Variables](/wiki/bakkesmod-sdk/plugin-tutorial/plugin-variables)

---

### 3. Plugin Variables

Next we will make your plugin alterable by a user. They might want to enable/disable the plugin, change how it acts, and pretty much infinite options. I'll be using a fairly simple plugin I've made as an example. I'll only be focusing on the settings, not the actual code that makes it work.
We'll continue the CoolPlugin from [Starting a Plugin](/wiki/bakkesmod-sdk/plugin-tutorial/starting-a-plugin)

We give the users control with the Console Variables, CVars. They allow a user to set a variable's value, and for our code to know when those values are changed. These can be used for string, int, bool, float, or color values. They also can persist across multiple sessions, so user settings are saved. If something being changed to the wrong value can break your plugin, don't use a CVar. Any internal plugin logic that can do unintended things if modified should be stored and controlled with class variables declared in your .cpp or .h files.

To create one, we call `cvarManager->registerCvar()`
There are 2-9 arguments. The first 2 are required, but the others are not
1. The name. This should include your plugin name and make sense
2. The default value as a string. If it's a boolean, remember 0 is false and 1 is true
3. A description of the CVar as a string. The name can describe your variable well, so this isn't super necessary
4. Whether your CVar is searchable. If someone starts typing the name of your CVar, and this is true it suggests the CVar. Defaults to `true`
5. Whether your CVar has a minimum value. Defaults to `false`
6. The minimum value of your CVar if it exists. If the previous is `false` this value can be anything and is ignored
7. Whether your CVar has a maximum value. Defaults to `false`
8. The maximum value of your CVar if it exists. If the previous is `false` this value can be anything and is ignored
9. Whether the CVar's value is stored between sessions. Defaults to `true`

To create an enable/disable CVar for the plugin and a ball distance CVar
```cpp
cvarManager->registerCvar("cool_enabled", "0", "Enable Cool", true, true, 0, true, 1);
cvarManager->registerCvar("cool_distance", "200.0", "Distance to place the ball above");
```

A user can use these by typing into the f6 console `cool_enabled 1` to enable cool, or `cool_distance 300` to move the ball away

Now we've got the CVar. But what do we do with it? There's two good ways to use it. Either we can do something when the value is changed, or we can get the value from the CVar as we use the value

First is `CVarWrapper.addOnValueChanged()`
This creates a function that is called when the value is changed. Quite simple. The string simply is what the old value was. The CVarWrapper holds the new value.
The `[](){}` is called a lambda, and is basically just a helper function. We put `[this]` so we can access our global variables inside the lambda. `(std::string oldValue, CVarWrapper cvar)` are the values passed to the lambda by the `addOnValueChanged` function. Every time the value changes and the lambda is called, the code inside `{}` happens.
The most common use is to change the value of a global variable.
If I had a variable `bool coolEnabled`, I could call `coolEnabled = cvar.getBoolValue();` inside the brackets to change its value
```cpp
cvarManager->registerCvar("cool_enabled", "0", "Enable Cool", true, true, 0, true, 1)
        .addOnValueChanged([this](std::string oldValue, CVarWrapper cvar) {
            coolEnabled = cvar.getBoolValue();
        });
```

The second is by getting the CVar and then its value in another function. We can call
`cvarManager->getCvar()` with the name of any CVar to get and/or set the value of that CVar, even ones that aren't from our plugin.
```cpp
CVarWrapper distanceCVar = cvarManager->getCvar("cool_distance");
if (!distanceCVar) { return; }
float distance = distanceCVar.getFloatValue();
```

We can put together this knowledge to add to `CoolPlugin.cpp`  
[https://github.com/ubelhj/BakkesModStarterPlugin/blob/plugin-variables/CoolPlugin/CoolPlugin.cpp](https://github.com/ubelhj/BakkesModStarterPlugin/blob/plugin-variables/CoolPlugin/CoolPlugin.cpp)

This isn't enough though. Who wants to be changing an integer value with a console command? Gross. What if we had sliders? Checkboxes? Buttons? Next we'll add a simple GUI  
[Plugin Interface](/wiki/bakkesmod-sdk/plugin-tutorial/plugin-interface)

---

### 4. Plugin Interface

This will go over creating a user-friendly interface. There's buttons, sliders, checkboxes, dropdowns, and all sorts of options. It also assumes you're using the template, as the template automatically includes the ImGui GUI library we'll be using. It also adds a file we'll be editing

Yet again we have the `CoolPlugin` from [Plugin Variables](/wiki/bakkesmod-sdk/plugin-tutorial/plugin-variables) and we want to add a button to activate our cooler ball on top, a checkbox to enable cool, and a slider to choose the distance that the ball is placed from your car

First we need to uncomment some code in `CoolPlugin.h`

At the class declaration, uncomment `SettingsWindowBase` and `RenderSettings`
```cpp
class CoolPlugin: public BakkesMod::Plugin::BakkesModPlugin, public SettingsWindowBase

// ...

void RenderSettings() override;
```

Now we can define those 3 functions to create the interface. We will put these in a new file `CoolPluginSettings.cpp` but they can be in any `.cpp` file that includes `CoolPlugin.h`, such as `CoolPlugin.cpp`

To add a file, right click the src (or any other) folder in the solution explorer, then add a new item  
![Adding a new item to the solution](https://cdn.bakkesplugins.com/uploads/a4c5f007-8370-4092-abba-fda6c90604ff-1_solution_explorer_new_item.png)  

Add a .cpp file with the name of your choice.  
![Naming the cpp file](https://cdn.bakkesplugins.com/uploads/5909d644-9027-47a9-a1e6-c3d621ff0102-1_solution_explorer_add_cpp.png)

In the file, make sure to include `pch.h` and `CoolPlugin.h` at the top
```cpp
#include "pch.h"
#include "CoolPlugin.h"
```

Now you can create the interface. We'll start with simple text but this is what we'll be modifying in the rest of this page  
There's a hugely important thing to consider with this function. Never call on or change any Rocket League / Bakkesmod values here. If you do, you will crash. It's happening outside of the game, and cannot safely alter it. That means that if you have a `CVarWrapper.addOnValueChanged()` that alters the state of the game, it will be unsafe to use here  
```cpp
void CoolPlugin::RenderSettings() {
    ImGui::TextUnformatted("A really cool plugin");
}
```

Lets start building the plugin interface. First we'll have a button that'll call `CoolerBallOnTop`. The button as well as most other interactable ImGui components has a boolean property. If it's true, that means it's been interacted with. So when the button has been clicked, we'll use the cvarManager to call `CoolerBallOnTop`. But `CoolerBallOnTop` uses the ServerWrapper and alters the game. It'll crash! We can wrap it inside `gameWrapper->Execute()`. We'll also add hover text because why not
```cpp
if (ImGui::Button("Ball On Top")) {
  gameWrapper->Execute([this](GameWrapper* gw) {
    cvarManager->executeCommand("CoolerBallOnTop");
  });
}
if (ImGui::IsItemHovered()) {
  ImGui::SetTooltip("Activate Ball On Top");
}
```

![The button](https://cdn.bakkesplugins.com/uploads/1e0a423e-c28a-4604-bdc2-a21ef9430fb6-1_button-example.png)

Now let's do a checkbox for `cool_enabled`. First we need to get the CVar, then use it. The `bool enabled` is necessary, as the checkbox uses that to store whether or not the checkbox should be checked. You can't just use the CVar
```cpp
CVarWrapper enableCvar = cvarManager->getCvar("cool_enabled");
if (!enableCvar) { return; }
bool enabled = enableCvar.getBoolValue();
if (ImGui::Checkbox("Enable plugin", &enabled)) {
  enableCvar.setValue(enabled);
}
if (ImGui::IsItemHovered()) {
  ImGui::SetTooltip("Toggle Cool Plugin");
}
```
And finally a slider for the distance CVar.
ImGui elements use `char *` instead of `std::string`
You can easily convert between with `std::string.c_str()` and `std::string newStringVariableName(char *)`
```cpp
CVarWrapper distanceCvar = cvarManager->getCvar("cool_distance");
if (!distanceCvar) { return; }
float distance = distanceCvar.getFloatValue();
if (ImGui::SliderFloat("Distance", &distance, 0.0, 500.0)) {
  distanceCvar.setValue(distance);
}
if (ImGui::IsItemHovered()) {
  std::string hoverText = "distance is " + std::to_string(distance);
  ImGui::SetTooltip(hoverText.c_str());
}
```

![The slider](https://cdn.bakkesplugins.com/uploads/81d8514f-1f8b-4a0d-ba87-cf9a67ef0b24-1_slider-example.png)

We finally have a settings file using all of our CVars. There's a load more things you can do with ImGui, but hopefully this is enough to get the right idea and get started. I hope that by covering these three elements I covered most of what plugins need to use. ImGui is complicated and most plugins don't use it yet. Feel free to ask questions

Here's the final code  
[https://github.com/ubelhj/BakkesModStarterPlugin/blob/plugin-interface/CoolPlugin/CoolPluginSettings.cpp](https://github.com/ubelhj/BakkesModStarterPlugin/blob/plugin-interface/CoolPlugin/CoolPluginSettings.cpp)  
[https://github.com/ubelhj/BakkesModStarterPlugin/tree/plugin-interface/](https://github.com/ubelhj/BakkesModStarterPlugin/tree/plugin-interface/)  

And here's a useful interactable imgui demo  
[https://pthom.github.io/imgui_manual_online/manual/imgui_manual.html](https://pthom.github.io/imgui_manual_online/manual/imgui_manual.html)

---

### 5. Third party dependencies

For all submitted plugins that are using other libraries. We require the use of vcpkg for installing and building these.

## Why should I bother?

When submitting a plugin to bakkesplugins.com for review. The review process involves a human beeing reading every single line of the code you've submitted.
Since we don't accept pinky promises on whether or not there's any 'surprises' in the library code you've attached. Every line of code submitted has to be verified.

In the past this involved scouring the web for the official source and doing a file diff to verify nothing had changed. This was tedious, booring and error prone - Leading to no one wanting to do the job.

To make the review process faster and more managable. We now therefore require all plugin dependencies to be fetched from the dependeny manager vcpkg.

## Get started with vcpk

Follow these steps to install and integrate vcpkg with visual studio.

### Step 1: Install vcpkg

Just follow the install instructions from microsoft <https://vcpkg.io/en/getting-started>

### Step 2: Add visual studio integration

This should have already been explained in the installation guide from step 1.  
Simply run this command in the same folder as where you installed vcpkg `vcpkg integrate install`

### Step 3: Define dependencies

Create a file named `vcpkg.json` in the root of your project with the following content. This will be a the starting point for defining your dependencies.

```js
{
    "$schema": "https://raw.githubusercontent.com/microsoft/vcpkg-tool/main/docs/vcpkg.schema.json",
    "dependencies": []
}
```

You can find the packages available on vcpkg here: <https://vcpkg.io/en/packages.html>

For more advanced scenarios read the documentation for the manifest file here: <https://learn.microsoft.com/en-us/vcpkg/users/manifests>

This is a example for a vcpkg.json file that would pull and build nlohmann, eventpp websocketpp and cpr:

```js
{
    "$schema": "https://raw.githubusercontent.com/microsoft/vcpkg-tool/main/docs/vcpkg.schema.json",
    "dependencies": [
        "nlohmann-json",
        "eventpp",
        "websocketpp",
        "cpr"
    ]
}
```

### Step 4: Configure visual studio

You should now go into your project properties and configure vcpkg. You should now have a new section named `vcpkg` Go here and set `Use vcpkg Manifest` to Yes. Also set `Use Static Libraries` to Yes (see attached screenshot)

![vcpkg-vs](https://cdn.bakkesplugins.com/uploads/77371812-5ec5-4bd0-a0b7-c7dea6cd6548-1_vcpkg-vs.png)

### Step 5: gitignore

optionally you could(you really should..) add `vcpkg_installed/` to your .gitignore file

### Step 6: Test it out

When you build your project in visual studio for the first time, vcpkg will now download and build your dependencies.

#### TLDR summary

1. Install vcpkg
2. Integrate vcpkg with visual studio
3. Add manifest file with dependencies
4. Tell visual studio to use vcpkg manifest and to link statically 
5. Success

Notes:
The Imgui that bm uses is not on vcpkg. It's fine to use the files in the sdk or from the template directly. Same with fmt (but pulling it from vcpkg is preferred)

---

### Best Practices

These practices are super important to follow to make sure you don't accidentally crash the game. 

1. Nullchecking  
Any BakkesMod wrapper is a pointer under the hood, and that pointer could be null! 
If you try and use a null wrapper, you'll crash. 
This is super simple, just make sure you nullcheck every time you obtain a new wrapper.

```cpp
ServerWrapper server = gameWrapper->GetCurrentGameState();

// Just check if the server is null
if (!server) { 
    // The server is null! Maybe you want to log here
    // LOG("Null Server!");

    // Otherwise just return out of the function
    return; 
}

// Now you're safe and the server is valid
```

2. Don't store wrappers  
As the game is played, the pointers underneath wrappers might become invalid or point to unexpected memory locations. 
Imagine you stored a ServerWrapper for use later, but by the time you use it you're in a new match.  
Solve this by obtaining wrappers as you use them.  
Wrappers are safe to pass to functions or lambdas for immediate use, but should never be stored as class variables.  
Also don't pass them to any callbacks or timeouts where the wrappers will be used later in time

---


# Code Snippets

# Code Snippets

Documentation for code snippets.

---

### Code Snippet Example

If you want to add a new snippet to the page, simply make a code snippet using markdown!

Make code blocks using 3 `
```
// Wow look at this code  
int a = 1;  
int b = 1;  
if (a + b == 11) {  
    LOG("Success");  
} else {  
    LOG("Failure");  
}  
```

becomes
```cpp
// Wow look at this code  
int a = 1;  
int b = 1;  
if (a + b == 11) {  
    LOG("Success");  
} else {  
    LOG("Failure");  
}  
```

---

### Canvas

The [CanvasWrapper](/wiki/bakkesmod-sdk/classes/wrappers/canvaswrapper) is a powerful tool that allows you to draw GUI elements on the screen

![hitbox plugin](https://cdn.bakkesplugins.com/uploads/80bca23e-218e-419a-b903-3ce5388f0579-1_hitboxplugin.jpeg)  
For example this is used by the [Hitbox Plugin](https://bakkesplugins.com/plugins/view/19) by mega and HalfwayDead/Rocket Science  
![OBSCounter](https://cdn.bakkesplugins.com/uploads/3d079508-1bb1-4fdf-9414-7525ed7caf23-1_obscounter.png)  
You can also do more basic things, like text in my stat counter plugin [OBSCounter](https://bakkesplugins.com/plugins/view/126)

First in your `.h`, define a function that will take a canvas.  
`void Render(CanvasWrapper canvas);`  
Then you need to register the function in your `.cpp` code. This can be done in OnLoad or any other place where you want to start the rendering loop
```cpp
gameWrapper->RegisterDrawable([this](CanvasWrapper canvas) {
    Render(canvas);
});
```
Now every single frame, the `Render` function is called with a valid canvas to draw on.

In Render you can do a ton of stuff with the canvas. Here's a quick definition of one. Look for plugin homepages on [BakkesPlugins](https://bakkesplugins.com/) for examples and source code
```cpp
// in a .cpp file 
void CoolPlugin::Render(CanvasWrapper canvas) {
    // defines colors in RGBA 0-255
    LinearColor colors;
    colors.R = 255;
    colors.G = 255;
    colors.B = 0;
    colors.A = 255;
    canvas.SetColor(colors);

    // sets position to top left
    // x moves to the right
    // y moves down
    // bottom right would be 1920, 1080 for 1080p monitors
    canvas.SetPosition(Vector2F{ 0.0, 0.0 });

    // says hi
    // draws from the last set position
    // the two floats are text x and y scale
    // the false turns off the drop shadow
    canvas.DrawString("Hi Cool Dude", 2.0, 2.0, false);
}
```
![basic example](https://cdn.bakkesplugins.com/uploads/dc3f71f6-76c9-461d-977c-b0179af58ea7-1_basiccanvas.png)

The result is a bit underwhelming, but there's loads more you can do with this awesome tool.  
You can include [images](/wiki/bakkesmod-sdk/code-snippets/creating-image-wrapper) too  
I highly recommend checking the [Hitbox Plugin Source](https://github.com/Aberinkula/HitboxPlugin) if you want to render things relative to locations on the field  
[CanvasTool Plugin](https://bakkesplugins.com/plugins/view/272) also lets you test out some canvas things easier

---

### Creating an ImageWrapper

## Creating an ImageWrapper
* You should use some kind of reference type that manages the memory. `std::shared_ptr<ImageWrapper>` is my recommendation.
* The [ImageWrapper](/wiki/bakkesmod-sdk/classes/wrappers/imagewrapper) is a resource managing class. This means that you have to keep this object "alive" as long as you want to use the image. When the object goes out of scope \ gets destructed it cleans up the resource it manages.
* Stick to `std::shared_ptr` and you almost can't mess up.
* [See all ImageWrapper functions here](/wiki/bakkesmod-sdk/classes/wrappers/imagewrapper)

0. Add the declaration to your .h
```cpp
 std::shared_ptr<ImageWrapper> myImage;



## Using an ImageWrapper
 1. Use the constructor with the second or third argument true (or both). This can cause stuttering based on the size of the image. This is best done in your plugin's onLoad. You only need to load the image once, and onLoad is expected to be a bit stuttery already
```cpp
// ImageWrapper(std::string path, bool canvasLoad = false, bool ImGuiLoad = false);
myImage = std::make_shared<ImageWrapper>(gameWrapper->GetDataFolder() / "MyPluginFolder" / "MyImage.png", true, true);


2. Load the resource if you didn't use the optional args. This also causes lag so it's best to do this in onLoad or in a place that isn't called often
```cpp
myImage->LoadForCanvas();
myImage->LoadForImGui();
```

3. 
    a. Pass it to the canvas wrapper in a drawable callback after ensuring it's loaded
```cpp
if (myImage->IsLoadedForCanvas()) {
 canvas.DrawTexture(myImage.get(), 1); 
 // there are multiple functions in the canvaswrapper that accept ImageWrapper*
}
```

    b. Use it in your imgui render code. Again checking that it's loaded
```cpp
if (myImage->IsLoadedForImGui() && auto pTex = myImage->GetImguiTex()) {
    auto rect = myImage->GetSizeF();
    ImGui::Image(pTex, { rect.width, rect.height });
}
```


PS:
The backend will load in the resource whenever you try to use the images if it hasn't already been loaded. So you actually don't have to worry about it, but the loading can cause some lag\stutters.
So it's better to do it during loading the plugin -  when such behaviour can be expected.

---

### How CVars Work

Cvars are the global variables of the mod. Bakkesmod has many [pre-defined CVars](https://bakkesmod.fandom.com/wiki/Category:Cvars#List_of_all_cvars) that your plugin can also set/unset.
Cvars are strings, but can represent bool/float values in string form. The declaration of a CVar is:
```cpp
// in include\bakkesmod\wrappers\cvarmanagerwrapper.h

// <summary>Register a CVar.</summary>
// <param name="cvar">The string name of your CVar</param>
// <param name="defaultValue">The default value of your CVar</param>
// <param name="desc">A description of your Cvar which is logged</param>
// <param name="searchAble">If the cvar will pop-up when entering values in the console.</param>
// <param name="hasMin">If there is a minumum value (i.e. a float slider)</param>
// <param name="min">if hasMin=true, this is the value.</param>
// <param name="hasMax">If there is a maximum value</param>
// <param name="max">if hasmax=true, this is the value.</param>
// <param name="saveToCfg">If true, will add the Cvar and default value to config.cfg</param>
// <returns> Vector of std::string keybinds.</returns>
registerCvar(std::string cvar, std::string defaultValue, std::string desc = "", bool searchAble = true, bool hasMin = false, float min = 0, bool hasMax = false, float max = 0, bool saveToCfg = true);
```

Notice the min/max arguments. The BM rebound plugin uses this i.e. "rebound_shotspeed" as a value between 0-2000.

In plugin onLoad(), cvarManager->registerCVar() is called, this is where you tell bakkesmod details about your CVar and give it a default string values.

When bakkesmod loads, it'll execute plugins.cfg which loads each plugin. This is where onLoad() is fired, CVars are created and default values are set for each plugin. Then config.cfg will execute, which will set saved-values.

To save a value, you need to call 'writeconfig' in the console, or programmatically with:

```cpp
cvarManager->executeCommand("writeconfig", false); //false just means don't log the writeconfig
```

The config.cfg file containes all cvars/alias. Find this file in bakkesmod/cfg/config.cfg. This file is your route to easily saving CVars across sessions.
Note that 'writeconfig' will also execute binds.cfg, and save current keybinds.

If you want to reload a plugin while BM is running already but keep its values, make sure you use writeconfig before the plugin is unloaded, then unload/load or reload the plugin, now the plugin will have registered it cvars, but with default values. Then execute config.cfg to get your stored values back. In the console: 'exec config.cfg'.

Here is a typical example changing a CVar with ImGui checkbox:
```cpp
// myplugin.h
class MyPlugin : public BakkesMod::Plugin::BakkesModPlugin, public BakkesMod::Plugin::PluginWindow
{
    std::shared_ptr<bool> enabled;
    virtual void onLoad();
    virtual void onUnload();
    //...
    public:
    //...
    private:
    //...
}

// myplugin.cpp
enabled = std::make_shared<bool>(true);

cvarManager->registerCvar("plugin_enabled", "0", "Enable/Disable the plugin. 0 = false = disabled, 1 = true = enabled")
    .bindTo(enabled); // bind the global shared pointer to this CVar

cvarManager->getCvar("plugin_enabled").addOnValueChanged([this](std::string oldValue, CVarWrapper cvar) {
//addOnValueChanged is a callback, where everything in this block will be called whenever the cvar value is changed. You can call functions, or just log information.
        if (cvar.getStringValue() == "1") {
            LOG("Plugin Enabled");
        };
        if (cvar.getStringValue() == "0") {
            LOG("Plugin Disabled");
        };
    }
);

// mypluginGUI.cpp
static auto pluginCvar = cvarManager->getCvar("plugin_enabled");
auto pluginEnabled = pluginCvar.getBoolValue();

if (ImGui::Checkbox("Enable Plugin", &pluginEnabled)) {
pluginCvar.setValue(pluginEnabled);
cvarManager->executeCommand("writeconfig", false); // save CVar to config.cfg
}
```

After registering the CVar, we bind it to the global bool. Bind to is pretty much a wrapper for:
```cpp
void bindTo(CVar cvar, std::shared_ptr<T> abc)
{
    cvar.addOnValueChanged([this](std::string, CVarWrapper cvar) {
        *abc = cvar.getTValue();
    });
}
```

Here's another way:
```cpp
// myplugin.h:
static constexpr const char\* pluginEnabled = "1"; // 1 is the default value here

// myplugin.cpp
cvarManager->registerCvar(pluginEnabled, "1", "Enable/Disable the plugin. 0 = false = disabled, 1 = true = enabled", true, true, 0, true, 1);
// all default values for registerCvar are used as arguments here, but not needed if they're just defaults
// usage would be the same.
```

In addition to these methods, you can just make functions to call within .addOnValueChanged() to set Cvar values.

---

### Persistent Storage

Use these files in your plugin files to store persistent data with cvars but without risking the cvars being cleared

Usage
```cpp
// preferably in onload but at least before you use it (genious advice)
persistent_storage_ = std::make_shared<PersistentStorage>(this, "Hats", true, true);

//register the cvar like "normal"
auto cvar = persistent_storage_->RegisterPersistentCvar("hats_preset", "", "The selected preset", true);
```

Download these files to use  
[PersistentStorage.h](/files/PersistentStorage.h)  
[PersistentStorage.cpp](/files/PersistentStorage.cpp)

---

### Known Playlist IDs

All (known) playlist IDs, last updated October 2025. Some playlists which the writer wasn't familiar with are listed under internal codenames. 
``` javascript 
-2: Intermission // Used internally to indicate to the server the client is switching playlists.
0: Casual // Generic id to indicate ALL casual playlists, commonly used in API responses related to player skill/mmr. 
1: Duel
2: Doubles
3: Standard
4: Chaos
6: Private Match
7: Season
8: Exhibition
9: Training
10: Duel (Ranked)
11: Doubles (Ranked)
13: Standard (Ranked)
15: Snow Day
16: Rocket Labs
17: Hoops
18: Rumble
19: Workshop
20: Custom Training Editor
21: Custom Training
22: Tournament Match (Custom)
23: Dropshot
24: Local Match
26: External Match (Ranked)
27: Hoops (Ranked)
28: Rumble (Ranked)
29: Dropshot (Ranked)
30: Snow Day (Ranked)
31: Ghost Hunt
32: Beach Ball
33: Spike Rush
34: Tournament Match (Automatic)
35: Rocket Labs
37: Dropshot Rumble
38: Heatseeker
41: Boomer Ball
43: Heatseeker Doubles
44: Winter Breakaway
46: Gridiron
47: Super Cube
48: Tactical Rumble
49: Spring Loaded
50: Speed Demon
52: Gotham City Rumble
54: Knockout
55: confidential_thirdwheel_test
61: Ranked 4v4 Quads
62: MagnusFutball
64: GodBallSpooky
65: GodBallHaunted
66: GodBallRicochet
67: CubicSpooky
68: GForceFrenzy
70: RumShotDoubles
72: Territory
73: OnlineFreeplay
74: TerritoryDoubles
75: GodballTerritory
76: GodballTerritoryDoubles
77: NonStandardSoccar
79: SnowdayTerritory
80: RunItBack
81: CarWars
82: PizzaParty
83: PushThePuck
84: Possession
86: FCShowdown
87: Sacrifice
88: JumpJam
```

Find the ID of the current playlist with this code. [Click here for GameSettingPlaylistWrapper reference](/wiki/bakkesmod-sdk/classes/wrappers/gameevent/gamesettingplaylistwrapper)  
```cpp
ServerWrapper sw = gameWrapper->GetCurrentGameState();
if (!sw) return;
GameSettingPlaylistWrapper playlist = sw.GetPlaylist();
if (!playlist) return;
int playlistID = playlist.GetPlaylistId();
```

---

### Plugin Exclusive Logging

Want to see the console logging from just your plugin? Open a new PowerShell window and use this command

```Shell
Get-Content (-join($env:APPDATA, "\bakkesmod\bakkesmod\bakkesmod.log")) -wait -tail 1 | select-string -pattern "PLUGIN_NAME"
```

---

### Plugin Types

```cpp
BAKKESMOD_PLUGIN(CoolPlugin, "Cool Plugin", plugin_version, PLUGINTYPE_FREEPLAY)
```

These are a relic of an unrealized feature. Most do nothing.

It's easiest to just leave it as `PLUGINTYPE_FREEPLAY` from the template, but you can use any of the non-threaded options interchangeably.  
`PLUGINTYPE_FREEPLAY` through `PLUGINTYPE_REPLAY` plugins all work in any gamemode or playlist

`PLUGINTYPE_THREADED` means that the plugin's `onLoad()` is called in its own thread  
`PLUGINTYPE_THREADEDUNLOAD` means that the plugin's `onUnload()` is called in its own thread  
These are only necessary if you have something very complicated and time consuming in your onload or unload.  
All of the others do nothing.  


```cpp
PLUGINTYPE_FREEPLAY = 0x01, // Plugin works everywhere
PLUGINTYPE_CUSTOM_TRAINING = 0x02, // Plugin works everywhere
PLUGINTYPE_SPECTATOR = 0x04, // Plugin works everywhere
PLUGINTYPE_BOTAI = 0x08, // Plugin works everywhere
PLUGINTYPE_REPLAY = 0x10, // Plugin works everywhere
PLUGINTYPE_THREADED = 0x20, // Plugin works everywhere and onLoad() is called in its own thread
PLUGINTYPE_THREADEDUNLOAD = 0x40 // Plugin works everywhere and onUnload() is called in its own thread
```

---

### Toasts

Toasts are cool little popups that show on the top right of your game

![toast example](https://cdn.bakkesplugins.com/uploads/f9aefafb-c3f9-4b63-bd1a-6f45d276d1e7-1_toast-example.png)

These are useful for quick notifications that you want to disappear quickly

First you need to enable them in f2 - Misc - Enable notifications

![enable notifications](https://cdn.bakkesplugins.com/uploads/ef5ed6f0-a178-48da-93d0-63ca42b41ff2-1_enable-notifications.png)

Watch out though! Every user also has to enable this, so make sure your plugin page or [interface](/wiki/bakkesmod-sdk/wiki/bakkesmod-sdk/plugin-tutorial/plugin-interface) is quite clear that this is necessary

Now to the code

Make a toast pop up with [GameWrapper](/wiki/bakkesmod-sdk/classes/wrappers/gamewrapper)[.Toast()](https://github.com/bakkesmodorg/BakkesModSDK/blob/master/include/bakkesmod/wrappers/GameWrapper.h#L111)

There are 7 parameters. The first two are required. They are
1. The title, which is the word on top
1. The text, which is the smaller text below
1. The texture name which I'll explain in the next paragraph
1. The time to show the toast, which defaults to 3.5 seconds
1. The [ToastType](/wiki/bakkesmod-sdk/enums/toasttype), which chooses the color and defaults to black
1. The width in pixels. The toast automatically resizes to fit your text, so don't worry about these too much
1. The height in pixels

If you want to have a custom image, load it in your onLoad() using [GameWrapper](/wiki/bakkesmod-sdk/classes/wrappers/gamewrapper).LoadToastTexture(). Provide the function with a name for your toast, and the path to the image. Most likely you'll want it to be in the data folder.

Toasts can be one of 4 colors based on the ToastType

![Image of 4 toast colors](https://cdn.bakkesplugins.com/uploads/7adc1198-8947-4d52-8bf6-ea654889245e-1_toast-colors.png)

I've made an example using CoolPlugin. I put this code in the onLoad() in CoolPlugin.cpp and cool.png in the `bakkesmod/data` folder
```cpp
gameWrapper->LoadToastTexture("cool", gameWrapper->GetDataFolder() / "cool.png");

cvarManager->registerNotifier("cool_toast", [this](std::vector<std::string> args) {
    gameWrapper->Toast("Whoa you're cool", "Super cool", "cool", 5.0, ToastType_Warning);
}, "", PERMISSION_ALL);
```

The result when I call `cool_toast` from the f6 console

![toast in action](https://cdn.bakkesplugins.com/uploads/0e6141cb-a393-4441-a81f-c1c9be31ebb9-1_toast-in-action.png)

For perspective, it's quite small but still legible

![image of toast zoomed in](https://cdn.bakkesplugins.com/uploads/f9aefafb-c3f9-4b63-bd1a-6f45d276d1e7-1_toast-example.png)

Full code here
[https://github.com/ubelhj/BakkesModStarterPlugin/tree/toasts](https://github.com/ubelhj/BakkesModStarterPlugin/tree/toasts)

---

### Using HttpWrapper

## [HttpWrapper](/wiki/bakkesmod-sdk/classes/wrappers/http/httpwrapper)

## Sending a simple http request with a string body
```cpp
CurlRequest req;
req.url = "https://httpbin.org/anything";
req.body = "testing with body";

LOG("sending body request");
HttpWrapper::SendCurlRequest(req, [this](int code, std::string result)
{
    LOG("Body result{}", result);
});
```
## Perform an HTTP/S JSON request
```cpp
CurlRequest req;
req.url = "https://httpbin.org/anything";
req.body = R"T({
    "userId": 1,
    "id": 1,
    "title": "delectus aut autem",
    "completed": false
})T";

HttpWrapper::SendCurlJsonRequest(req, [this](int code, std::string result)
{
    LOG("Json result{}", result);
});
```
## Sending a file over HTTP/S
```cpp
{
CurlRequest req;
req.url = "https://httpbin.org/anything";
req.body = "testing body with file output";

LOG("sending file output body request");
HttpWrapper::SendCurlRequest(req, L"file_output_test.txt", [this](int code, std::wstring result)
{
    LOG("sending file output body request: returned");
});
```
## Downloading a file with download progress callbacks and reading the raw downloaded data
```cpp
CurlRequest req;
req.url = "url";
req.progress_function = [](double file_size, double downloaded, ...)
{
    if (file_size != 0)
        LOG("Download progress {:.2f}%", downloaded / file_size * 100);
};

HttpWrapper::SendCurlRequest(req, [this](int code, char* data, size_t size)
{
    std::ofstream out_file {"test_image_raw.jpeg", std::ios_base::binary};
    if (out_file)
        out_file.write(data, size);
});
```
## Downloading a file to the filesystem
```cpp
CurlRequest req;
req.url = "url";

LOG("sending raw output image download request");
HttpWrapper::SendCurlRequest(req, L"test_image.jpeg", [this](int code, std::wstring out_path)
{
    LOG("file image download code: {}", code);
});
```

## Sending a request with FormData
```cpp
CurlRequest req;
req.url = "url";
req.form_data.push_back(FormField{FormField::Type::kString, "some data", "test_field"});
req.form_data.push_back(FormField{FormField::Type::kFile, "test_file.txt", "test_file"});

LOG("sending post request with fields (both file and string)");
HttpWrapper::SendCurlRequest(req, [this](int code, std::string res)
{
    LOG("Post result: {}", res);
});
```

---


# ImGui

# ImGui

Documentation for imgui.

---

### Color Picker

You've got a super cool looking overlay, but you want the user to change the color. Here's some code for a super nice color picker!

First, you need to be rendering something on the canvas. Let's say it's a single chunk of text. [Like this example](/wiki/bakkesmod-sdk/code-snippets/canvas).

Next, make a cvar for color. This is really simple. The default value can be encoded with hex
```cpp
cvarManager->registerCvar("cool_color", "#FFFFFF", "color of overlay");
```
Now in the canvas render, use the new color
```cpp
CVarWrapper textColorVar = cvarManager->getCvar("cool_color");
if (!textColorVar) {
    return;
}
LinearColor textColor = textColorVar.getColorValue();
canvas.SetColor(textColor);
// render text after setting the color
```

This code goes wherever you're rendering your ImGui
```cpp
CVarWrapper textColorVar = cvarManager->getCvar("cool_color");
if (!textColorVar) { return; }
// converts from 0-255 color to 0.0-1.0 color
LinearColor textColor = textColorVar.getColorValue() / 255;
if (ImGui::ColorEdit4("Text Color", &textColor.R)) {
    textColorVar.setValue(textColor * 255);
}
```

Here's the final result. First when you hover the ColorButton  
![colorpickerhover](https://cdn.bakkesplugins.com/uploads/91bc227d-de08-4acd-9f21-f12c81f80065-1_colorpickerhover.png)  
And here's when the ColorPicker popup is opened by clicking the ColorButton  
![colorpickeropen](https://cdn.bakkesplugins.com/uploads/cf7a7b79-af65-4927-bfa8-01914c808cd7-1_colorpickeropen.png)  

There's multiple flags to change the ColorEdit. If you want a color wheel, or alpha, or just the color button, there's many `ImGuiColorEditFlags` found on line 1150 of `IMGUI/imgui.h`  
For example, I like this combination
```cpp
ImGui::ColorEdit4("Text Color With Flags", &textColor.R, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar)
```
With these flags it looks like this  
![colorpickerflags](https://cdn.bakkesplugins.com/uploads/baf474fd-cf78-49b5-8aa4-0b08408f6606-1_colorpickerflags.png)

[Click here to see exactly how this code is used in CoolPlugin's source](https://github.com/ubelhj/BakkesModStarterPlugin/tree/imgui)

---

### Custom Fonts

Want to add a custom fancy font to your plugin? It's really easy! You can use any .ttf font, although many have copyrights. Check out [https://fonts.google.com/](https://fonts.google.com/) for a bunch of free fonts to use.

First, make a pointer to save your font. Do this in your .h as a variable
```cpp
ImFont* myFont;
```

Next load the font using the GUIManager. This can be done pretty much anywhere, but I recommend using your OnLoad or SetImGuiContext functions. OnLoad happens at startup of the plugin, and SetImGuiContext happens when the plugin's ImGui is initialized. Either will hopefully ensure your font is loaded before you use it (barring any errors)
```cpp
auto gui = gameWrapper->GetGUIManager();

// This syntax requires c++17
// Pick any name you want for the font here
auto [res, font] = gui.LoadFont("font.tff 40px", "font.ttf", 40);

if (res == 0) {
    LOG("Failed to load the font!");
} else if (res == 1) {
    LOG("The font will be loaded");
} else if (res == 2 && font) {
    myFont = font;
}
```

Finally use the font in your ImGui code. Remember that when you push a font, you have to pop it when you're done. 
```cpp
// First ensure the font is actually loaded
if (!myFont) {
    auto gui = gameWrapper->GetGUIManager();
    myFont = gui.GetFont("font.tff 40px");
}

// Now use the font
if (myFont) {
    ImGui::PushFont(myFont);
    ImGui::Text("This is using a custom font");
    ImGui::PopFont();
} else {
    ImGui::Text("The custom font haven't been loaded yet");
}
```

Here's an example using the open source Ubuntu font, as compared to the ImGui default
![An example of custom font](https://cdn.bakkesplugins.com/uploads/104fcbcc-4617-49ed-970c-418b84425ffb-1_font-comparison.png)

The code for this is taken from [Martinn's awesome custom font example](https://github.com/Martinii89/BakkesmodPluginCustomFontExamle). I just made this page to get more visibility for it.

---

### Draggable location

Imagine you have a plugin that renders something on the screen. Users might want to move it to a different location. Dragging's a great way to make it move, and here's how to get it to work! 

First, you need to be rendering something on the canvas. Let's say it's a single chunk of text, with its location defined at the top left. [Like this example](/wiki/bakkesmod-sdk/code-snippets/canvas).

Next we'll add a simple level of customizability with location cvars. X is the left to right axis, and Y is top to bottom. (0, 0) is the top left of the screen
```cpp
cvarManager->registerCvar("cool_x_location", "0", "set x location of the overlay");
cvarManager->registerCvar("cool_y_location", "0", "set y location of the overlay");
```
Now in the canvas render, just use those new locations
```cpp
CVarWrapper xLocCvar = cvarManager->getCvar("cool_x_location");
if (!xLocCvar) { return; }
float xLoc = xLocCvar.getFloatValue();

CVarWrapper yLocCvar = cvarManager->getCvar("cool_y_location");
if (!yLocCvar) { return; }
float yLoc = yLocCvar.getFloatValue();

canvas.SetPosition(Vector2F{ xLoc, yLoc });
// now render everything else
```

Now the user can change the location of the overlay in the console, but now you can add it to the GUI too. In your `RenderSettings()` add these lines
```cpp
CVarWrapper xLocCvar = cvarManager->getCvar("cool_x_location");
if (!xLocCvar) { return; }
float xLoc = xLocCvar.getFloatValue();
if (ImGui::SliderFloat("X Location", &xLoc, 0.0, 5000.0)) {
    xLocCvar.setValue(xLoc);
}
CVarWrapper yLocCvar = cvarManager->getCvar("cool_y_location");
if (!yLocCvar) { return; }
float yLoc = yLocCvar.getFloatValue();
if (ImGui::SliderFloat("Y Location", &yLoc, 0.0, 5000.0)) {
    yLocCvar.setValue(yLoc);
}
```

Now you can move the x and y locations with sliders, and finally we can add dragging.  
First add a checkbox to enable the dragging mode. Without that check there's some weird interactions on when dragging will happen.  
Then the code is fairly simple. Put it where you're doing your ImGui rendering. It sets the cvar values to the x and y locations of the mouse if left click is held. 
```cpp
// inDragMode is just a bool declared at the top of the .cpp file
ImGui::Checkbox("Drag Mode", &inDragMode);

if (inDragMode) {
    if (ImGui::IsAnyWindowHovered() || ImGui::IsAnyItemHovered()) {
        // doesn't do anything if any ImGui is hovered over
        return;
    }
    // drag cursor w/ arrows to N, E, S, W
    ImGui::SetMouseCursor(2);
    if (ImGui::IsMouseDown(0)) {
        // if holding left click, move
        // sets location to current mouse position
        ImVec2 mousePos = ImGui::GetMousePos();
        xLocCvar.setValue(mousePos.x);
        yLocCvar.setValue(mousePos.y);
    }
}
```
This video shows the code in action! 
<video controls="controls" width="1000" preload="metadata">
    <source src="/video/imguidrag.mp4" 
            type="video/mp4" />
</video>

[Click here to see exactly how this code is used in CoolPlugin's source](https://github.com/ubelhj/BakkesModStarterPlugin/tree/imgui)

---

### Overlays

Want the powerful flexibility of ImGui for a simple text overlay? Here's how. 

First make sure you've uncommented all of the plugin window code in the template's .h file. Then in your GUI file go to the render() function. For a custom overlay, you'll need to prepare flags for it to appear the way you want it to.  
These are found at [line 744 of imgui.h](https://github.com/Martinii89/BakkesmodPluginTemplate/blob/master/IMGUI/imgui.h#L744)

Some useful flags for an overlay are:  
`ImGuiWindowFlags_NoTitleBar` removes the title bar  
`ImGuiWindowFlags_NoScrollbar` disables a scroll bar from appearing  
`ImGuiWindowFlags_AlwaysAutoResize` automatically resizes the window to fit the content  
`ImGuiWindowFlags_NoDecoration` removes the scrollbar, title bar, and disables manual resizing  
`ImGuiWindowFlags_NoFocusOnAppearing` prevents the overlay from stealing focus from other ImGui windows  
`ImGuiWindowFlags_NoInputs` makes the mouse pass right through the window to RL  
`ImGuiWindowFlags_NoBackground` disables the background of the window  

Here's an example of starting a window with some flags
```cpp
ImGuiWindowFlags WindowFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize 
    | ImGuiWindowFlags_NoFocusOnAppearing;
// If moveOverlay is true, the user can move the overlay 
//  When false the overlay window no longer accepts input
// I find this useful to connect to a CVar and checkbox in the plugin's settings
//  You can allow a user to move the overlay only when they want
if (!moveOverlay) {
    WindowFlags |= ImGuiWindowFlags_NoInputs;
}

// creates the window with the given flags
if (!ImGui::Begin(menuTitle_.c_str(), &isWindowOpen_, WindowFlags))
{
    ImGui::End();
    return;
}

// Do your overlay rendering with full ImGui here!
```

If you don't want the overlay to close when a user presses esc, change this function to return false 
```cpp
bool PluginName::IsActiveOverlay() {
    return false;
}
```

---

### ImGui Basics

This section is dedicated to ImGui, a library used for creating a user-friendly interface. There's buttons, sliders, checkboxes, dropdowns, and all sorts of options. It also assumes you're using the template, as the template automatically includes the ImGui library. 

And here's a useful interactable imgui demo to see what these look like in action  
[https://pthom.github.io/imgui_manual_online/manual/imgui_manual.html](https://pthom.github.io/imgui_manual_online/manual/imgui_manual.html)

---


# Useful Info

# Useful Info

Documentation for useful info.

---

### Scripts for BakkesPlugins

### Note:
These tools are not moderated by BakkesPlugins staff, so use at your own risk. That said, we checked the source(s) <u>at the time of writing</u> and determined that it is safe to run on your computer

## Bakkesmod Scripts
Creating the uploads for Bakkesplugins can be annoying. Discord member @joanbga created some public scripts that can make this process much easier! 
View and download them in their repo [found here on GitHub](https://github.com/joanbga/BakkesmodScripts)! 
The README in the repo will always be the most up to date on tooling

## [CreateSourceZipScript](https://github.com/joanbga/BakkesmodScripts/blob/main/CreateSourceZipScript)
A PowerShell script that automatically creates a compliant ZIP package for BakkesMod plugin submissions to Bakkesplugins. The script handles different project structures, intelligently selects necessary files, and packages everything according to BakkesMod requirements.

Key Features:
- Automatic detection of Visual Studio project structures
- Smart file selection and exclusion
- Preservation of your project organization in the output
- Support for all standard BakkesMod plugin requirements

## [CreateInstallZipScript](https://github.com/joanbga/BakkesmodScripts/blob/main/CreateInstallZipScript)
This script is helpful for distributing test versions of the plugin to friends. Built binaries should not be submitted to BakkesPlugins because they need the source code to verify
safety

A PowerShell script that generates installation ZIP packages for end users of your BakkesMod plugins. The script extracts version information from your code and packages only the files needed for installation.

Key Features:
- Automatic version detection from version.h
- Creates end-user friendly installation packages
- Includes only required files (DLL, settings, etc.)
- Versioned output filenames for easier distribution

---

### Crash Dumps

Modding RL is finnicky. You're going to crash the game at some point. This will tell you how to learn from a crash and solve your issue.

Crashes are often logged in .dmp files. They are found at
`My Documents/My Games/rocketleague/TAGame/Logs`

Unfortunately sometimes there is no .dmp file. That often means you crashed in a non-game thread, like rendering or websockets. For those you might just need to do some logging to find out where you crashed

![image of files](https://cdn.bakkesplugins.com/uploads/6a77f786-2d7e-4c3c-8bc2-5f955f6a3993-1_crash-dump-file-naming.png)

Make sure you have the exact version of the source code that crashed the game and the .pdb of that exact build loaded in a visual studio project. For this I highly recommend using git. Even for private projects, this can save you when you have to look back at old versions of code. Make a commit for the source code that you've uploaded to bakkesplugins.

Drag the .dmp over the visual studio window

You'll get a Minidump File Summary, which explains in super basic terms what the issue was. It won't be specific though, so hit the green "Debug with Native Only" button

![Debug with Native Only](https://cdn.bakkesplugins.com/uploads/0580ee1f-f214-4fc2-96b7-f7f575b6a2b6-1_debug-with-native.png)

Next it will try to load symbols. If this is taking more than a couple seconds, it's loading too many symbols. Cancel the loading and go to Debug -> Options -> Symbols

![uploads/ddfe1eaf-1a27-4d1d-bf48-80e9be4687e5-1_vs-options.png](https://cdn.bakkesplugins.com/uploads/ddfe1eaf-1a27-4d1d-bf48-80e9be4687e5-1_vs-options.png)

Make sure that Microsoft Symbol Servers is the only symbol location selected and that Load only specified modules is selected

![uploads/9328b213-dcaf-4e09-a777-9a2aa99c604d-1_vs-load-specified-modules.png](https://cdn.bakkesplugins.com/uploads/9328b213-dcaf-4e09-a777-9a2aa99c604d-1_vs-load-specified-modules.png)

Then try to debug with native again. Be aware that these settings mysteriously can reset themselves (they did so in the time it took me to write this up) so you might have to refer back here fairly often.

![uploads/f7c74e47-eea6-4d27-984d-9d3688b764c5-1_null-pointer-debug-example.png](https://cdn.bakkesplugins.com/uploads/f7c74e47-eea6-4d27-984d-9d3688b764c5-1_null-pointer-debug-example.png)

If Visual Studio can find the issue, it will then tell you the issue and automatically jump to it. I dereferenced a null pointer! Oops

Sometimes it's a little less easy, and the issue happens in pluginsdk.dll or RocketLeague.exe. In this case Visual Studio will try its best to find it but may fail

![uploads/f7174ea6-4628-421d-b30f-2cbe6ec2b922-1_null-wrapper-debug-example.png](https://cdn.bakkesplugins.com/uploads/f7174ea6-4628-421d-b30f-2cbe6ec2b922-1_null-wrapper-debug-example.png)

This crash happened because of a null ball! [Remember the best practices!](/wiki/bakkesmod-sdk/plugin-tutorial/best-practices) Whenever you use any type that ends with `Wrapper` like this `BallWrapper` you must nullcheck it before using it. These 4 lines of code will save your plugin from crashing

```cpp
BallWrapper* ball = (BallWrapper*) nullptr;

if (!ball) {
  LOG("null ball");
  // I am saved from crashing!
  return;
}

// guaranteed to only happen on valid balls and not crash
ball->SetLocation(Vector(0, 0, 0));
```

Once you've found the issue, hit the red square up top

![uploads/d00130ad-e3be-49e6-be43-b033932b4909-1_stop-debugging.png](https://cdn.bakkesplugins.com/uploads/d00130ad-e3be-49e6-be43-b033932b4909-1_stop-debugging.png)

Hopefully you've found your issue. If not, try adding many `LOG("");` in places where you're suspicious a crash may have happened. If a log doesn't happen that you expected to, you likely crashed before that statement. Debugging can be tedious, but hopefully the .dmp files can help

---

### GitHub Action Build

GitHub Actions provide a powerful way to automatically build and test your BakkesMod plugins whenever you push code to your repository. This guide will walk you through setting up a GitHub Action workflow that can build your plugin automatically.

## Why Use GitHub Actions?

- **Automated Building**: Automatically build your plugin on every commit or pull request
- **Continuous Integration**: Catch build errors early before they reach users


## Setting Up Your Repository

### 1. Add GitHub Action Workflow

Create a new file in your repository at `.github/workflows/build.yml`:

```yaml
name: Build BakkesMod Plugin

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  build:
    runs-on: windows-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4


    - name: Checkout BakkesMod SDK
      uses: actions/checkout@v4
      with:
        repository: bakkesmodorg/BakkesModSDK
        path: bakkesmod\bakkesmodsdk


    - name: Check for vcpkg manifest
      id: vcpkg_check
      shell: pwsh
      run: |
        $manifest = Test-Path "$env:GITHUB_WORKSPACE/vcpkg.json"
        Write-Host "vcpkg manifest present: $manifest"
        echo "vcpkg_manifest=$manifest" | Out-File -FilePath $env:GITHUB_OUTPUT -Append

    - name: Checkout vcpkg
      if: steps.vcpkg_check.outputs.vcpkg_manifest == 'True'
      uses: actions/checkout@v4
      with:
        repository: microsoft/vcpkg
        path: vcpkg

    - name: Bootstrap vcpkg
      if: steps.vcpkg_check.outputs.vcpkg_manifest == 'True'
      run: |
        .\vcpkg\bootstrap-vcpkg.bat

    - name: Integrate vcpkg
      if: steps.vcpkg_check.outputs.vcpkg_manifest == 'True'
      run: |
        .\vcpkg\vcpkg.exe integrate install


    - name: Setup MSBuild
      uses: microsoft/setup-msbuild@v2

    - name: Find solution file
      id: find_sln
      shell: pwsh
      run: |
        $sln = Get-ChildItem -Path $env:GITHUB_WORKSPACE -Filter *.sln -Recurse | Select-Object -First 1
        if ($null -eq $sln) { Write-Error 'No .sln file found!'; exit 1 }
        Write-Host "sln path: $($sln.FullName)"
        echo "SOLUTION_PATH=$($sln.FullName)" | Out-File -FilePath $env:GITHUB_ENV -Append

    - name: Build
      run: msbuild /m /p:Configuration=Release /p:BakkesModPath=$env:GITHUB_WORKSPACE\bakkesmod /p:PostBuildEventUseInBuild=false $env:SOLUTION_PATH


    - name: Prepare build artifacts
      run: |
        # Create artifacts/plugins directory
        New-Item -ItemType Directory -Force -Path "artifacts/plugins"
        
        if (Test-Path "plugins") {
          Get-ChildItem -Path "plugins" -Include "*.dll", "*.pdb" -Recurse | ForEach-Object {
            Copy-Item $_.FullName -Destination "artifacts/plugins" -Force
          }
        }
        
        if (Test-Path "data") {
          Copy-Item -Path "data" -Destination "artifacts/data" -Recurse -Force
        }
        
        Write-Host "Artifacts contents:"
        Get-ChildItem -Path "artifacts" -Recurse

    - name: Create release archive
      run: |
        # Create zip file with timestamp
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $zipName = "${{ github.event.repository.name }}-$timestamp.zip"
        
        if (Test-Path "artifacts") {
          Compress-Archive -Path "artifacts\*" -DestinationPath $zipName
          echo "RELEASE_ZIP=$zipName" >> $env:GITHUB_ENV
        } else {
          Write-Error "No artifacts found to archive"
          exit 1
        }

    - name: Upload build artifacts
      uses: actions/upload-artifact@v4
      with:
        name: ${{ github.event.repository.name }}
        path: ${{ env.RELEASE_ZIP }}
        retention-days: 30
```


### 2. Commit and Push

Once you've created the workflow file:

```bash
git add .github/workflows/build.yml
git commit -m "Add GitHub Actions build workflow"
git push
```

---

### Live Debugging

You can live debug your code by attaching Rocket League to visual studio and stepping through it  
This page explains how  
https://github.com/MicrosoftDocs/visualstudio-docs/blob/main/docs/debugger/attach-to-running-processes-with-the-visual-studio-debugger.md

For rocket league, you need to attach to the process "RocketLeague.exe" (see capture, sorry for the french UI ^^)
![uploads/d1a9d5cd-7367-4613-99fa-6af5d6ac6ad2-1_debugger-attach-process.png](https://cdn.bakkesplugins.com/uploads/d1a9d5cd-7367-4613-99fa-6af5d6ac6ad2-1_debugger-attach-process.png)

If you're in an infinte loop (which never happens to me 😏), you can press the "interrupt" button, and Visual Studio will jump to the part of the code that is currently being executed  
![uploads/ebfec21c-93ad-4060-b6e9-e8c7daaaeeda-1_debugger-pause.png](https://cdn.bakkesplugins.com/uploads/ebfec21c-93ad-4060-b6e9-e8c7daaaeeda-1_debugger-pause.png)

If you have a critical crash (if it's your code) Visual will jump to the line that raised the exception

Finally, you can put breakpoints by clicking the column before a line, and Visual will stop here and let you inspect the different values in your code.  
![uploads/e2ae09ad-2351-4d8c-b370-efc7f689d9b8-1_debugger-inspect.png](https://cdn.bakkesplugins.com/uploads/e2ae09ad-2351-4d8c-b370-efc7f689d9b8-1_debugger-inspect.png)

---

### Logging

Hitting f6 in Rocket League with BakkesMod running opens the console. You can log to this with a helpful template function, `LOG()`.

LOG uses a really smart formatting library to make your life easier called [fmt](https://fmt.dev/latest/index.html)

Here's an example. Simply put, it replaces any `{}` in the string with whatever arguments you put after
```cpp
float a = 10.10;
int b = 11;
std::string c = "12";
bool d = false;

LOG("a = {}, b = {}, c = {}, d = {}", a, b, c, d);
// outputs [class OBSCounter] a = 10.1, b = 11, c = 12, d = false
```

There's one tiny caveat. Make sure that you only use LOG after the line `_globalCvarManager = cvarManager;` is run in onLoad(), as it depends on `_globalCvarManager`. You can use it immediately after that line. 

[Check out fmt docs here for more features](https://fmt.dev/latest/index.html)

---

### MMR

These code snippets will help you track a player's MMR. BakkesMod only gets player MMR in matches you're in, so it's best used for the primary player or any players in your party. You can't get any arbitrary player's MMR. 

In yourplugin.h, create a smart pointer for the token. Even if you don't use the token, BakkesMod uses it to ensure that MMR is still being tracked. If you don't declare it using a C++ pointer, you might not get any MMR updates. 
```cpp
std::unique_ptr<MMRNotifierToken> notifierToken;
```

When you want to start tracking MMR, likely in onLoad(), register the notifier. Each time BakkesMod gets any player's MMR, it will do whatever is inside the {}. For best style, it calls a function written elsewhere. 
```cpp
notifierToken = gameWrapper->GetMMRWrapper().RegisterMMRNotifier(
    [this](UniqueIDWrapper id) {
        updateStats(id);
    }
);
```

Here's an example callback function called updateStats. You can get any playlist's MMR for that player now, see [playlist IDs here](/wiki/bakkesmod-sdk/code-snippets/playlist-id)
```cpp
void PluginName::updateStats(UniqueIDWrapper id)
{
  float mmr = gameWrapper->GetMMRWrapper().GetPlayerMMR(id, playlistID);
  LOG("{} MMR is: {}", id.GetIdString(), mmr);
}
```

To ensure the player is someone you care about, you can compare their ID to the primary or any other player's PRI
```cpp
PlayerControllerWrapper pc = gameWrapper->GetPlayerController();
if (!pc) { return; }
PriWrapper pri = pc.GetPRI();
if (!pri) { return; }
UniqueIDWrapper primaryID = pri.GetUniqueIdWrapper();

if (id == primaryID) {
    // This is Player 1's new MMR!
}
```

---

### Multiplayer Plugins

Netcode plugins are a powerful way to handle asymmetrical multiplayer plugins. It can allow clients to control the match, when normally only the host can. This is much better described in this video by CinderBlock

[https://youtu.be/jyFuN8k2uIs](https://youtu.be/jyFuN8k2uIs)

If you want to make your own netcode plugin, there's a template for it just like normal plugins. It does all the setup shown in the above video automatically. Follow the README to install it 

[https://github.com/ubelhj/BakkesmodNetcodePluginTemplate](https://github.com/ubelhj/BakkesmodNetcodePluginTemplate)

---

### Other Languages

### [ARCHIVED]
###### This page is archived and kept around to showcase what has existed in the past. As of writing, the only accepted way to submit plugins to BakkesPlugins is with the plugin template from the tutorial section. Of course, no one can stop you from writing plugins in any way you desire but there will be no one around to help with getting stuff working. Moreover, distribution submissions won't be accepted

---


There's a couple super useful libraries that let you write plugins in other programming languages. We can't really guarantee any tutorials in this server as we don't have anyone with much experience in them. If you use them feel free to add help and make a pull request. You're still welcome to ask for help with them in the [discord](https://discord.gg/HMptXSzCvU), but most people here use the standard C++ library and it may be hard to get specific help

## Python by Stanbroek  
![BakkesMod Python Logo](https://cdn.bakkesplugins.com/uploads/9c0b1bf3-4309-43fa-a2f5-46a16f2f6fc6-1_bmpy.png) 
[https://github.com/Stanbroek/BakkesModSDK-Python](https://github.com/Stanbroek/BakkesModSDK-Python)


##  Rust by Arator  
![BakkesMod Rust Logo](https://cdn.bakkesplugins.com/uploads/f8cdcda6-d831-4ae7-aec3-79ac1f92e4f7-1_bmrust.png)
[https://docs.rs/bakkesmod/0.2.1/bakkesmod/](https://docs.rs/bakkesmod/0.2.1/bakkesmod/)

---

### Submitting to Bakkesplugins

You've looked through the docs in desperation, angrily screamed at your computer, crashed a couple times, but it's all worth it! You have made a super cool plugin that you want to share! Nice job!

## Account Setup

Plugins are officially hosted at [https://bakkesplugins.com/](https://bakkesplugins.com/) (here!!!). Register if you have to. Once you're logged in, you can create and manage your plugins through the dashboard.

## Creating Your Plugin

Navigate to your account dashboard and click "Create New Plugin". You'll need to provide:

### Required Information

1. **Plugin Name** - At least 5 characters
2. **Short Description** - At least 5 characters, one-sentence summary
3. **Description** - At least 30 characters, detailed explanation of what your plugin does (supports Markdown formatting)
4. **Source Code ZIP** - Your plugin source code packaged as a .zip file (max 20MB)
5. **Version Number** - Semantic version format (e.g., 1.0.0, 1.2.3)

### Optional Information

- **Homepage URL** - Link to your plugin's homepage or documentation
- **Source Code URL** - Link to your plugin's repository (GitHub, GitLab, etc.)
- **Banner Image** - PNG or JPEG image, max 1MB wider than tall (16:9 ratio recommended)
- **Tags** - Select relevant tags to help users find your plugin
- **Visibility Settings**:
  - **Publicly Visible** - Whether anyone can view your plugin
  - **Listed** - Whether your plugin appears in browse pages and search results

## Source Code Requirements

### What to Include in Your ZIP

Your source code ZIP should contain:

- The `.sln` (solution) file
- The `.vcxproj` (project) file  
- All `.cpp`, `.h`, and `.hpp` source files
- Any additional data files (`.set` files, images, etc.)

**Important:** Use the [BakkesMod Plugin Template](https://github.com/Martinii89/BakkesmodPluginTemplate) to ensure your plugin builds smoothly for moderators.

### File Structure Examples

**If your .sln is outside the solution folder**, include:
- The `.sln` file
- The entire plugin folder containing `.vcxproj` and source files

**If your .sln is inside the solution folder**, include:
- The entire folder with both `.sln` and `.vcxproj`
- All source files

### Third-Party Dependencies

If your plugin uses third-party dependencies, please see the [Plugin Dependencies Guide](/wiki/bakkesmod-sdk/plugin-tutorial/3rdparty-dependencies).

If your plugin requires external .dll files or has special build steps:
- Make this clear in your source code URL description or in a README file
- External files must be from safe, accessible sources
- Example: Nvidia Highlights requires a pre-built .dll from Nvidia, which moderators could safely obtain

## Review Process

After you submit your plugin:

1. **Submission Received** - Your plugin enters the review queue with status "Pending Review"
2. **Automated Build** - The system will queue your plugin for automated building (status: "In Build Queue" → "Building")
3. **Moderator Review** - Moderators manually review the code of all plugins
4. **Decision** - You'll be notified via email and on-site notification when your plugin is:
   - **Approved** - Your plugin is now live and available for download
   - **Rejected** - You'll receive specific feedback on why and can resubmit

### Review Timeline

- Reviews are handled by volunteer moderators
- Timeline varies but generally is less than a week
- Please be patient and respectful of volunteers' time

### Tracking Your Submission

You can view the status of your submissions in your dashboard under the "Submissions" tab of your plugin. Statuses include:

- **Pending Review** - Waiting for moderator review
- **In Build Queue** - Queued for automated compilation
- **Building** - Currently being compiled
- **Approved** - Approved and published
- **Rejected** - Not approved (with reason provided)

## Updating Your Plugin

To release a new version:

1. Go to your plugin management page
2. Navigate to the "Submissions" tab
3. Upload a new source ZIP with an updated version number
4. Your new version will enter the review queue

You can have multiple submissions, but only one can be active at a time. Moderators will only look at the latest submission (previous ones will be marked as "skipped" in your dashboard). You can delete pending submissions if needed, for example if you made an error but want to re-use a version number.

## Important Policies

BakkesPlugins retains the right to deny or remove any plugin. **A plugin will not be approved if it:**

- Can be considered malicious or harmful
- Downloads and/or runs potentially dangerous files
- Breaks Rocket League's Code of Conduct rules for mods
- Violates Psyonix's policies for third-party modifications
- Contains inappropriate content
- Infringes on copyrights or intellectual property

If your plugin requires files from outside sources or has unusual build requirements, it may be denied if:
- The required files cannot be safely obtained. Use vcpkg as much as possible when adding 3rd party libraries
- The build process is unreasonably complex
- Moderators cannot verify the safety of external dependencies

## Need Help?

- Check the [BakkesMod SDK documentation](/wiki/bakkesmod-sdk) for development help
- Join the BakkesMod Discord server for community support
- Review the [Plugin Development Tutorial](/wiki/bakkesmod-sdk/plugin-tutorial) for best practices


---

### Troubleshooting Bakkesmod

Made it here because your BakkesMod isn't working? No worries, check this other wiki here  
[https://bakkesmod.fandom.com/wiki/Troubleshooting](https://bakkesmod.fandom.com/wiki/Troubleshooting)

---

### Useful Links

Here's some links that we couldn't find a good place for

Creating a .set file [https://bakkesmod.fandom.com/wiki/Plugin_settings_files](https://bakkesmod.fandom.com/wiki/Plugin_settings_files)  
ImGui demo [https://pthom.github.io/imgui_manual_online/manual/imgui_manual.html](https://pthom.github.io/imgui_manual_online/manual/imgui_manual.html)  
ImGui builder program [https://github.com/Code-Building/ImGuiBuilder](https://github.com/Code-Building/ImGuiBuilder)  
Useful game values (map size, boost locations, etc.) [https://github.com/RLBot/RLBot/wiki/Useful-Game-Values](https://github.com/RLBot/RLBot/wiki/Useful-Game-Values)  
UDK site for canvas. The BM [CanvasWrapper](/wiki/bakkesmod-sdk/classes/wrappers/canvaswrapper) has some of these features [https://docs.unrealengine.com/udk/Three/CanvasTechnicalGuide.html](https://docs.unrealengine.com/udk/Three/CanvasTechnicalGuide.html)  
[CanvasWrapper](/wiki/bakkesmod-sdk/classes/wrappers/canvaswrapper) library for improved tools [https://github.com/CinderBlocc/RenderingTools](https://github.com/CinderBlocc/RenderingTools)  
Unreal Command [https://bakkesmod.fandom.com/wiki/Unreal_command](https://bakkesmod.fandom.com/wiki/Unreal_command)

---

### Writing Files

Use `gameWrapper->GetBakkesModPath()` and use that as the base path to write to (this will be `%appdata%/bakkesmod/bakkesmod`). This works for both Epic and Steam installs 

Make sure you've included fstream
```cpp
#include <fstream>
```

C++17
```cpp
std::ofstream stream( gameWrapper->GetBakkesModPath() / "data" / "abc.txt" );
std::ofstream stream( gameWrapper->GetDataFolder() / "abc.txt" ); //Note the removal of "data"
stream << "def";
// %appdata%/bakkesmod/bakkesmod/data/abc.txt now includes "def"
```
Older C++
```cpp
//Note that windows paths require wide strings, so need to use the L to indicate a wide string
std::ofstream stream( gameWrapper->GetBakkesModPathW() + L"data/abc.txt" );
std::ofstream stream( gameWrapper->GetDataFolderW() + L"abc.txt" );
```

We highly recommend using C++17. This is the default in the template. If not using the template, in Visual Studio go to your project's properties -> C/C++ -> Language -> C++ Language Standard -> `ISO C++17 Standard (/std:c++17)`.
This will allow you to use std::filesystem::path which prevents a lot of_pathing errors (like weird characters in folder names and missing/mixed folder separators) and in general is just much easier to work with.

---


# Functions

# Functions

Documentation for functions.

---

### Using Function Hooks

Function hooks are a powerful part of BakkesMod. After a function is hooked, any time it is called by Rocket League, your code is called as well. Many useful ones are documented in [Commonly Hooked Functions](/wiki/bakkesmod-sdk/functions/commonly-hooked-functions). If you want to better understand them or hook a function that's undocumented, hopefully this page will help.

There are multiple types of hooks. First are the two simpler ones. `gameWrapper` is a [GameWrapper](/wiki/bakkesmod-sdk/classes/wrappers/gamewrapper) and will be accessible from any point in your code.
```cpp
gameWrapper->HookEvent("Function TAGame.Car_TA.SetVehicleInput",
  [this](std::string eventName) {
    // Your Code here
    // Call another function or just do your things here
});

gameWrapper->HookEventPost("Function TAGame.Car_TA.SetVehicleInput",
  [this](std::string eventName) {
    // Your Code here
});
```

`HookEvent` runs your code as the function is called
`HookEventPost` runs your code as the function is returned from
They return a `std::string` which is the name of the calling function. This is useful to differentiate what your caller is if you have multiple hooks calling the same code.

```cpp
gameWrapper->HookEventWithCaller<CarWrapper>("Function TAGame.Car_TA.SetVehicleInput",
  [this](CarWrapper caller, void* params, std::string eventname) {
    // Your Code here
});

gameWrapper->HookEventWithCallerPost<CarWrapper>("Function TAGame.Car_TA.SetVehicleInput",
  [this](CarWrapper caller, void* params, std::string eventname) {
    // Your Code here
});
```

The two WithCaller variants do the same thing, but let you know a bit more about what is using the function

They provide a `caller` value which is obtained from the function's name. After `Function TAGame` there will be a class name ending in `_TA`. If the name matches a wrapper name, you can hook it as that wrapper. `Car_TA` becomes [CarWrapper](/wiki/bakkesmod-sdk/wiki/bakkesmod-sdk/classes/wrappers/gameobject/carwrapper) for example. The image below breaks up a function call into parts. The class is GameEvent-TA, which can be cast to a [ServerWrapper](/wiki/bakkesmod/sdk/classes/wrappers/gameevent/serverwrapper).  
![Function naming convention](https://cdn.bakkesplugins.com/uploads/081d6069-07f5-47e2-9768-1202c996451c-1_functionnaming.png)  
The hook also provides a `void *` pointer. This points at the location of any parameters provided to the function. These aren't well documented, and are hard to guess, so feel free to ask about them if you think you need them. A couple useful functions and parameters can be found in [Commonly Hooked Functions](/wiki/bakkesmod-sdk/functions/commonly-hooked-functions)

For example `Function TAGame.Car_TA.OnHitBall` has a [BallWrapper](/wiki/bakkesmod-sdk/classes/wrappers/gameobject/ballwrapper) parameter. To access it, cast the `void *` pointer to a struct including the parameters

```cpp
struct CarHitBallParams {
  // This is a pointer to the ball's address 
  //  but needs to be constructed into a BallWrapper before use
  uintptr_t ball;
};

gameWrapper->HookEventWithCallerPost<CarWrapper>("Function TAGame.Car_TA.OnHitBall",
  [this](CarWrapper caller, void* params, std::string eventname) {
    // This cast is only safe if you're 100% sure the params are correct
    CarHitBallParams* params = (CarHitBallParams*) params; 
    BallWrapper ballHit = BallWrapper(params->ball);
    // Now you know what ball was hit!
});
```

Find functions with the [function scanner](/wiki/bakkesmod-sdk/functions/function-scanner)

Note that hooking the same function multiple times does not work properly.
Only the first callback will be executed. The others are silently ignored.

```cpp
/* Called  */ gameWrapper->HookEvent(eventName, [this](...) { LOG("Initial HookEvent"); });
/* Ignored */ gameWrapper->HookEvent(eventName, [this](...) { LOG("Another HookEvent"); });
/* Called  */ gameWrapper->HookEventPost(eventName, [this](...) { LOG("Initial HookEventPost"); });
/* Ignored */ gameWrapper->HookEventPost(eventName, [this](...) { LOG("Another HookEventPost"); });
/* Ignored */ gameWrapper->HookEventWithCaller<ActorWrapper>(eventName, [this](...) { LOG("Initial HookEventWithCaller"); });
/* Ignored */ gameWrapper->HookEventWithCaller<ActorWrapper>(eventName, [this](...) { LOG("Another HookEventWithCaller"); });
/* Ignored */ gameWrapper->HookEventWithCallerPost<ActorWrapper>(eventName, [this](...) { LOG("Initial HookEventWithCallerPost"); });
/* Ignored */ gameWrapper->HookEventWithCallerPost<ActorWrapper>(eventName, [this](...) { LOG("Another HookEventWithCallerPost"); });
```

---

### Function Scanner

Trying to hook an event but don't know what event to use? There's way more events out there than can be documented, so you can find one yourself with the function scanner

**I highly recommend this video** I made to understand how to use it, but there's also a text guide below
[https://www.youtube.com/watch?v=gDZ1wWKE8aw](https://www.youtube.com/watch?v=gDZ1wWKE8aw)

## Set the -dev option

On Steam:
Right click the game and hit properties
Then add the option  
![properties](https://cdn.bakkesplugins.com/uploads/994e1925-776e-4914-aeb5-66035e019fe7-1_steam-context-menu.png)  
![added option](https://cdn.bakkesplugins.com/uploads/b18a01f5-1faf-4767-990c-f3972208ccdf-1_steam-launch-options.png)

On Epic: 

Hit Settings
Scroll down to Rocket League, enable Additional Command Line arguments, then add -dev  
![epic settings](https://cdn.bakkesplugins.com/uploads/022dfb35-5320-4a4f-849a-e7a8715d379e-1_epic-settings.png)  
![epic added option](https://cdn.bakkesplugins.com/uploads/d73bf311-890b-49be-936d-9434c98a566b-1_epic-launch-options.png)

## Reboot and open scanner

Now that you've enabled -dev, reboot rocket league with bakkesmod open
Once injected, open the console with f6 and type `togglemenu devtools`
The scanner will now open  
![scanner](https://cdn.bakkesplugins.com/uploads/74751035-714b-4496-821f-06a2a5df7b5e-1_function-scanner.png)

## Use the scanner

You can select any words you think are relevant and add them to the whitelist. Any words that aren't relevant can be added to the blacklist. Separate words with `,` but no space `a, b` will match `a` and `[space]b`. When you hit apply, any functions containing words matching the whitelist and not matching the blacklist will appear. These will appear as they happen, so it's a good way to find out when functions fire, and what timing you want. [This video](https://www.youtube.com/watch?v=gDZ1wWKE8aw) does a much better job of explaining this part

## Hook the Function

When you've found the right function, hook it using `gameWrapper->HookEvent()`. [Details here](/wiki/bakkesmod-sdk/functions/using-function-hooks). Every time that function happens, your code will be run. This can be used to know when something important happens (a goal), get pointers to hard-to-find wrappers, or just be used to help you do something on a consistent timer.

If you want to use the caller, the Bakkesmod caller will be based off of the Rocket League calling class / calling object name. Most functions are called by classes named Something_TA. Remove the _TA at the end and look for any wrappers with similar names. If a wrapper matches, that is the caller. For example, `"Function TAGame.Car_TA.OnHitBall"` is called by a `Car_TA`. The prefix is `Car`, so it's wrapped by [CarWrapper](/wiki/bakkesmod-sdk/classes/wrappers/gameobject/carwrapper). Not all callers are wrapped though. If you absolutely need the caller and it's unwrapped, ask for help and give some reasoning as to why you need the caller. If it's useful, it might be added to BakkesMod.

This image shows the breakdown of the function name  
![functionnaming.png](https://cdn.bakkesplugins.com/uploads/081d6069-07f5-47e2-9768-1202c996451c-1_functionnaming.png)

Hooks are the hardest part of Bakkesmod programming, but can have the most interesting results

---

### Commonly Hooked Functions

This is a list of functions that we use regularly, as well as when they fire. We'll never be able to document them all but hopefully this helps. Hook them with [Function Hooks](/wiki/bakkesmod-sdk/functions/using-function-hooks)

The Bakkesmod caller will be based off of the Rocket League calling class / calling object name. Most functions are called by classes named Something_TA. Remove the _TA at the end and look for any wrappers with similar names. If a wrapper matches, that is the caller. For example, `"Function TAGame.Car_TA.OnHitBall"` is called by a `Car_TA`. The prefix is `Car`, so it's wrapped by [CarWrapper](/wiki/bakkesmod-sdk/classes/wrappers/gameobject/carwrapper). Not all callers are wrapped though.

This image shows the breakdown of the function name. The class shown is GameEvent_TA, which can be cast to a [ServerWrapper](/wiki/bakkesmod-sdk/classes/wrappers/gameevent/serverwrapper)  
![functionnaming.png](https://cdn.bakkesplugins.com/uploads/081d6069-07f5-47e2-9768-1202c996451c-1_functionnaming.png)

## [BallWrapper Caller](/wiki/bakkesmod-sdk/classes/wrappers/gameobject/ballwrapper)
`"Function TAGame.Ball_TA.OnCarTouch"`  
This event is called when a ball hits a car. The caller is the ball that hit, and the first parameter is the [CarWrapper](/wiki/bakkesmod-sdk/classes/wrappers/gameobject/carwrapper) touched
```cpp
struct BallCarTouchParams {
    uintptr_t car;
}
```

`"Function TAGame.Ball_TA.OnHitGoal"`  
Happens when a goal is scored.  
Note that this event is also called during the goal replay, if the player does not cancel it. If you want to filter out the event during goal replay, you can hook to
`"Function GameEvent_Soccar_TA.ReplayPlayback.BeginState` and `"Function GameEvent_Soccar_TA.ReplayPlayback.EndState"` in order to e.g. set a `bool` flag while replay playback is active.

`"Function TAGame.Ball_TA.Explode"`  
Happens when a ball explodes. Usually this means a goal was scored

## [BoostWrapper Caller](/wiki/bakkesmod-sdk/classes/wrappers/gameobject/carcomponent/boostwrapper)
`"Function TAGame.VehiclePickup_Boost_TA.Pickup"`
Happens when a car picks up a boost
```cpp
struct BoostPickupParams {
    uintptr_t car;
}
```

## [CarWrapper Caller](/wiki/bakkesmod-sdk/classes/wrappers/gameobject/carwrapper)
`"Function TAGame.Car_TA.SetVehicleInput"`  
This event is called every physics tick while you are playing. This is 120 times per second. This can be highly useful but isn't perfect. It doesn't fire while spectating matches. While hosting a match it fires once per tick per car in the match

`"Function TAGame.Car_TA.OnHitBall"`  
This event is called when a car hits a ball. The caller is the [CarWrapper](/wiki/bakkesmod-sdk/classes/wrappers/gameobject/carwrapper) that hit, and the first parameter is the [BallWrapper](/wiki/bakkesmod/sdk/classes/wrappers/gameobject/ballwrapper) touched
```cpp
struct CarBallTouchParams {
    uintptr_t ball;
}
```

## [ReplayDirectorWrapper Caller](/wiki/bakkesmod-sdk/classes/wrappers/gameevent/replaydirectorwrapper)
`"Function TAGame.ReplayDirector_TA.OnScoreDataChanged"`  
Happens when a goal replay shows a goal's scorer, assister, speed, and other information

## [ServerWrapper Caller](/wiki/bakkesmod-sdk/classes/wrappers/gameevent/serverwrapper)
`"Function TAGame.GameEvent_Soccar_TA.EventMatchEnded"`  
Happens at the end of a match when the podium is shown. If someone quits early it is not called

`"Function TAGame.GameEvent_Soccar_TA.OnGameTimeUpdated"`  
Happens when the time on the game clock changes. Note that this is not real time. At 0:00 with the ball up it won't be called

`"Function TAGame.GameEvent_Soccar_TA.OnOvertimeUpdated"`  
Happens when overtime begins

`"Function GameEvent_Soccar_TA.ReplayPlayback.BeginState"`  
Happens when a goal replay begins

`"Function GameEvent_Soccar_TA.ReplayPlayback.EndState"`  
Happens when a goal replay ends

`"Function GameEvent_Soccar_TA.Active.StartRound"`  
Happens when all players are set for a kickoff. This happens at every single kickoff

`"Function GameEvent_Soccar_TA.Countdown.BeginState"`  
Happens at the start of a kickoff countdown or at the beginning/reset of freeplay

`"Function TAGame.GameEvent_Soccar_TA.Destroyed"`  
Happens when a match is left or quit by the player. The caller may be null or unsafe to use as the match is destroyed

`"Function TAGame.GameEvent_TA.AddCar"`
Happens whenenever a car spawns. Useful if modifying car physics that may reset after a respawn

## [StatEventWrapper Caller](/wiki/bakkesmod-sdk/classes/wrappers/gameobject/stats/stateventwrapper)
### [StatEventWrapper functions are explained here](/wiki/bakkesmod-sdk/functions/stat-events)
`"Function TAGame.GFxHUD_TA.HandleStatTickerMessage"`  
Called any time a stat should appear on the top right ticker, even if it's disabled  
`"Function TAGame.GFxHUD_TA.HandleStatEvent"`  
Called any time a stat should appear in the center of the screen, even if it's disabled  

## [TeamWrapper Caller](/wiki/bakkesmod-sdk/classes/wrappers/gameobject/teamwrapper)
`"Function TAGame.Team_TA.PostBeginPlay"`  
Happens twice at the start of a match, once per team that is totally joined and ready. It can be used to detect the start of a match. This only works when the player is in the match as it starts, so it's inconsistent in casual where you can join matches in progress

## [TrainingEditor Caller](bakkesmod_api/Classes/Wrappers/GameEvent/TrainingEditorWrapper/)
`"Function TAGame.TrainingEditorMetrics_TA.TrainingShotAttempt"`  
Called whenever an attempt is started in a custom training pack. An attempt counts as started if the player starts moving in any way. Selecting a different shot, or resetting the shot does not fire this event (until the player moves).
If you are only interested in the initial ball touch for each shot, you can react to only the first `"Function TAGame.Ball_TA.OnCarTouch"` event after each `TrainingShotAttempt` for example.

`"Function TAGame.GameEvent_TrainingEditor_TA.OnInit"`  
Called whenever a new training pack is being loaded, or the current training pack was restarted.  
Be aware that `_gameWrapper->IsInCustomTraining()` will not yet return `true` at this point.

## Unwrapped callers
`"Function TAGame.GFxData_MainMenu_TA.MainMenuAdded"`  
Happens when the main menu is opened, either through booting the game or leaving a mode

`"Function Engine.GameViewportClient.Tick"`  
This event is called every rendered frame. This is very powerful, but again can have issues. Different framerates and stutters can make the number of ticks different on different PCs

---

### Hook Event With Any Caller

Add this to your plugin header and you can hook with caller for "anything". This will do a blind cast, so make absolutely sure you're hooking the right function or you will crash. There are no type checks here!

```cpp
template <typename T, typename std::enable_if<std::is_base_of<ObjectWrapper, T>::value>::type*>
void GameWrapper::HookEventWithCaller(std::string eventName,
                                      std::function<void(T caller, void* params, std::string eventName)> callback)
{
    auto wrapped_callback = [callback](ActorWrapper caller, void* params, std::string eventName)
    {
        callback(T(caller.memory_address), params, eventName);
    };
    HookEventWithCaller<ActorWrapper>(eventName, wrapped_callback);
}
```

And this for Post
```cpp
template <typename T, typename std::enable_if<std::is_base_of<ObjectWrapper, T>::value>::type*>
void GameWrapper::HookEventWithCallerPost(std::string eventName,
                                      std::function<void(T caller, void* params, std::string eventName)> callback)
{
    auto wrapped_callback = [callback](ActorWrapper caller, void* params, std::string eventName)
    {
        callback(T(caller.memory_address), params, eventName);
    };
    HookEventWithCallerPost<ActorWrapper>(eventName, wrapped_callback);
}
```

---

### Modify vehicle inputs

Modifying vehicle input in BMSDK is only possible in offline modes or local games.
Since this is a frequent ask, here is a resource page on how to do it.

# The trap: don't use SetInput
Many aspiring programmers find the
[CarWrapper::SetInput](https://bakkesmodwiki.github.io/wiki/bakkesmod-sdk/classes-wrappers-gameobject-carwrapper-#voidhttpsencppreferencecomwcpplanguagetypesvoid_typenbspdivsetwbr-inputwbr-controllerwbr-input-inputdiv)
function and think that their prayers are answered.
Unfortunately, this function does not work particularly well.
It is best to steer clear of this one entirely.

# The good-enough: consider PlayerControllerWrapper::SetVehicleInput
The function [PlayerControllerWrapper::SetVehicleInput](https://wiki.bakkesplugins.com/wiki/bakkesmod-sdk/classes-wrappers-playercontrollerwrapper-)
is simple to use and works well enough for most purposes.
However, some users have noticed that it does have minor inconsistencies,
which can compound in sensitive cases like a TAS tool.
If that is something you care about, you can use the hook (described below) instead.

# The best: SetVehicleInput hook
The hook SetVehicleInput is the best way to control car inputs,
but it also requires some more work than the above function.
This code snippet shows a working example:

```cpp
gameWrapper.HookEventWithCaller(
  "Function TAGame.Car_TA.SetVehicleInput",
  [] (CarWrapper caller, void* params, std::string eventName) {
    ControllerInput* input = static_cast<ControllerInput*>(params);
    input->jump = 1;
  });
```

This hook runs once per car, every physics tick.

---

### Stat Events

Stat events are a great way to show a slightly more complicated plugin logic. They are fairly common to use, use multiple wrappers, event hooks, and have some unique caveats. Also I made a plugin with them so I know how they work. Hopefully this helps someone

First add this include to the top of your .h  
`#include "bakkesmod/wrappers/GameObject/Stats/StatEventWrapper.h"`

You need to hook some events to use stat events. This depends on your use case.  
`"Function TAGame.GFxHUD_TA.HandleStatTickerMessage"` is called any time a stat should appear on the top right ticker, even if it's disabled.  
`"Function TAGame.GFxHUD_TA.HandleStatEvent"` is called any time a stat should appear in the center of the screen, even if it's disabled.  
The stat ticker catches stat events from all players, but fails to see first touches, clears, and centers. 
HandleStatEvent catches all stats, but only for the primary player. Depending on your use case you may need to only hook one or both. The ServerWrappers in these hooks are placeholders and we won't use them, as there is no GFxHUD_TA wrapper. Check [using function hooks for more details](/wiki/bakkesmod-sdk/functions/using-function-hooks)

```cpp
//  We need the params so we hook with caller, but there is no wrapper for the HUD
gameWrapper->HookEventWithCallerPost<ServerWrapper>("Function TAGame.GFxHUD_TA.HandleStatTickerMessage",
    [this](ServerWrapper caller, void* params, std::string eventname) {
        onStatTickerMessage(params);
    });

// hooked whenever the primary player earns a stat
gameWrapper->HookEventWithCallerPost<ServerWrapper>("Function TAGame.GFxHUD_TA.HandleStatEvent",
    [this](ServerWrapper caller, void* params, std::string eventname) {
        onStatEvent(params);
    });
```

Now we have the functions hooked, and we need the parameters. Either define these in your .h or right before using them in your .cpp
```cpp
// The structure of a ticker stat event
struct StatTickerParams {
    // person who got a stat
    uintptr_t Receiver;
    // person who is victim of a stat (only exists for demos afaik)
    uintptr_t Victim;
    // wrapper for the stat event
    uintptr_t StatEvent;
};

// structure of a stat event
struct StatEventParams {
    // always primary player
    uintptr_t PRI;
    // wrapper for the stat event
    uintptr_t StatEvent;
};
```

Now you can use them. For reference see [StatEventWrapper](/wiki/bakkesmod-sdk/classes/wrappers/gameobject/stats/stateventwrapper) and [PriWrapper](/wiki/bakkesmod/sdk/classes/wrappers/gameobject/priwrapper)
```cpp
void ClassName::onStatTickerMessage(void* params) {
    StatTickerParams* pStruct = (StatTickerParams*)params;
    PriWrapper receiver = PriWrapper(pStruct->Receiver);
    PriWrapper victim = PriWrapper(pStruct->Victim);
    StatEventWrapper statEvent = StatEventWrapper(pStruct->StatEvent);
}

void ClassName::onStatEvent(void* params) {
    StatEventParams* pStruct = (StatEventParams*)params;
    PriWrapper playerPRI = PriWrapper(pStruct->PRI);
    StatEventWrapper statEvent = StatEventWrapper(pStruct->StatEvent);
}
```

Now you probably want to do something special with a specific event. Get the event's name with `StatEventWrapper.GetEventName()`. This gets the stat's internal names regardless of game language. This is safest to use for string comparisons. For logging or displaying in the game's current language you can use `StatEventWrapper.GetLabel()`. Here's a list of all the `GetEventName`s. The only confusing one is that a "Demolish" is one demolition and a "Demolition" is an extermination. OwnGoal is a fun ticker event that does exactly what you'd expect, yet is never displayed anywhere in game. KO_ stats belong to the Knockout gamemode
```cpp
"Demolish"
"Demolition"
"Goal"
"Win"
"MVP"
"AerialGoal"
"BackwardsGoal"
"BicycleGoal"
"LongGoal"
"TurtleGoal"
"PoolShot"
"OvertimeGoal"
"HatTrick"
"Assist"
"Playmaker"
"Save"
"EpicSave"
"Savior"
"Shot"
"Center"
"Clear"
"FirstTouch"
"BreakoutDamage"
"BreakoutDamageLarge"
"LowFive"
"HighFive"
"HoopsSwishGoal"
"BicycleHit"
"OwnGoal"
"KO_Winner"
"KO_Knockout"
"KO_DoubleKO"
"KO_TripleKO"
"KO_Death"
"KO_LightHit"
"KO_HeavyHit"
"KO_AerialLightHit"
"KO_AerialHeavyHit"
"KO_HitTaken"
"KO_BlockTaken"
"KO_Grabbed"
"KO_Thrown"
"KO_LightBlock"
"KO_HeavyBlock"
"KO_PlayerGrabbed"
"KO_PlayerThrown"
```

Now I'll just put together a simple example to track if the main player is demoed. As always this is assuming this is using the template
```cpp
// add to onload 
gameWrapper->HookEventWithCallerPost<ServerWrapper>("Function TAGame.GFxHUD_TA.HandleStatTickerMessage",
    [this](ServerWrapper caller, void* params, std::string eventname) {
        onStatTickerMessage(params);
    });
```

```cpp
// defined elsewhere in .cpp
void ClassName::onStatTickerMessage(void* params) {
    struct StatTickerParams {
        uintptr_t Receiver;
        uintptr_t Victim;
        uintptr_t StatEvent;
    };

    StatTickerParams* pStruct = (StatTickerParams*)params;
    PriWrapper receiver = PriWrapper(pStruct->Receiver);
    PriWrapper victim = PriWrapper(pStruct->Victim);
    StatEventWrapper statEvent = StatEventWrapper(pStruct->StatEvent);

    if (statEvent.GetEventName() == "Demolish") {
        LOG("a demolition happened >:(");
        if (!receiver) { LOG("Null reciever PRI"); return; }
        if (!victim) { LOG("Null victim PRI"); return; }

        // Find the primary player's PRI
        PlayerControllerWrapper playerController = gameWrapper->GetPlayerController();
        if (!playerController) { LOG("Null controller"); return; }
        PriWrapper playerPRI = playerController.GetPRI();
        if (!playerPRI) { LOG("Null player PRI"); return; }

        // Compare the primary player to the victim
        if (playerPRI.memory_address != victim.memory_address) {
            LOG("Hah you got demoed get good {}", victim.GetPlayerName().ToString());
            return;
        }
        // Primary player is the victim!
        LOG("I was demoed!!! {} is toxic I'm uninstalling", receiver.GetPlayerName().ToString());
    }
}
```

Hopefully all of this is useful to someone using stat events in the future!

---

### NOTIFIER_PERMISSION

When a notifier should be usable. Most of the time `PERMISSION_ALL` works best, but any of these can be used depending on your usecase.

For example, when a notifier is set to `PERMISSION_MENU`, your notifier code is only run when the command is used in the main menu.

---

### Rotator

A description of the orientation of an object. [This github page describes what 3D space is in Rocket League](https://github.com/RLBot/RLBot/wiki/Useful-Game-Values)

Some additional notes:
* Rotator applies rotations in the order yaw-pitch-roll.
* The identity rotation for a car (i.e. P=Y=R=0) corresponds to the car being upright with its nose pointing towards the positive x-axis. If the car were at the center of the field, its nose would face towards the left mid-boost, from the blue team's perspective.
* Example: to get a car facing straight up, wheels toward the orange goal, you could yaw 90 degrees right, then pitch up 90 degrees. This corresponds to P=16383, Y=16383, R=0. You could also yaw 90 degrees left, pitch 90 degrees up, then roll 180 degrees. This corresponds to P=16383, Y=-16384, R=32767.

---
