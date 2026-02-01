# SuiteSpot Complete Logic Map
## A Non-Technical Guide to How Everything Works

---

# PART 1: THE BIG PICTURE

## What is SuiteSpot?

SuiteSpot is a plugin for Rocket League that **automatically loads training content after your matches end**. Instead of manually navigating menus after every game, SuiteSpot can:
- Load a training pack for you to practice
- Load a freeplay map (any stadium)
- Load a workshop map (custom community maps)
- Automatically queue you into the next match

Think of it as your personal assistant that sets up your practice session the moment your game finishes.

---

## How the Plugin is Organized (The Hub-and-Spoke System)

Imagine a bicycle wheel:
- **The HUB (center)** = `SuiteSpot.cpp` - This is the "brain" that controls everything
- **The SPOKES (extending out)** = All the other components that do specific jobs

The hub creates all the spokes when the plugin starts, tells them what to do, and cleans them up when the plugin closes. The spokes don't talk directly to each other - they report back to the hub.

```
                    ┌─────────────────┐
                    │   SUITESPOT     │
                    │   (The Hub)     │
                    │                 │
                    │ Creates all     │
                    │ components at   │
                    │ startup         │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ MANAGERS      │  │ UI COMPONENTS   │  │ DATA STORAGE    │
│               │  │                 │  │                 │
│ • MapManager  │  │ • SettingsUI    │  │ • MapList       │
│ • SettingsSync│  │ • TrainingPackUI│  │ • DefaultPacks  │
│ • AutoLoad    │  │ • LoadoutUI     │  │ • ConstantsUI   │
│ • TrainingPack│  │ • StatusMessage │  │                 │
│ • Workshop    │  │ • HelpersUI     │  │                 │
│ • Loadout     │  │                 │  │                 │
│ • UsageTracker│  │                 │  │                 │
└───────────────┘  └─────────────────┘  └─────────────────┘
```

---

# PART 2: THE MAIN BRAIN - SUITESPOT.CPP

## What SuiteSpot.cpp Does

This is the command center. Here's what happens:

### When Rocket League Starts (onLoad function):
1. Creates all the helper components (managers and UI pieces)
2. Makes sure the data folders exist on your computer
3. Loads your saved workshop maps
4. Loads the training pack database
5. Sets up "hooks" - special connections to game events
6. Registers all your saved settings

### When a Match Ends (GameEndedEvent function):
1. The game signals "match ended"
2. SuiteSpot receives this signal
3. It tells AutoLoadFeature to do its job
4. If you're loading a training pack, it records that you used it

### When You Close Rocket League (onUnload function):
1. Saves your usage statistics
2. Cleans up all the components it created
3. Releases memory properly

### The "Pack Healer" Feature (TryHealCurrentPack function):
Sometimes training pack data gets corrupted or has wrong information. When you load a training pack, SuiteSpot can automatically fix the shot count if it's wrong - it reads the actual number from the game and updates the database.

---

# PART 3: THE AUTOMATION ENGINE - AUTOLOADFEATURE

## What AutoLoadFeature Does

This is the workhorse that decides what happens after a match. When called:

1. **Checks if enabled** - If you turned off SuiteSpot, it does nothing
2. **Reads your settings** - What mode did you select? What delays did you set?
3. **Determines what to load** based on your chosen mode:

### If Freeplay Mode:
- Finds which stadium you selected
- Waits for your specified delay
- Executes command: `load_freeplay [stadium_code]`

### If Training Mode:
- Checks your "Quick Picks" selection (your chosen training pack)
- If nothing selected, falls back to your most-used packs
- If still nothing, uses the default "Flicks Picks" list
- Waits for your specified delay
- Executes command: `load_training [pack_code]`
- Records that you used this pack (for the favorites feature)

### If Workshop Mode:
- Finds the workshop map file path you selected
- Waits for your specified delay
- Executes command: `load_workshop "[file_path]"`

### If Auto-Queue is On:
- After loading the map, schedules a `queue` command
- This puts you in line for the next match

### Safety Feature - Minimum Delay:
Even if you set 0 seconds delay, the plugin always waits at least 0.1 seconds. This prevents the game from crashing because Rocket League needs a moment to finish transitioning between states.

---

# PART 4: YOUR SETTINGS - SETTINGSSYNC

## What SettingsSync Does

This component remembers all your preferences. Think of it as the plugin's memory.

### Settings It Tracks:

| Setting | What It Controls |
|---------|-----------------|
| `enabled` | Is SuiteSpot turned on? |
| `mapType` | 0=Freeplay, 1=Training, 2=Workshop |
| `autoQueue` | Should it queue you after loading? |
| `delayQueueSec` | Seconds to wait before queuing (0-300) |
| `delayFreeplaySec` | Seconds to wait before loading freeplay |
| `delayTrainingSec` | Seconds to wait before loading training |
| `delayWorkshopSec` | Seconds to wait before loading workshop |
| `currentFreeplayCode` | Which stadium is selected (e.g., "Park_P") |
| `currentTrainingCode` | Which training pack code is selected |
| `currentWorkshopPath` | Which workshop map file is selected |
| `quickPicksSelected` | Which pack from Quick Picks is chosen |
| `quickPicksListType` | 0=Flicks Picks, 1=Your Favorites |
| `quickPicksCount` | How many packs to show in Quick Picks |

### How Settings Are Saved:

BakkesMod uses "CVars" (Console Variables) - these are automatically saved to a configuration file. When you change a slider or checkbox:
1. The UI tells SettingsSync to update
2. SettingsSync updates its internal copy AND tells BakkesMod to save it
3. Next time you open Rocket League, BakkesMod restores all your settings

---

# PART 5: FINDING MAPS - MAPMANAGER

## What MapManager Does

This component is the "map finder." It knows where things are stored on your computer.

### Folder Locations It Manages:

```
%APPDATA%\bakkesmod\bakkesmod\data\
├── SuiteSpot\
│   └── TrainingSuite\
│       ├── training_packs.json    (2000+ training packs)
│       └── pack_usage_stats.json  (your usage history)
└── WorkshopMapLoader\
    └── workshopmaploader.cfg      (workshop folder path)
```

### Workshop Map Discovery:

When you click "Refresh" in the workshop section:
1. MapManager checks the configured workshop folder path
2. Also checks default Steam and Epic Games locations
3. For each subfolder, it looks for `.upk` files (map files)
4. It reads any `.json` metadata files for map names, authors, descriptions
5. It finds preview images (.jpg, .png, .jfif)
6. Creates a sorted list of all discovered maps

### The Official Map List:

MapManager also provides the list of all 73 official Rocket League stadiums - from "AquaDome" to "Wasteland (Night)". Each has a code (like "Park_P" for Beckwith Park) that the game uses internally.

---

# PART 6: TRAINING PACK DATABASE - TRAININGPACKMANAGER

## What TrainingPackManager Does

This is the library system for training packs - it manages over 2,000 packs!

### Core Functions:

**Loading Packs:**
- Reads `training_packs.json` from disk
- Parses all pack data (name, creator, difficulty, tags, shot count, etc.)
- Sorts everything alphabetically
- Tracks when the database was last updated

**Searching and Filtering:**
When you use the Training Pack Browser:
- **Search**: Finds packs by name or code (case-insensitive)
- **Filter by Difficulty**: Bronze, Silver, Gold, Platinum, Diamond, Champion, Supersonic Legend
- **Filter by Tags**: Aerials, Wall Play, Ground Shots, etc.
- **Filter by Video**: Only show packs with tutorial videos
- **Filter by Shots**: Minimum number of practice shots
- **Sort**: By name, creator, difficulty, shots, likes, or plays

**Managing Custom Packs:**
- Add your own training packs with custom codes
- Edit existing pack information
- Delete packs you don't want

**The "Heal" Feature:**
If a pack's shot count is wrong, the plugin can fix it automatically when you load that pack.

**Updating the Database:**
When you click "Update Pack List":
1. A PowerShell script runs in the background
2. Downloads data from prejump.com (a training pack website)
3. Fetches all 230+ pages of packs (~2,300 total)
4. Merges new data with your custom packs
5. Saves everything to disk

---

# PART 7: DOWNLOADING WORKSHOP MAPS - WORKSHOPDOWNLOADER

## What WorkshopDownloader Does

This component lets you search for and download workshop maps from the internet (RLMAPS database).

### Search Flow:
1. You type a search term (like "dribble" or "rings")
2. WorkshopDownloader queries the RLMAPS API
3. Results come back with map names, authors, descriptions
4. Preview images are downloaded and cached locally
5. Results appear in a grid you can browse

### Download Flow:
1. You click "Download" on a map you want
2. A popup asks which version/release to download
3. You confirm the download
4. The ZIP file downloads (progress bar shows status)
5. PowerShell extracts the ZIP
6. The `.udk` file is renamed to `.upk` (Rocket League's format)
7. A metadata JSON file is created
8. The map is ready to use!

### Safety Features:
- Only one search or download runs at a time
- Old search results are discarded if you start a new search
- File names are sanitized (no special characters that break Windows)
- Timeout protection prevents infinite waiting

---

# PART 8: CAR PRESETS - LOADOUTMANAGER

## What LoadoutManager Does

This manages your car presets (loadouts). In Rocket League, you can save different car configurations:

### Functions:
- **GetLoadoutNames()**: Returns list of all your saved presets
- **GetCurrentLoadoutName()**: Shows which preset is currently equipped
- **SwitchLoadout(name)**: Changes to a specific preset by name
- **SwitchLoadout(index)**: Changes to a preset by its position number
- **RefreshLoadoutCache()**: Reloads the list if you created new presets

### Safety:
All communication with the game happens on the game's main thread to prevent crashes.

---

# PART 9: USAGE STATISTICS - PACKUSAGETRACKER

## What PackUsageTracker Does

This remembers which training packs you use and how often.

### Data Tracked:
For each pack you load:
- **Pack Code**: The unique identifier
- **Load Count**: How many times you've used it
- **Last Loaded**: When you last used it

### "Your Favorites" Feature:
When you select "Your Favorites" in Quick Picks:
1. PackUsageTracker sorts packs by load count (most used first)
2. For ties, it uses "last loaded" as tiebreaker
3. Returns your top N packs

This means the more you use a pack, the easier it is to find!

---

# PART 10: THE SETTINGS MENU - SETTINGSUI

## What SettingsUI Does

This is what you see when you press F2 and click on SuiteSpot. It has three tabs:

### Tab 1: "Map Select"
- **Auto-Queue Toggle**: Enable/disable automatic queuing
- **Delay Sliders**: Set wait times before loading/queuing
- **Mode Selection**: Radio buttons for Freeplay/Training/Workshop
- **Map Selection**:
  - Freeplay: Dropdown of all 73 stadiums
  - Training: Quick Picks list (Flicks Picks or Your Favorites)
  - Workshop: Two-panel browser with preview images

### Tab 2: "Loadout Management"
- Shows currently equipped car preset
- Dropdown to select a different preset
- "Apply Loadout" button
- "Refresh Loadouts" button

### Tab 3: "Workshop Browser"
- Search box for finding maps online
- Grid of search results with preview images
- Download buttons for each map
- Shows download progress

### Status Bar:
At the top, shows what will happen next:
- Current mode (Freeplay/Training/Workshop)
- Selected map/pack name
- Delay settings
- Queue status

### "Load Now" Button:
Big button that immediately loads your selected content without waiting for a match to end.

---

# PART 11: THE TRAINING PACK BROWSER - TRAININGPACKUI

## What TrainingPackUI Does

This is a floating window for browsing all 2,000+ training packs.

### Features:
- **Search Box**: Find packs by name or code
- **Filters**: Difficulty, tags, minimum shots, video only
- **Sortable Columns**: Click headers to sort by name, difficulty, shots, likes, plays
- **Pack Details**: Name, creator, difficulty badge, shot count, popularity
- **Actions**: Set as post-match load, load immediately, add to favorites
- **Custom Pack Form**: Add your own packs manually
- **Update Button**: Download latest pack database

### Visual Elements:
- Color-coded difficulty badges (Bronze=brown, Gold=golden, Diamond=blue, etc.)
- YouTube icon for packs with tutorial videos
- Hover tooltips showing creator and tags

---

# PART 12: SUPPORTING COMPONENTS

## StatusMessageUI - Notification System
Shows temporary colored messages:
- **Green**: Success ("Pack loaded!")
- **Red**: Error ("Failed to download")
- **Yellow**: Warning ("Applying...")
- **Blue**: Information ("Loadouts refreshed")

Messages fade away automatically or have a "Dismiss" button.

## HelpersUI - Shared UI Tools
Reusable components like:
- Input boxes with range limits
- Dropdowns with tooltips
- Checkboxes that auto-save
- Status message helpers

## DefaultPacks - Built-in Training Packs
10 pre-installed packs:
1. Aerial Shots - Redirect (44 shots)
2. Ultimate Warm-Up (50 shots)
3. Backboard Therapy (40 shots)
4. Ground Shots (50 shots)
5. Wall to Air Dribble (3 shots)
6. Musty's Speed Flip Kickoff (3 shots)
7. Flip Reset (10 shots)
8. Power Shots (20 shots)
9. Saves (50 shots)
10. Double Shot Playground (50 shots)

## ConstantsUI - Visual Settings
All the colors, sizes, and spacing used throughout the UI.

## Logging - Activity Recording
Records what the plugin does to help with troubleshooting.

## Version - Version Tracking
Current version: 1.0.0 (Build 649)

---

# PART 13: HOW EVERYTHING CONNECTS

## The Complete Flow: Match Ends → Training Loads

```
1. YOU FINISH A MATCH
         │
         ▼
2. Rocket League fires "MatchEnded" event
         │
         ▼
3. SuiteSpot.cpp receives the event (GameEndedEvent function)
         │
         ▼
4. SuiteSpot asks SettingsSync: "What are the user's settings?"
         │
         ▼
5. SuiteSpot tells AutoLoadFeature: "Execute!"
         │
         ▼
6. AutoLoadFeature checks the mode (Freeplay/Training/Workshop)
         │
         ├── If FREEPLAY: Gets stadium code → Schedules load_freeplay command
         │
         ├── If TRAINING: Gets pack code → Schedules load_training command
         │                              → Tells PackUsageTracker to record usage
         │
         └── If WORKSHOP: Gets file path → Schedules load_workshop command
         │
         ▼
7. After delay, BakkesMod executes the scheduled command
         │
         ▼
8. YOU'RE IN YOUR TRAINING!
         │
         ▼
9. If Auto-Queue is on: "queue" command runs → You're in line for next match
```

## Data Storage Locations

```
%APPDATA%\bakkesmod\bakkesmod\
├── cfg\
│   └── config.cfg           ← Your settings (CVars)
│
└── data\
    └── SuiteSpot\
        └── TrainingSuite\
            ├── training_packs.json    ← 2000+ training packs
            └── pack_usage_stats.json  ← Your usage history
```

---

# PART 14: KEY SAFETY RULES THE PLUGIN FOLLOWS

1. **Never load immediately during match-end** - Always use at least 0.1 second delay to prevent crashes

2. **Thread safety** - Uses locks (mutexes) when multiple things might access data at once

3. **Graceful error handling** - If something fails, log it and continue instead of crashing

4. **Automatic saving** - Settings and usage stats save immediately so nothing is lost

5. **Background operations** - Downloads and updates happen in the background so the game doesn't freeze

6. **Stale data protection** - If new search starts, old results are ignored

7. **File name safety** - Special characters are removed from file names to prevent Windows errors

---

# SUMMARY

SuiteSpot is organized into clear components:

| Component | Job |
|-----------|-----|
| **SuiteSpot** | The brain - coordinates everything |
| **AutoLoadFeature** | Decides what loads after matches |
| **SettingsSync** | Remembers your preferences |
| **MapManager** | Finds maps on your computer |
| **TrainingPackManager** | Manages 2000+ training packs |
| **WorkshopDownloader** | Downloads maps from the internet |
| **LoadoutManager** | Switches car presets |
| **PackUsageTracker** | Tracks your most-used packs |
| **SettingsUI** | The F2 menu interface |
| **TrainingPackUI** | The floating pack browser |
| **LoadoutUI** | Car preset selection |
| **StatusMessageUI** | Temporary notifications |
| **HelpersUI** | Shared UI building blocks |

Everything flows through the hub (SuiteSpot.cpp), which creates components at startup, routes events between them, and cleans up when finished.

---

# APPENDIX: QUICK REFERENCE CARDS

## A. CVar Quick Reference

All settings use the `suitespot_` prefix:

| CVar Name | Type | Default | Description |
|-----------|------|---------|-------------|
| `suitespot_enabled` | bool | true | Master on/off switch |
| `suitespot_map_type` | int | 0 | 0=Freeplay, 1=Training, 2=Workshop |
| `suitespot_auto_queue` | bool | false | Queue after loading |
| `suitespot_delay_queue` | float | 60.0 | Seconds before queuing |
| `suitespot_delay_freeplay` | float | 3.0 | Seconds before freeplay load |
| `suitespot_delay_training` | float | 3.0 | Seconds before training load |
| `suitespot_delay_workshop` | float | 3.0 | Seconds before workshop load |
| `suitespot_freeplay_code` | string | "Park_P" | Current freeplay stadium |
| `suitespot_training_code` | string | "" | Current training pack code |
| `suitespot_workshop_path` | string | "" | Current workshop map path |
| `suitespot_quickpicks_selected` | int | -1 | Quick Picks selection index |
| `suitespot_quickpicks_list_type` | int | 0 | 0=Flicks, 1=Favorites |
| `suitespot_quickpicks_count` | int | 5 | Number of Quick Picks shown |

## B. File Locations Quick Reference

| File | Location | Purpose |
|------|----------|---------|
| Plugin DLL | `%APPDATA%\bakkesmod\bakkesmod\plugins\SuiteSpot.dll` | The plugin itself |
| Settings | `%APPDATA%\bakkesmod\bakkesmod\cfg\config.cfg` | All CVars |
| Training Packs | `%APPDATA%\bakkesmod\bakkesmod\data\SuiteSpot\TrainingSuite\training_packs.json` | Pack database |
| Usage Stats | `%APPDATA%\bakkesmod\bakkesmod\data\SuiteSpot\TrainingSuite\pack_usage_stats.json` | Your history |
| Workshop Config | `%APPDATA%\bakkesmod\bakkesmod\data\WorkshopMapLoader\workshopmaploader.cfg` | Workshop folder path |

## C. Console Commands Quick Reference

| Command | Example | Description |
|---------|---------|-------------|
| `load_freeplay` | `load_freeplay Park_P` | Load freeplay on a specific map |
| `load_training` | `load_training ABC1-DEF2-GHI3-JKL4` | Load a training pack by code |
| `load_workshop` | `load_workshop "C:\path\to\map.upk"` | Load a workshop map |
| `queue` | `queue` | Queue for the next match |

---

*This document provides a complete non-technical overview of SuiteSpot's logic and architecture. For technical implementation details, see `architecture.md`.*
