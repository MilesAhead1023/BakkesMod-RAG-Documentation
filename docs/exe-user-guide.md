# BakkesMod RAG - Executable User Guide

## Overview

The BakkesMod RAG app is available as a **standalone Windows executable** built with NiceGUI. It opens as a native desktop window with 7 tabs (Query, Code Gen, Settings, and more). No Python installation or browser required!

## Quick Start

### For End Users (Using the Pre-built Executable)

1. **Download the executable package**
   - Extract the `BakkesModRAG` folder to your desired location
   - The folder should contain:
     - `BakkesModRAG.exe` - The main application
     - `docs/` - Documentation files
     - `docs_bakkesmod_only/` - BakkesMod SDK documentation
     - `templates/` - Code generation templates
     - `.env.example` - Environment variable template
     - Various DLL and data files

2. **Configure API Keys**
   - Copy `.env.example` to `.env`:
     ```cmd
     copy .env.example .env
     ```
   - Edit `.env` with a text editor and add your API keys:
     ```env
     OPENAI_API_KEY=sk-your-openai-key-here
     ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
     GOOGLE_API_KEY=your-google-api-key-here
     ```
   - **Required**: You need all three API keys for full functionality

3. **Run the Application**
   - Double-click `BakkesModRAG.exe`
   - Or run from Command Prompt:
     ```cmd
     BakkesModRAG.exe
     ```
   - A native desktop window will open with the application UI

4. **First Run**
   - The first time you run, the application will:
     - Build the RAG index from documentation (2-5 minutes)
     - Create a `rag_storage/` folder to cache the index
     - Start the web interface
   - Subsequent runs are much faster (~30 seconds)

## For Developers (Building the Executable)

### Prerequisites

- **Windows 10 or Windows 11**
- **Python 3.8 or higher** installed
- **Git** (optional, for cloning the repository)
- **API Keys** for testing:
  - OpenAI API key
  - Anthropic API key
  - Google/Gemini API key

### Building from Source

#### Method 1: Using the Build Script (Recommended)

1. **Clone or download the repository:**
   ```cmd
   git clone https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation.git
   cd BakkesMod-RAG-Documentation
   ```

2. **Run the build batch file:**
   ```cmd
   build_exe.bat
   ```
   
   This will:
   - Create a virtual environment (if needed)
   - Install all dependencies including PyInstaller
   - Build the executable
   - Verify the output

3. **Find your executable:**
   ```
   dist/BakkesModRAG/BakkesModRAG.exe
   ```

#### Method 2: Manual Build

1. **Create virtual environment:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   pip install pyinstaller
   ```

3. **Build the executable:**
   ```cmd
   pyinstaller --clean --noconfirm nicegui_app.spec
   ```

   Or run in development mode (no build needed):
   ```cmd
   python nicegui_app.py
   ```

### Build Configuration

The build is configured in `nicegui_app.spec`:

- **Entry point**: `nicegui_app.py`
- **Included data**:
  - `docs/` - Main documentation
  - `docs_bakkesmod_only/` - BakkesMod SDK docs
  - `templates/` - Code generation templates
  - `.env.example` - Configuration template
- **Hidden imports**: All LlamaIndex, NiceGUI, and LLM provider modules
- **Output**: `dist/BakkesModRAG/` (COLLECT/directory mode)

### Customizing the Build

Edit `nicegui_app.spec` to customize:

```python
# Add an icon
icon='path/to/icon.ico'

# Change console visibility
console=False  # Hide console window

# Add version info (requires additional configuration)
version='version_info.txt'
```

## Distribution

### Package Contents

The `dist/BakkesModRAG/` folder contains everything needed:

```
BakkesModRAG/
├── BakkesModRAG.exe          # Main executable (~50-100MB)
├── docs/                     # Documentation files
├── docs_bakkesmod_only/      # BakkesMod SDK documentation
├── templates/                # Code generation templates
├── .env.example              # Configuration template
├── README.txt                # Quick start guide
└── [various DLLs and data files]
```

### Distribution Size

- **Total package**: ~200-500 MB (compressed: ~100-200 MB)
- **After first run**: +500 MB for RAG index storage

### Creating a Distributable Package

1. **Build the executable** (see above)

2. **Copy the distribution folder:**
   ```cmd
   xcopy /E /I dist\BakkesModRAG BakkesModRAG_v1.0
   ```

3. **Create a README:**
   ```cmd
   copy dist\BakkesModRAG\README.txt BakkesModRAG_v1.0\
   ```

4. **Zip the folder:**
   - Right-click → Send to → Compressed (zipped) folder
   - Or use 7-Zip for better compression

5. **Distribute:**
   - Upload to GitHub Releases
   - Share via cloud storage
   - Include installation instructions

## Troubleshooting

### Executable Won't Start

**Issue**: Double-clicking the exe does nothing

**Solution**:
- Run from Command Prompt to see error messages:
  ```cmd
  cd path\to\BakkesModRAG
  BakkesModRAG.exe
  ```
- Check for error messages in the console window

### Missing API Keys

**Issue**: "API key not found" errors

**Solution**:
- Ensure `.env` file exists in the same directory as the exe
- Verify all three API keys are set:
  ```env
  OPENAI_API_KEY=sk-...
  ANTHROPIC_API_KEY=sk-ant-...
  GOOGLE_API_KEY=...
  ```
- Keys should not have quotes or extra spaces

### Slow First Run

**Issue**: Application takes 5+ minutes to start first time

**Solution**:
- This is normal - the app is building the RAG index
- Subsequent runs will be much faster (~30 seconds)
- The `rag_storage/` folder caches the index

### Window Doesn't Appear

**Issue**: Console opens but no application window appears

**Solution**:
- Wait a moment for the native window to initialize
- Check the console for error messages
- Ensure `.env` file is properly configured

### "Import Error" or "Module Not Found"

**Issue**: Errors about missing modules

**Solution**:
- The exe should be self-contained
- If this happens, rebuild with:
  ```cmd
  pyinstaller --clean --noconfirm nicegui_app.spec
  ```
- Check that all required data directories are present

### Large File Size

**Issue**: Executable package is very large (>500 MB)

**Solution**:
- This is expected - includes Python runtime + all dependencies
- Use zip compression for distribution
- Consider using UPX compression (enabled by default in spec file)

## Performance

### Startup Time

- **First run**: 2-5 minutes (building index)
- **Subsequent runs**: ~30 seconds (loading cached index)
- **With existing cache**: ~10 seconds

### Memory Usage

- **Idle**: ~500 MB - 1 GB
- **During query**: +200-500 MB
- **With cache loaded**: ~1-2 GB total

### Disk Space

- **Executable package**: ~200-500 MB
- **RAG index storage**: ~500 MB (after first run)
- **Total**: ~700 MB - 1 GB

## Advanced Configuration

### Environment Variables

Create a `.env` file with additional settings:

```env
# Required API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional: Cost Management
DAILY_BUDGET_USD=10.0

# Optional: Logging
LOG_LEVEL=INFO

# Optional: Storage Location
RAG_STORAGE_DIR=./custom_storage

# Optional: Server Configuration (for Gradio/Docker only)
# GRADIO_SERVER_PORT=7860
# GRADIO_SERVER_NAME=0.0.0.0
```

### Command Line Options

Currently, the executable uses default settings. To customize:

1. Create a config file
2. Modify the spec file to accept command-line arguments
3. Rebuild the executable

## Security Considerations

### API Key Safety

- **Never share** your `.env` file
- **Never commit** API keys to version control
- **Restrict access** to the executable folder
- Keys are loaded at runtime, not embedded in the exe

### Antivirus False Positives

Some antivirus software may flag PyInstaller executables:

- This is a **false positive**
- PyInstaller creates self-extracting archives that some AV flags
- Add an exception in your antivirus for the exe
- Or build from source and verify yourself

## Updates

### Updating the Executable

1. Download the latest version
2. Replace the old `BakkesModRAG.exe`
3. Keep your `.env` file and `rag_storage/` folder
4. Restart the application

### Updating Documentation

The docs are embedded in the executable. To update:

1. Get the latest source code
2. Rebuild the executable with `build_exe.bat`
3. The new exe will include updated docs

## Support

### Getting Help

- **Documentation**: [GitHub Repository](https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation)
- **Issues**: [GitHub Issues](https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation/discussions)

### Reporting Bugs

When reporting issues, include:

1. Windows version (10 or 11)
2. Executable version
3. Error messages from the console
4. Steps to reproduce
5. Whether it's first run or subsequent run

### Feature Requests

Submit feature requests on GitHub Issues with:

- Clear description of the feature
- Use case / why it's needed
- Expected behavior
- Any implementation ideas

## License

This software is licensed under the MIT License. See the repository for full license text.

## Credits

- **BakkesMod**: Rocket League mod framework
- **NiceGUI**: Native desktop UI framework
- **LlamaIndex**: RAG framework
- **PyInstaller**: Python to executable converter

---

**Built with ❤️ for the BakkesMod community**
