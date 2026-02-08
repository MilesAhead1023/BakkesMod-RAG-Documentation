# Building the Windows Executable

This guide shows how to build a standalone Windows executable for the BakkesMod RAG GUI.

## Quick Build (Recommended)

```cmd
build_exe.bat
```

This single command will:
1. Create a virtual environment (if needed)
2. Install all dependencies
3. Install PyInstaller
4. Build the executable
5. Verify the output

## Output Location

After building, you'll find:

```
dist/BakkesMod_RAG_GUI/
├── BakkesMod_RAG_GUI.exe    ← Main executable
├── docs/                     ← Documentation files
├── docs_bakkesmod_only/      ← BakkesMod SDK docs
├── templates/                ← Code templates
├── .env.example              ← Config template
├── README.txt                ← User instructions
└── [DLLs and data files]
```

## Manual Build

If you prefer manual control:

```cmd
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install pyinstaller

# 3. Run build script
python build_exe.py

# Or use PyInstaller directly
pyinstaller --clean --noconfirm bakkesmod_rag_gui.spec
```

## Distribution

To create a distributable package:

1. **Zip the folder:**
   ```cmd
   # Navigate to dist folder
   cd dist
   
   # Create zip (using PowerShell)
   Compress-Archive -Path BakkesMod_RAG_GUI -DestinationPath BakkesMod_RAG_GUI_v1.0.zip
   ```

2. **Upload to GitHub Releases:**
   - Go to repository Releases page
   - Create a new release
   - Upload the zip file
   - Add release notes

## Testing the Executable

Before distributing:

1. **Copy to a test location:**
   ```cmd
   xcopy /E /I dist\BakkesMod_RAG_GUI C:\Test\BakkesMod_RAG_GUI
   cd C:\Test\BakkesMod_RAG_GUI
   ```

2. **Create .env file:**
   ```cmd
   copy .env.example .env
   notepad .env
   ```
   Add your API keys

3. **Run the executable:**
   ```cmd
   BakkesMod_RAG_GUI.exe
   ```

4. **Verify:**
   - Console window opens showing startup
   - Web browser opens to http://localhost:7860
   - GUI loads successfully
   - Can query documentation
   - Can generate code

## Build Configuration

Edit `bakkesmod_rag_gui.spec` to customize:

### Add an Icon

```python
exe = EXE(
    # ... other settings ...
    icon='path/to/icon.ico',  # Add this line
)
```

### Hide Console Window

```python
exe = EXE(
    # ... other settings ...
    console=False,  # Change from True to False
)
```

### Exclude More Modules

```python
excludes=[
    'pytest', 'unittest', '_pytest',
    'IPython', 'jupyter',
    'matplotlib', 'scipy', 'numpy.distutils',
    # Add more here
],
```

## Troubleshooting

### Build Fails with Import Errors

**Solution:** Add missing module to `hiddenimports` in the spec file:

```python
hiddenimports += ['missing_module_name']
```

### Executable Size Too Large

The executable will be large (~200-500 MB) because it includes:
- Python runtime
- All dependencies (Gradio, LlamaIndex, LLM providers)
- Documentation files

To reduce size:
- Enable UPX compression (already enabled)
- Exclude unnecessary modules
- Use external data files instead of bundling

### Build Takes a Long Time

First build: 5-10 minutes (PyInstaller analyzes all dependencies)
Subsequent builds: 2-5 minutes (uses cache)

### Antivirus Flags the Executable

This is a common false positive with PyInstaller executables:
- Add exception in your antivirus
- Or sign the executable with a code signing certificate
- Or build from source on the target machine

## Advanced: Code Signing

To sign the executable (requires certificate):

```cmd
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com dist\BakkesMod_RAG_GUI\BakkesMod_RAG_GUI.exe
```

## Requirements

- Windows 10 or 11
- Python 3.8+ (for building)
- ~2 GB free disk space (for build artifacts)
- ~500 MB for final distribution package

## Support

For issues with building the executable:
- Check [EXE_USER_GUIDE.md](EXE_USER_GUIDE.md)
- Open an issue on GitHub
- Include build output and error messages

---

**Ready to distribute to users who don't have Python installed!**
