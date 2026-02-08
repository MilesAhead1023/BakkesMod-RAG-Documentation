"""
Build Script for BakkesMod RAG GUI Windows Executable
======================================================

This script automates the process of building a standalone Windows .exe
for the BakkesMod RAG Documentation system.

Usage:
    python build_exe.py

Output:
    dist/BakkesMod_RAG_GUI/BakkesMod_RAG_GUI.exe
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def print_header(msg: str):
    """Print a formatted header message"""
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70 + "\n")

def check_dependencies():
    """Check if required build dependencies are installed"""
    print_header("Checking Build Dependencies")
    
    try:
        import PyInstaller
        print(f"✓ PyInstaller {PyInstaller.__version__} installed")
    except ImportError:
        print("✗ PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller installed")
    
    # Check if spec file exists
    spec_file = Path("bakkesmod_rag_gui.spec")
    if not spec_file.exists():
        print(f"✗ Spec file not found: {spec_file}")
        return False
    print(f"✓ Spec file found: {spec_file}")
    
    return True

def clean_build_dirs():
    """Remove old build artifacts"""
    print_header("Cleaning Build Directories")
    
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"Removing {dir_path}...")
            shutil.rmtree(dir_path)
            print(f"✓ Removed {dir_path}")
        else:
            print(f"• {dir_path} does not exist (skipping)")

def build_executable():
    """Build the executable using PyInstaller"""
    print_header("Building Executable with PyInstaller")
    
    cmd = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
        "bakkesmod_rag_gui.spec"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 70)
    
    try:
        subprocess.check_call(cmd)
        print("\n" + "✓" * 70)
        print("✓ Build completed successfully!")
        print("✓" * 70)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Build failed with error code {e.returncode}")
        return False

def verify_output():
    """Verify the build output"""
    print_header("Verifying Build Output")
    
    exe_path = Path("dist/BakkesMod_RAG_GUI/BakkesMod_RAG_GUI.exe")
    
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"✓ Executable created: {exe_path}")
        print(f"✓ Size: {size_mb:.1f} MB")
        
        # Check for required data directories
        dist_dir = exe_path.parent
        required_dirs = ['docs', 'docs_bakkesmod_only', 'templates']
        
        for dir_name in required_dirs:
            dir_path = dist_dir / dir_name
            if dir_path.exists():
                print(f"✓ Data directory included: {dir_name}/")
            else:
                print(f"⚠ Warning: Missing data directory: {dir_name}/")
        
        return True
    else:
        print(f"✗ Executable not found: {exe_path}")
        return False

def create_readme():
    """Create a README file for the distribution"""
    print_header("Creating Distribution README")
    
    readme_content = """BakkesMod RAG GUI - Standalone Windows Executable
================================================

QUICK START:
------------
1. Extract this folder to your desired location
2. Create a .env file with your API keys (copy .env.example)
3. Run BakkesMod_RAG_GUI.exe
4. The GUI will open in your default web browser at http://localhost:7860

REQUIREMENTS:
-------------
- Windows 10 or Windows 11
- Internet connection for API calls
- API keys for:
  * OpenAI (OPENAI_API_KEY)
  * Anthropic (ANTHROPIC_API_KEY)
  * Google/Gemini (GOOGLE_API_KEY)

CONFIGURATION:
--------------
Create a .env file in the same directory as BakkesMod_RAG_GUI.exe:

    OPENAI_API_KEY=your_openai_key_here
    ANTHROPIC_API_KEY=your_anthropic_key_here
    GOOGLE_API_KEY=your_google_key_here

FIRST RUN:
----------
The first time you run the application, it will:
1. Build the RAG index from documentation files (2-5 minutes)
2. Store the index in ./rag_storage/
3. Start the web interface

Subsequent runs will be much faster (~30 seconds) as they load the cached index.

TROUBLESHOOTING:
----------------
- If the exe doesn't start, run it from Command Prompt to see error messages
- Ensure your .env file has all three API keys configured
- Check that the docs/ and docs_bakkesmod_only/ folders are present
- The application requires ~500MB disk space for the index storage

For detailed documentation, see:
https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation

SUPPORT:
--------
GitHub Issues: https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation/issues
"""
    
    readme_path = Path("dist/BakkesMod_RAG_GUI/README.txt")
    readme_path.write_text(readme_content)
    print(f"✓ Created: {readme_path}")

def main():
    """Main build process"""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  BakkesMod RAG GUI - Windows Executable Builder".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n✗ Dependency check failed. Please fix the issues and try again.")
        return 1
    
    # Step 2: Clean old builds
    clean_build_dirs()
    
    # Step 3: Build executable
    if not build_executable():
        print("\n✗ Build failed. Check the error messages above.")
        return 1
    
    # Step 4: Verify output
    if not verify_output():
        print("\n✗ Build verification failed.")
        return 1
    
    # Step 5: Create distribution README
    create_readme()
    
    # Success message
    print_header("Build Complete!")
    print("Your executable is ready:")
    print("  📁 Location: dist/BakkesMod_RAG_GUI/")
    print("  🚀 Executable: BakkesMod_RAG_GUI.exe")
    print("  📖 Instructions: README.txt")
    print("\nNext steps:")
    print("  1. Copy .env.example to dist/BakkesMod_RAG_GUI/.env")
    print("  2. Edit .env and add your API keys")
    print("  3. Run BakkesMod_RAG_GUI.exe")
    print("  4. Access the GUI at http://localhost:7860")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
