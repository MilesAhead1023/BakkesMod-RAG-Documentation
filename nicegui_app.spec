# nicegui_app.spec
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

datas = []
# Note: nicegui data files are now collected by pyinstaller-hooks-contrib >= 2025.10
# datas += collect_data_files('nicegui')  # hook handles this
datas += collect_data_files('llama_index')
datas += collect_data_files('anthropic')
datas += collect_data_files('openai')
datas += [('docs_bakkesmod_only', 'docs_bakkesmod_only')]
datas += [('templates', 'templates')]
datas += [('.env.example', '.')]

# google.genai (was google.generativeai in older versions)
try:
    datas += collect_data_files('google.genai')
except Exception:
    pass

hiddenimports = []
hiddenimports += collect_submodules('nicegui')
hiddenimports += collect_submodules('bakkesmod_rag')
hiddenimports += collect_submodules('llama_index')
hiddenimports += collect_submodules('anthropic')
hiddenimports += collect_submodules('openai')
hiddenimports += [
    'dotenv',
    'nest_asyncio',
    'tiktoken',
    'faiss',
    'pywebview',
]

if sys.platform == 'win32':
    hiddenimports += ['pywebview.platforms.winforms']

excludes = [
    # Dev/test tools
    'pytest', 'unittest', 'IPython', 'matplotlib', 'notebook', 'jupyter',
    # Heavy ML/data packages pulled in transitively — not used by any feature
    'sklearn', 'scipy', 'pandas',
    'boto3', 'botocore', 's3transfer',
    # Unused GUI toolkits
    'PyQt5', 'PyQt6', 'PySide6', 'wx', 'tkinter',
]

a = Analysis(
    ['nicegui_app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BakkesModRAG',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='docs/icon.ico' if os.path.exists('docs/icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BakkesModRAG',
)
