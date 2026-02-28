# -*- mode: python ; coding: utf-8 -*-
"""
Complete PyInstaller spec for BakkesMod RAG GUI with all dependencies properly bundled
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, get_module_file_attribute
import os

block_cipher = None

# Collect data files from all packages
datas = []
datas += [('docs', 'docs')]
datas += [('docs_bakkesmod_only', 'docs_bakkesmod_only')]
datas += [('templates', 'templates')]
datas += [('.env.example', '.')]

# Comprehensive Gradio bundling
datas += collect_data_files('gradio')
datas += collect_data_files('gradio_client')

# Other key packages with data files
datas += collect_data_files('llama_index')
datas += collect_data_files('anthropic')
datas += collect_data_files('openai')
datas += collect_data_files('pydantic')
datas += collect_data_files('safehttpx')
datas += collect_data_files('groovy')
datas += collect_data_files('orjson')
datas += collect_data_files('PIL')
datas += collect_data_files('pygments')

# All hidden imports
hiddenimports = []

# Gradio and its submodules
hiddenimports += collect_submodules('gradio')
hiddenimports += collect_submodules('gradio_client')
hiddenimports += [
    'gradio', 'gradio_client', 'gradio.components', 'gradio.themes',
    'gradio._simple_templates', 'gradio._simple_templates.simpledropdown'
]

# LlamaIndex comprehensive
hiddenimports += collect_submodules('llama_index')
hiddenimports += [
    'llama_index', 'llama_index.core', 'llama_index.core.schema',
    'llama_index.core.indices', 'llama_index.core.retrievers'
]

# LLM SDKs
hiddenimports += collect_submodules('anthropic')
hiddenimports += collect_submodules('openai')
hiddenimports += ['anthropic', 'openai']

# Other utilities
hiddenimports += collect_submodules('pydantic')
hiddenimports += collect_submodules('PIL')
hiddenimports += [
    'dotenv', 'nest_asyncio', 'pyarrow', 'pandas', 'numpy',
    'safehttpx', 'groovy', 'orjson', 'pygments', 'colorama'
]

# BakkesMod package
hiddenimports += collect_submodules('bakkesmod_rag')
hiddenimports += [
    'bakkesmod_rag', 'bakkesmod_rag.config', 'bakkesmod_rag.engine',
    'bakkesmod_rag.llm_provider', 'bakkesmod_rag.document_loader',
    'bakkesmod_rag.retrieval', 'bakkesmod_rag.cache', 'bakkesmod_rag.api',
    'bakkesmod_rag.guardrails', 'bakkesmod_rag.intent_router',
    'bakkesmod_rag.observability', 'bakkesmod_rag.resilience',
]

a = Analysis(
    ['rag_gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest', 'unittest', '_pytest', 'IPython', 'jupyter',
        'matplotlib', 'scipy'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BakkesMod_RAG_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='BakkesMod_RAG_GUI',
)
