# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for BakkesMod RAG GUI
Creates a standalone Windows executable with all dependencies
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all data files
datas = []

# Add documentation directories
datas += [('docs', 'docs')]
datas += [('docs_bakkesmod_only', 'docs_bakkesmod_only')]
datas += [('templates', 'templates')]
datas += [('.env.example', '.')]

# Collect data files from packages
datas += collect_data_files('gradio')
datas += collect_data_files('gradio_client')
datas += collect_data_files('llama_index')
datas += collect_data_files('anthropic')
datas += collect_data_files('openai')
datas += collect_data_files('tiktoken')
datas += collect_data_files('pygments')

# Hidden imports - modules that PyInstaller might miss
hiddenimports = []

# Gradio and dependencies
hiddenimports += collect_submodules('gradio')
hiddenimports += collect_submodules('gradio_client')
hiddenimports += ['gradio.components', 'gradio.themes']

# LlamaIndex and all its extensions
hiddenimports += collect_submodules('llama_index')
hiddenimports += ['llama_index.core', 'llama_index.llms', 'llama_index.embeddings']
hiddenimports += ['llama_index.llms.anthropic', 'llama_index.llms.openai']
hiddenimports += ['llama_index.embeddings.openai']
hiddenimports += ['llama_index.retrievers.bm25']
hiddenimports += ['llama_index.core.node_parser']
hiddenimports += ['llama_index.postprocessor.cohere_rerank']

# LLM Provider SDKs
hiddenimports += collect_submodules('anthropic')
hiddenimports += collect_submodules('openai')
hiddenimports += collect_submodules('google')
hiddenimports += ['google.genai']
hiddenimports += ['llama_index.llms.google_genai']

# Utilities
hiddenimports += collect_submodules('pygments')
hiddenimports += collect_submodules('colorama')

# Other dependencies
hiddenimports += ['tiktoken_ext.openai_public', 'tiktoken_ext']
hiddenimports += ['pydantic', 'pydantic.dataclasses']
hiddenimports += ['dotenv']
hiddenimports += ['nest_asyncio']
hiddenimports += ['flashrank']
hiddenimports += ['bm25s']

# Unified bakkesmod_rag package
hiddenimports += collect_submodules('bakkesmod_rag')
hiddenimports += [
    'bakkesmod_rag',
    'bakkesmod_rag.config',
    'bakkesmod_rag.llm_provider',
    'bakkesmod_rag.document_loader',
    'bakkesmod_rag.retrieval',
    'bakkesmod_rag.cache',
    'bakkesmod_rag.query_rewriter',
    'bakkesmod_rag.confidence',
    'bakkesmod_rag.code_generator',
    'bakkesmod_rag.compiler',
    'bakkesmod_rag.engine',
    'bakkesmod_rag.cost_tracker',
    'bakkesmod_rag.observability',
    'bakkesmod_rag.resilience',
    'bakkesmod_rag.answer_verifier',
    'bakkesmod_rag.query_decomposer',
    'bakkesmod_rag.cpp_analyzer',
    'bakkesmod_rag.feedback_store',
    'bakkesmod_rag.setup_keys',
    'bakkesmod_rag.sentinel',
    'bakkesmod_rag.evaluator',
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
        # Exclude test modules
        'pytest', 'unittest', '_pytest',
        # Exclude development tools
        'IPython', 'jupyter',
        # Exclude unnecessary packages
        'matplotlib', 'scipy', 'numpy.distutils'
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
    console=True,  # Show console for error messages
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Could add an icon file here
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BakkesMod_RAG_GUI',
)
