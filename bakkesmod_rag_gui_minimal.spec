# -*- mode: python ; coding: utf-8 -*-
"""
Minimal PyInstaller spec for BakkesMod RAG GUI (workaround for hook discovery issues)
"""

a = Analysis(
    ['rag_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('docs', 'docs'),
        ('docs_bakkesmod_only', 'docs_bakkesmod_only'),
        ('templates', 'templates'),
        ('.env.example', '.'),
    ],
    hiddenimports=[
        # Core dependencies
        'gradio', 'gradio_client',
        'llama_index',
        'llama_index.core',
        'llama_index.retrievers',
        'llama_index.retrievers.bm25',
        'anthropic',
        'openai',
        'pydantic',
        'dotenv',
        # BakkesMod package
        'bakkesmod_rag',
        'bakkesmod_rag.config',
        'bakkesmod_rag.engine',
        'bakkesmod_rag.llm_provider',
        'bakkesmod_rag.document_loader',
        'bakkesmod_rag.retrieval',
        'bakkesmod_rag.cache',
        'bakkesmod_rag.api',
        'bakkesmod_rag.guardrails',
        'bakkesmod_rag.intent_router',
        'bakkesmod_rag.observability',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'unittest', '_pytest', 'IPython', 'jupyter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

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
