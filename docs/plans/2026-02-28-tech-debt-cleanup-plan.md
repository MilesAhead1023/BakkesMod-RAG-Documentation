# Tech Debt Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean up all technical debt across the BakkesMod RAG Documentation project in 6 discrete, reviewable stages without losing any features.

**Architecture:** Surgical sweep — commit pending git deletions first to establish a clean baseline, then remove misplaced docs, prune integration tests (mark, don't delete), rebuild requirements.txt as three focused files, do light code cleanup, and update developer docs.

**Tech Stack:** Python 3.x, LlamaIndex, Gradio, Pydantic, pytest, pip, git

---

## Context

- Working branch: `feb-work`
- Package lives in `bakkesmod_rag/` (27 modules)
- Two entry points: `interactive_rag.py` (CLI), `rag_gui.py` (Gradio GUI)
- 563 tests in `tests/` across 29 files
- Design doc: `docs/plans/2026-02-28-tech-debt-cleanup-design.md`

---

## Stage 1 — Git Hygiene

### Task 1: Stage all pending deletions and modifications

**Files:**
- All files listed in `git status --short`

**Step 1: Verify what's staged**

```bash
git status --short
```

Expected: You'll see `M` (modified) and `D` (deleted) files, and `??` (untracked: `pytest.ini`, `tests/test_smoke.py`).

**Step 2: Stage all deletions and modifications**

```bash
git add -u
git add pytest.ini tests/test_smoke.py
```

`-u` stages tracked changes (modifications + deletions). The two `??` files are new and need explicit `add`.

**Step 3: Verify staging**

```bash
git status --short
```

Expected: Everything should show with a green `M` or `D` (staged), and `A` for the two new files.

**Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
chore: commit housekeeping — delete obsolete docs, plans, and old test files

Removes: DEEPWIKI_COMPARISON.md, FEATURE_ENHANCEMENTS.md,
PROJECT_COMPLETION_SUMMARY.md, TEST_RESULTS.md,
bakkesmod_rag_gui_minimal.spec, docs/bakkesmod-sdk-reference.md,
docs/bakkesmod_imgui_signatures_annotated.md, docs/plans/2026-02-05-*,
docs/plans/2026-02-07-*, test_smoke.py (root), test_smoke.spec.

Adds: pytest.ini, tests/test_smoke.py (new canonical location).

Updates: .gitignore, CLAUDE.md, CONTRIBUTING.md, Dockerfile,
QUICK_START.md, README.md, bakkesmod_rag_gui.spec,
docs/deployment-guide.md, rag_gui.py, requirements.txt.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
```

**Step 5: Verify clean working tree**

```bash
git status
```

Expected: `nothing to commit, working tree clean`

---

## Stage 2 — Docs Cleanup

### Task 2: Remove misplaced SuiteSpot docs

**Files:**
- Delete: `docs/LOGIC_MAP.md`
- Delete: `docs/architecture.md`

**Step 1: Verify these are SuiteSpot docs (not RAG system docs)**

Search for "SuiteSpot" in both files:
```bash
grep -l "SuiteSpot" docs/LOGIC_MAP.md docs/architecture.md
```

Expected: Both files listed (they describe a different plugin entirely).

**Step 2: Delete them**

```bash
git rm docs/LOGIC_MAP.md docs/architecture.md
```

**Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
docs: remove misplaced SuiteSpot plugin docs

LOGIC_MAP.md and architecture.md describe the SuiteSpot BakkesMod
plugin, not this RAG system. They were likely copied here by mistake.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Remove outdated planning/summary docs

**Files:**
- Delete: `docs/2026-gold-standard-architecture.md`
- Delete: `docs/2026-upgrade-summary.md`

**Step 1: Confirm content is already captured in CLAUDE.md**

Both files describe the architecture that is now fully documented in `CLAUDE.md`. They are historical planning docs, not current references.

**Step 2: Delete them**

```bash
git rm docs/2026-gold-standard-architecture.md docs/2026-upgrade-summary.md
```

**Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
docs: remove outdated 2026 planning and architecture summary docs

Content is now fully reflected in CLAUDE.md. These historical docs
add confusion without value.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Move root-level user guides into docs/

**Files:**
- Move: `BUILD_EXE_GUIDE.md` → `docs/build-exe-guide.md`
- Move: `EXE_USER_GUIDE.md` → `docs/exe-user-guide.md`
- Move: `GUI_USER_GUIDE.md` → `docs/gui-user-guide.md`
- Move: `DEBUG_GUIDE.md` → `docs/debug-guide.md`
- Move: `DEPLOYMENT_GUIDE.md` → merge into `docs/deployment-guide.md`
- Modify: `README.md` — update any links to these files

**Step 1: Move files with git mv**

```bash
git mv BUILD_EXE_GUIDE.md docs/build-exe-guide.md
git mv EXE_USER_GUIDE.md docs/exe-user-guide.md
git mv GUI_USER_GUIDE.md docs/gui-user-guide.md
git mv DEBUG_GUIDE.md docs/debug-guide.md
```

**Step 2: Merge DEPLOYMENT_GUIDE.md into docs/deployment-guide.md**

Open `DEPLOYMENT_GUIDE.md` and `docs/deployment-guide.md`. Merge any content from the root file that isn't in the docs version (check for differences). Then:

```bash
git rm DEPLOYMENT_GUIDE.md
```

**Step 3: Update README.md links**

Search README.md for references to the moved files:
```bash
grep -n "BUILD_EXE_GUIDE\|EXE_USER_GUIDE\|GUI_USER_GUIDE\|DEBUG_GUIDE\|DEPLOYMENT_GUIDE" README.md
```

Update any found links from `BUILD_EXE_GUIDE.md` → `docs/build-exe-guide.md`, etc.

**Step 4: Run smoke tests to confirm nothing broken**

```bash
pytest tests/test_smoke.py -v
```

Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
docs: consolidate user guides into docs/ directory

Moves BUILD_EXE_GUIDE, EXE_USER_GUIDE, GUI_USER_GUIDE, DEBUG_GUIDE
from root into docs/ with kebab-case naming. Merges DEPLOYMENT_GUIDE
into docs/deployment-guide.md. Updates README links.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Stage 3 — Test Suite Pruning

### Task 5: Mark integration tests with @pytest.mark.integration

**Files:**
- Modify: `tests/test_golden_queries.py`
- Modify: `tests/test_engine_integration.py`
- Modify: `tests/test_code_gen_integration.py`

**Step 1: Open each file and understand what it tests**

For each file, check if it requires a live index (`rag_storage/`), live API keys, or makes network calls.

**Step 2: Add marker to test classes or functions that require live resources**

In each test file, add the `@pytest.mark.integration` decorator to all test classes and functions that require API keys or a built index:

```python
import pytest

@pytest.mark.integration
class TestGoldenQueries:
    ...
```

Or at the function level:
```python
@pytest.mark.integration
def test_query_returns_correct_answer():
    ...
```

**Step 3: Add a module-level skip marker if ALL tests in file are integration**

At the top of each file (after imports), if every test needs live resources:

```python
pytestmark = pytest.mark.integration
```

This marks all tests in the module at once.

**Step 4: Verify fast suite still collects correctly**

```bash
pytest -m "not integration" --collect-only -q 2>&1 | tail -5
```

Expected: Fewer tests collected (integration tests excluded), no errors.

**Step 5: Run fast suite**

```bash
pytest -m "not integration" -v --tb=short 2>&1 | tail -20
```

Expected: All collected tests PASS (or xfail). Zero failures. No API key errors.

**Step 6: Commit**

```bash
git commit -m "$(cat <<'EOF'
test: mark integration tests — exclude from default pytest run

Golden queries, engine integration, and code gen integration tests
require a live index and API keys. Mark them @pytest.mark.integration
so the default test run (no flags) stays fast and key-free.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Review semi-integration tests

**Files:**
- Modify (if needed): `tests/test_self_rag.py`
- Modify (if needed): `tests/test_self_improving_codegen.py`
- Modify (if needed): `tests/test_semantic_chunking.py`

**Step 1: Open each file and search for live API calls**

```bash
grep -n "os.environ\|api_key\|RAGEngine()\|build_index\|query_engine" \
  tests/test_self_rag.py tests/test_self_improving_codegen.py tests/test_semantic_chunking.py
```

**Step 2: For any tests using real API calls, add @pytest.mark.integration**

Use same technique as Task 5. If a test mocks the LLM with `MockLLM`, it doesn't need marking.

**Step 3: Run fast suite again**

```bash
pytest -m "not integration" -q 2>&1 | tail -5
```

Expected: PASSED (number decreases only for newly marked tests).

**Step 4: Commit if any changes were made**

```bash
git add tests/
git commit -m "$(cat <<'EOF'
test: mark remaining semi-integration tests

Reviews test_self_rag, test_self_improving_codegen, and
test_semantic_chunking for live API dependencies. Marks any tests
that require real resources as @pytest.mark.integration.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Stage 4 — Requirements Rebuild

### Task 7: Audit actual imports in the package

**Files:**
- Read: all files in `bakkesmod_rag/`, `interactive_rag.py`, `rag_gui.py`

**Step 1: Extract all third-party imports**

Run this to get a sorted list of all third-party module names imported across the package:

```bash
grep -rh "^import\|^from" bakkesmod_rag/ interactive_rag.py rag_gui.py \
  | grep -v "^from \." \
  | sed 's/^import //;s/^from //;s/ .*//' \
  | sort -u \
  | grep -v "^#"
```

**Step 2: Categorize each import**

Map each module name to a pip package name. Cross-reference against the current `requirements.txt`. Note any imports that aren't in requirements (missing) or any requirements that have no matching imports (potentially unused).

Standard library modules to **exclude** from deps: `os`, `sys`, `re`, `json`, `logging`, `pathlib`, `typing`, `dataclasses`, `datetime`, `hashlib`, `time`, `copy`, `functools`, `itertools`, `collections`, `abc`, `io`, `subprocess`, `shutil`, `tempfile`, `threading`, `asyncio`, `inspect`, `traceback`, `warnings`, `contextlib`.

**Step 3: Write the audit results**

Create a temporary note (in your head or a scratch file) listing:
- Core runtime deps (needed for basic query/GUI)
- Dev deps (testing only)
- Optional deps (observability, evaluation, API server, advanced reranking)

---

### Task 8: Write requirements.txt (core only)

**Files:**
- Modify: `requirements.txt`

**Step 1: Replace requirements.txt with core-only content**

Based on the audit from Task 7, write `requirements.txt` with only the packages needed to run the core RAG + GUI. Include a comment block at the top explaining the three-file split.

The content should look like:

```
# Core runtime dependencies — required to run the RAG system and GUI
# For development/testing: pip install -r requirements-dev.txt
# For optional features (observability, evaluation, API): pip install -r requirements-optional.txt

# LlamaIndex core
llama-index-core>=0.14.6
llama-index-embeddings-openai>=0.1.0
llama-index-embeddings-huggingface>=0.1.0
llama-index-retrievers-bm25>=0.1.0
llama-index-postprocessor-cohere-rerank>=0.1.0
llama-index-postprocessor-flag-embedding-reranker>=0.1.0
llama-index-postprocessor-flashrank-rerank>=0.1.0
FlagEmbedding>=1.3.0
tree-sitter-language-pack>=0.6.0
faiss-cpu>=1.7.4
nest-asyncio>=1.5.0
tenacity>=8.0.0
tiktoken>=0.5.0

# LLM Providers
llama-index-llms-google-genai>=0.1.0
llama-index-llms-anthropic>=0.1.0
llama-index-llms-openai>=0.1.0
llama-index-llms-openrouter>=0.4.0
google-genai>=1.33.0
anthropic>=0.25.0
openai>=1.0.0
cohere>=5.0.0

# GUI & Web
gradio>=4.0.0

# Utilities
pandas>=2.0.0
pygments>=2.15.0
colorama>=0.4.6
python-dotenv>=1.0.0
pydantic>=2.0.0

# MCP Integration
mcp>=0.1.0
```

**Step 2: Verify the file is correct**

```bash
cat requirements.txt
```

---

### Task 9: Write requirements-dev.txt

**Files:**
- Create: `requirements-dev.txt`

**Step 1: Create the file**

```
# Development and testing dependencies
# Install with: pip install -r requirements.txt -r requirements-dev.txt

pytest>=7.0.0
pytest-asyncio>=0.21.0
fakeredis>=2.0.0
coverage>=7.0.0
```

---

### Task 10: Write requirements-optional.txt

**Files:**
- Create: `requirements-optional.txt`

**Step 1: Create the file**

```
# Optional dependencies for advanced features
# Observability: pip install arize-phoenix openinference-instrumentation-llama-index prometheus-client opentelemetry-api
# Evaluation: pip install ragas datasets
# Distributed cache: pip install redis
# ColBERT reranking: pip install ragatouille
# REST API server: pip install fastapi uvicorn httpx

# Observability & Monitoring
arize-phoenix>=4.0.0
openinference-instrumentation-llama-index>=2.0.0
prometheus-client>=0.19.0
opentelemetry-api>=1.20.0

# Evaluation & Quality Assurance
ragas>=0.1.0
datasets>=2.0.0

# Distributed Cache
redis>=5.0.0

# Advanced Reranking (ColBERT)
ragatouille>=0.0.8

# REST API Server
fastapi>=0.110.0
uvicorn>=0.27.0
httpx>=0.25.0
```

---

### Task 11: Verify core package imports cleanly

**Step 1: Test that the package imports without optional deps**

```bash
python -c "import bakkesmod_rag; print('OK')"
```

Expected: `OK` (no import errors for optional deps).

**Step 2: If import errors occur, fix them**

Open the failing module. Find the import causing the error. Wrap it in a try/except with a helpful warning:

```python
try:
    import arize_phoenix
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
```

Then guard any usage: `if PHOENIX_AVAILABLE: ...`

Many modules already do this — check `observability.py` and `resilience.py` for examples.

**Step 3: Run smoke tests**

```bash
pytest tests/test_smoke.py -v
```

Expected: All 5 tests PASS.

**Step 4: Commit**

```bash
git add requirements.txt requirements-dev.txt requirements-optional.txt
git commit -m "$(cat <<'EOF'
chore: rebuild requirements as core / dev / optional split

requirements.txt: core runtime only (LlamaIndex, LLM providers, Gradio)
requirements-dev.txt: pytest, fakeredis, coverage
requirements-optional.txt: arize-phoenix, ragas, redis, ragatouille,
fastapi — all features preserved, now explicit about what's optional.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Stage 5 — Code Cleanup

### Task 12: Fix dead imports and bare except clauses

**Files:**
- Modify: any `bakkesmod_rag/*.py` files with issues

**Step 1: Find bare except clauses**

```bash
grep -rn "except:" bakkesmod_rag/ interactive_rag.py rag_gui.py
```

**Step 2: Fix each bare except**

Replace `except:` with `except Exception:`. Never use bare `except:` — it catches `SystemExit` and `KeyboardInterrupt` which is almost never intended.

Before:
```python
try:
    result = risky_call()
except:
    result = None
```

After:
```python
try:
    result = risky_call()
except Exception:
    result = None
```

**Step 3: Find unused imports**

```bash
python -m py_compile bakkesmod_rag/*.py && echo "Syntax OK"
```

For more thorough analysis, if `pyflakes` is available:
```bash
python -m pyflakes bakkesmod_rag/ 2>&1 | grep "imported but unused"
```

If `pyflakes` is not installed, manually scan each module's import block — look for any `import X` where `X` never appears in the file body.

**Step 4: Remove confirmed-unused imports**

Only remove an import if you are 100% sure it is unused in that file. Search for the module name before removing.

**Step 5: Run smoke tests**

```bash
pytest tests/test_smoke.py -v
```

Expected: All 5 tests PASS.

**Step 6: Run full fast suite**

```bash
pytest -m "not integration" -q
```

Expected: All tests PASS.

**Step 7: Commit**

```bash
git add bakkesmod_rag/ interactive_rag.py rag_gui.py
git commit -m "$(cat <<'EOF'
refactor: fix bare except clauses and remove unused imports

Replace bare except: with except Exception: throughout package.
Remove confirmed-unused imports. No logic changes.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Ensure UTF-8 stdout/stderr on Windows

**Files:**
- Modify: `interactive_rag.py` (if not already done)
- Modify: `rag_gui.py` (if not already done)

**Step 1: Check entry points for UTF-8 wrapper**

```bash
grep -n "TextIOWrapper\|utf-8\|encoding" interactive_rag.py rag_gui.py
```

**Step 2: If missing, add UTF-8 wrapping near the top of each entry point**

After the stdlib imports, before any other code:

```python
import sys
import io

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
```

**Step 3: Run smoke tests**

```bash
pytest tests/test_smoke.py -v
```

Expected: All 5 PASS.

**Step 4: Commit if changes were made**

```bash
git add interactive_rag.py rag_gui.py
git commit -m "$(cat <<'EOF'
fix: ensure UTF-8 stdout/stderr encoding on Windows

Wraps stdout/stderr with UTF-8 TextIOWrapper on win32 per project
coding conventions. Prevents UnicodeEncodeError for non-ASCII output.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Stage 6 — Update Developer Documentation

### Task 14: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Verify the module table matches reality**

Open `CLAUDE.md` and find the package module table. Cross-reference every entry against the actual files in `bakkesmod_rag/`:

```bash
ls bakkesmod_rag/*.py | sed 's|bakkesmod_rag/||;s|.py||'
```

Add any modules listed in `ls` but missing from the table. Remove any table entries for files that no longer exist.

**Step 2: Update the Commands section**

Add the three-file requirements install instructions:

```bash
# Install core dependencies
pip install -r requirements.txt

# Install with dev tools (for testing)
pip install -r requirements.txt -r requirements-dev.txt

# Install optional features (observability, evaluation, API server)
pip install -r requirements-optional.txt
```

**Step 3: Remove references to deleted docs**

Search for any links to docs we deleted:

```bash
grep -n "LOGIC_MAP\|2026-gold-standard\|2026-upgrade-summary\|DEEPWIKI\|FEATURE_ENHANCEMENTS\|PROJECT_COMPLETION" CLAUDE.md
```

Remove or update any found references.

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(CLAUDE.md): sync module table, update install commands, remove stale links

Module table now matches actual bakkesmod_rag/ contents.
Install commands reference the new 3-file requirements split.
Removed references to deleted documentation files.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Update README.md and QUICK_START.md

**Files:**
- Modify: `README.md`
- Modify: `QUICK_START.md`

**Step 1: Update README install instructions**

Find the installation section. Update to reference `requirements.txt` (core) and mention `requirements-dev.txt` for contributors.

**Step 2: Update test command in QUICK_START.md**

The default test command should now be:

```bash
# Run fast tests (no API keys needed)
pytest -m "not integration" -v

# Run all tests including integration (requires API keys + built index)
pytest -v
```

**Step 3: Update any doc links that changed in Stage 2**

```bash
grep -n "BUILD_EXE_GUIDE\|EXE_USER_GUIDE\|GUI_USER_GUIDE\|DEBUG_GUIDE\|DEPLOYMENT_GUIDE" README.md QUICK_START.md
```

Update links to point to the new `docs/` paths.

**Step 4: Run final smoke test**

```bash
pytest tests/test_smoke.py -v
pytest -m "not integration" -q
```

Expected: All PASS.

**Step 5: Final commit**

```bash
git add README.md QUICK_START.md
git commit -m "$(cat <<'EOF'
docs: update README and QUICK_START for 3-file requirements and new doc paths

Updated install instructions to reference requirements.txt (core),
requirements-dev.txt (testing), requirements-optional.txt (extras).
Updated pytest command to use -m "not integration" for default run.
Updated doc links to new docs/ paths.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification

After all 15 tasks are complete:

```bash
# 1. Clean working tree
git status

# 2. Fast test suite passes
pytest -m "not integration" -v

# 3. Package imports cleanly
python -c "import bakkesmod_rag; print('RAGEngine:', bakkesmod_rag.RAGEngine)"

# 4. Entry points have valid syntax
python -m py_compile interactive_rag.py rag_gui.py && echo "Entry points OK"

# 5. Docs directory is clean
ls docs/
```

Expected state:
- `git status` → clean
- All non-integration tests PASS
- Package imports without errors
- `docs/` contains only RAG-system-relevant docs
- No SuiteSpot content anywhere in the project
