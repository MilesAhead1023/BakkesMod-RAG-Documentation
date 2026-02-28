# Tech Debt Cleanup & Project Reorganization — Design

**Date:** 2026-02-28
**Branch:** feb-work
**Approach:** Surgical Sweep (Approach A)

## Summary

Full 6-stage cleanup of the BakkesMod RAG Documentation project:
remove obsolete docs/tests/plans, prune the test suite, rebuild the
dependency toolchain from scratch (preserving all features), light code
cleanup, and update developer-facing documentation.

---

## Stage 1 — Git Hygiene

**Goal:** Commit all pending deletions/modifications as a clean baseline.

Files being committed as deleted/modified (already in working tree):
- `.gitignore`, `CLAUDE.md`, `CONTRIBUTING.md`, `Dockerfile` — modified
- `QUICK_START.md`, `README.md` — modified
- `bakkesmod_rag_gui.spec` — modified
- `docs/deployment-guide.md` — modified
- `rag_gui.py`, `requirements.txt` — modified
- `DEEPWIKI_COMPARISON.md` — deleted
- `FEATURE_ENHANCEMENTS.md` — deleted
- `PROJECT_COMPLETION_SUMMARY.md` — deleted
- `TEST_RESULTS.md` — deleted
- `bakkesmod_rag_gui_minimal.spec` — deleted
- `docs/bakkesmod-sdk-reference.md` — deleted
- `docs/bakkesmod_imgui_signatures_annotated.md` — deleted
- `docs/plans/2026-02-05-fix-gemini-rag-stack.md` — deleted
- `docs/plans/2026-02-07-code-generation-mode.md` — deleted
- `docs/plans/2026-02-07-rag-phase1-enhancements.md` — deleted
- `docs/plans/2026-02-07-rag-phase2-enhancements.md` — deleted
- `test_smoke.py` — deleted (root-level, superseded by tests/test_smoke.py)
- `test_smoke.spec` — deleted

**Success criteria:** `git status` shows clean working tree after commit.

---

## Stage 2 — Docs Cleanup

**Goal:** Remove misplaced and outdated documentation.

**Delete:**
- `docs/LOGIC_MAP.md` — describes "SuiteSpot" plugin, wrong project
- `docs/architecture.md` — describes "SuiteSpot" plugin, wrong project
- `docs/2026-gold-standard-architecture.md` — old planning doc, content captured in CLAUDE.md
- `docs/2026-upgrade-summary.md` — old planning doc, content captured in CLAUDE.md

**Consolidate root-level guides into docs/:**
- Move `BUILD_EXE_GUIDE.md` → `docs/build-exe-guide.md`
- Move `EXE_USER_GUIDE.md` → `docs/exe-user-guide.md`
- Move `GUI_USER_GUIDE.md` → `docs/gui-user-guide.md`
- Move `DEBUG_GUIDE.md` → `docs/debug-guide.md`
- `DEPLOYMENT_GUIDE.md` → merge/replace `docs/deployment-guide.md`
- Update any cross-references in README.md and QUICK_START.md

**Keep:**
- `docs/bakkesmod-sdk-guides.md`
- `docs/CODE_GENERATION_GUIDE.md`
- `docs/deployment-guide.md` (updated)
- `docs/rag-setup.md`

**Success criteria:** `docs/` only contains RAG-system-relevant documentation. No SuiteSpot content remains.

---

## Stage 3 — Test Suite Pruning

**Goal:** Keep a fast, reliable test suite. Remove tests that require live API keys or a built index.

**Keep:**
- `tests/test_smoke.py` — syntax + import validation, config defaults, query rewriter
- `tests/conftest.py` — shared fixtures (MockLLM, MockEmbedModel, etc.)
- All unit tests that use only mocks (no live API calls):
  - test_adaptive_retrieval, test_answer_verifier, test_api, test_cache,
    test_circuit_breaker, test_code_generator, test_code_validator,
    test_compiler, test_confidence, test_config, test_cost_tracker,
    test_cpp_analyzer, test_feedback_store, test_guardrails, test_gui,
    test_intent_router, test_observability, test_query_decomposer,
    test_query_rewriter, test_setup_keys, test_template_engine

**Review/prune:**
- `tests/test_golden_queries.py` — requires live index; mark as `@pytest.mark.integration`
- `tests/test_engine_integration.py` — requires live index; mark as `@pytest.mark.integration`
- `tests/test_code_gen_integration.py` — requires live LLM; mark as `@pytest.mark.integration`
- `tests/test_self_rag.py` — review for mock coverage
- `tests/test_self_improving_codegen.py` — review for mock coverage
- `tests/test_semantic_chunking.py` — review for mock coverage

**pytest.ini** — already configured with `integration` and `slow` markers. Ensure default run excludes integration tests.

**Success criteria:** `pytest -m "not integration"` passes with no API keys in environment.

---

## Stage 4 — Requirements Rebuild

**Goal:** Rebuild requirements from actual import analysis, split into three files.

**Method:**
1. Scan all 27 `bakkesmod_rag/` modules + both entry points for `import` statements
2. Map stdlib vs third-party vs local
3. Derive minimum required packages
4. Split into:

**`requirements.txt`** (core runtime — must install to use):
- llama-index-core, llama-index-embeddings-openai, llama-index-embeddings-huggingface
- llama-index-retrievers-bm25
- llama-index-postprocessor-cohere-rerank, llama-index-postprocessor-flag-embedding-reranker, llama-index-postprocessor-flashrank-rerank
- llama-index-llms-google-genai, llama-index-llms-anthropic, llama-index-llms-openai, llama-index-llms-openrouter
- FlagEmbedding, tree-sitter-language-pack, faiss-cpu
- nest-asyncio, tenacity, tiktoken
- google-genai, anthropic, openai, cohere
- gradio, pandas, pygments, colorama, python-dotenv, pydantic

**`requirements-dev.txt`** (development/testing):
- pytest, pytest-asyncio, fakeredis, coverage

**`requirements-optional.txt`** (advanced features):
- arize-phoenix, openinference-instrumentation-llama-index (observability)
- prometheus-client, opentelemetry-api (metrics)
- mcp (MCP server integration)
- ragas, datasets (RAG evaluation)
- redis (distributed cache)
- ragatouille (ColBERT reranking)
- fastapi, uvicorn, httpx (REST API server)

**Success criteria:** `pip install -r requirements.txt` succeeds and all core features work.

---

## Stage 5 — Code Cleanup

**Goal:** Light housekeeping only — no feature changes.

- Remove dead/unused imports in each module
- Fix bare `except:` clauses → `except Exception:`
- Ensure stdout/stderr use UTF-8 TextIOWrapper on Windows (per CLAUDE.md)
- Remove any `# TODO`, `# FIXME`, or `# HACK` comments that reference removed features
- No refactoring of logic, no API changes

**Success criteria:** All linting passes, no functional changes, tests still pass.

---

## Stage 6 — Update CLAUDE.md + README

**Goal:** Developer documentation reflects current state.

- Update `CLAUDE.md` package module table (add new modules, verify all listed files exist)
- Update `requirements.txt` install instructions to mention the three-file split
- Update `README.md` with accurate quick-start instructions
- Update `QUICK_START.md` to reflect the pruned test command (`pytest -m "not integration"`)

**Success criteria:** A new contributor can follow CLAUDE.md and README to set up and run the system correctly.

---

## Implementation Order

```
Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6
```

Each stage ends with a git commit so changes are isolated and reviewable.

## Risk Mitigation

- Never delete a module file without checking all import sites first
- Mark integration tests rather than deleting them — they are valuable for manual validation
- Test suite must pass (non-integration) before each commit
- Requirements rebuild: verify with `python -c "import bakkesmod_rag"` after each install
