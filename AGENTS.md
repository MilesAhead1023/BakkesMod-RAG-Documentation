# Repository Guidelines

## Project Structure & Module Organization
- `bakkesmod_rag/`: core Python package (engine, retrieval, generation, config, API, observability).
- `tests/`: primary pytest suite (`test_*.py`), including `tests/test_smoke.py` for fast syntax and import checks.
- `docs/`: architecture, setup, deployment, and code-generation documentation.
- `docs_bakkesmod_only/`: BakkesMod SDK reference headers and focused source material.
- `templates/`: C++ plugin template files used by generation workflows.
- Runtime artifacts: `rag_storage/` (index data), `build/`, and `dist/` are generated outputs; do not treat them as source.

## Build, Test, and Development Commands
- `python -m venv venv` then `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix): create and activate environment.
- `pip install -r requirements.txt -r requirements-dev.txt`: install core + dev/test dependencies.
- `pip install -r requirements-optional.txt`: install optional features (observability, evaluation, API server).
- `python -m bakkesmod_rag.sentinel`: validate environment/API key setup.
- `python -m bakkesmod_rag.comprehensive_builder`: build or refresh local RAG indices in `rag_storage/`.
- `python interactive_rag.py`: run CLI interface.
- `python rag_gui.py`: run local web GUI.
- `pytest -m "not integration" -v`: run fast tests (no API keys needed).
- `pytest -v` or `pytest tests/test_cache.py -v`: run all tests (including integration) or a targeted module.

## Coding Style & Naming Conventions
- Follow PEP 8, 4-space indentation, and max line length of 100.
- Use descriptive `snake_case` for functions/modules and `PascalCase` for classes.
- Keep public APIs documented with concise docstrings.
- Prefer small, single-purpose modules inside `bakkesmod_rag/` rather than large mixed files.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio` for async coverage.
- Place tests under `tests/` and name files/functions as `test_*.py` / `test_*`.
- Mock external LLM or network dependencies to keep tests deterministic and low-cost.
- Add or update tests for every behavior change; cover both happy path and failure path.

## Commit & Pull Request Guidelines
- Use Conventional Commit style seen in history: `feat: ...`, `fix: ...`, `refactor: ...`.
- Keep subject lines imperative and concise (about 72 chars max).
- PRs should include: clear summary, why the change is needed, linked issues, and screenshots for UI changes.
- Before opening a PR, run relevant tests locally and update docs when behavior or commands change.
