# Contributing to BakkesMod RAG Documentation

Thank you for your interest in contributing to the BakkesMod RAG Documentation system! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please ensure you are familiar with and follow these standards in all project interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/BakkesMod-RAG-Documentation.git
   cd BakkesMod-RAG-Documentation
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip and virtualenv
- Git
- API keys for all three LLM providers (OpenAI, Anthropic, and Google/Gemini)

### Environment Setup

1. **Create a virtual environment**:
   
   **Windows:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```
   
   **Linux/Mac:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   GOOGLE_API_KEY=your_gemini_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   ```
   
   The Python scripts read these values from environment variables via `os.environ` and do **not**
   automatically load the `.env` file. Before running any Python scripts, make sure these
   variables are exported in your shell.
   
   **Windows PowerShell:**
   ```powershell
   Get-Content .env | ForEach-Object {
       if ($_ -match '^\s*#' -or -not $_) { return }
       $name, $value = $_ -split '=', 2
       [Environment]::SetEnvironmentVariable($name, $value, 'Process')
   }

   python interactive_rag.py
   ```
   
   **Linux/Mac:**
   ```bash
   # load variables from .env into the current shell
   export $(grep -v '^#' .env | xargs)

   # now run your script
   python interactive_rag.py
   ```

4. **Build the RAG index** (required for testing):
   The integration tests and sentinel checks expect RAG indices to be available under
   the `./rag_storage/` directory. Use the comprehensive builder to build the indices:
   ```bash
   python -m bakkesmod_rag.comprehensive_builder  # builds indices in ./rag_storage/
   ```

For more detailed setup instructions, see [docs/rag-setup.md](docs/rag-setup.md).

## Making Changes

### Branching Strategy

- Create a new branch for each feature or bugfix
- Use descriptive branch names:
  - `feature/add-new-retrieval-method`
  - `bugfix/fix-cache-invalidation`
  - `docs/update-setup-guide`

```bash
git checkout -b feature/your-feature-name
```

### Commit Messages

Write clear, concise commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

Examples:
```
Add support for custom embedding models

- Implement HuggingFace model loading
- Add configuration options
- Update documentation

Fixes #123
```

## Testing

### Running Tests

Install core + development dependencies:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

Run all tests before submitting a pull request:

```bash
# Run all tests
pytest tests/ -v

# Run tests in a specific file
pytest tests/test_cache.py -v

# Run specific test
pytest tests/test_cache.py::test_cache_hit -v
```

### Writing Tests

- Add tests for new features
- Update existing tests when modifying functionality
- Ensure tests are deterministic and don't rely on external state
- Mock API calls to avoid unnecessary costs during testing

## Submitting Changes

### Before Submitting

1. **Update your branch** with the latest changes from upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all tests**:
   ```bash
   pytest tests/ -v
   ```

3. **Check code style**:
   Make sure your changes follow the project's [Style Guidelines](#style-guidelines)
   (PEP 8, docstrings, line length, etc.). You may use your preferred formatter
   and linter locally, but the project does not require any specific tooling.

### Creating a Pull Request

1. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub with:
   - A clear title describing the change
   - A detailed description of what changed and why
   - References to related issues
   - Screenshots (if applicable for UI changes)

3. **Respond to feedback** and make requested changes

4. **Keep your PR up to date** with the main branch

## Style Guidelines

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and single-purpose
- Maximum line length: 100 characters

### Documentation Style

- Use Markdown for all documentation
- Keep documentation up-to-date with code changes
- Include code examples where helpful
- Use clear, concise language

### Code Organization

- Keep related functionality together
- Separate concerns into different modules
- Use descriptive file names
- Add comments for complex logic

## Areas for Contribution

We welcome contributions in the following areas:

- **Features**: New retrieval strategies, LLM integrations, caching improvements
- **Bug Fixes**: Fix issues reported in the issue tracker
- **Documentation**: Improve guides, add examples, fix typos
- **Testing**: Increase test coverage, add integration tests
- **Performance**: Optimize query speed, reduce API costs
- **Evaluation**: Improve RAG quality metrics and evaluation tools

## Questions?

If you have questions or need help:

- Check the [documentation](docs/)
- Open a [discussion](https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation/discussions)
- Ask in an issue or pull request

Thank you for contributing! 🚀
