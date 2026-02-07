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

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

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
- API keys for at least one LLM provider (OpenAI, Gemini, or Anthropic)

### Environment Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   GEMINI_API_KEY=your_gemini_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   ```

4. **Build the RAG index** (required for testing):
   ```bash
   python rag_builder.py
   ```

For more detailed setup instructions, see [DEVELOPMENT.md](DEVELOPMENT.md).

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

Run all tests before submitting a pull request:

```bash
# Run smoke tests
pytest test_smoke.py -v

# Run integration tests
pytest test_rag_integration.py -v

# Run specific test
pytest test_smoke.py::test_query_with_cache -v
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
   pytest test_smoke.py test_rag_integration.py -v
   ```

3. **Check code style**:
   ```bash
   # Format code (if using black)
   black *.py
   
   # Check for issues
   flake8 *.py
   ```

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
