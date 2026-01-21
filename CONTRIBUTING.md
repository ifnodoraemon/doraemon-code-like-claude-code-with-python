# Contributing to Polymath

Thank you for your interest in contributing to Polymath! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/polymath-ai/polymath.git
   cd polymath
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up pre-commit hooks** (optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_specific.py

# Run with verbose output
pytest -v
```

## Code Style

We use:
- **Ruff** for linting and formatting
- **MyPy** for type checking
- **Black-compatible** formatting (via Ruff)

Before submitting a PR:
```bash
# Format code
ruff format src/ tests/

# Check linting
ruff check src/ tests/

# Type check
mypy src/
```

## Submitting Changes

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Update documentation as needed
   - Follow existing code style

3. **Test your changes**
   ```bash
   pytest
   ruff check src/
   mypy src/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```
   
   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for tests
   - `refactor:` for code refactoring
   - `chore:` for maintenance tasks

5. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Pull Request Guidelines

- Fill out the PR template completely
- Link related issues
- Ensure all CI checks pass
- Request review from maintainers
- Be responsive to feedback

## Reporting Issues

When reporting bugs, please include:
- Polymath version
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

## Questions?

Feel free to open a Discussion on GitHub or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
