---
title: Contributing
---

# Contributing to ProbJax

We welcome contributions to ProbJax! This guide will help you get started.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/probjax.git
   cd probjax
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install in development mode**:
   ```bash
   python -m pip install -e ".[dev]"
   ```

## Development Workflow

### Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for code formatting and linting:

```bash
# Format code
ruff format .

# Check for linting issues
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest tests/test_specific.py
```

### Making Changes

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure they follow the code style

3. **Write tests** for new functionality

4. **Run the test suite** to ensure everything passes

5. **Commit your changes** with a clear message:
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## Code Guidelines

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) guidelines
- Use type hints for function signatures
- Write docstrings for all public functions and classes
- Keep functions small and focused

### Docstrings

Use NumPy-style docstrings:

```python
def function(param1: int, param2: str) -> bool:
    """Short description of the function.

    Longer description if needed.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> function(1, "hello")
    True
    """
```

### Testing

- Write tests for all new functionality
- Use pytest fixtures for common setup
- Test edge cases and error conditions
- Aim for high code coverage

## Documentation

### Building Documentation Locally

```bash
cd docs
make html
python -m http.server --directory _build/html
```

### Documentation Style

- Use Markdown for documentation files
- Include code examples where appropriate
- Keep documentation up to date with code changes

## Reporting Issues

When reporting issues, please include:

1. **Description** of the problem
2. **Steps to reproduce** the issue
3. **Expected behavior**
4. **Actual behavior**
5. **Environment details**:
   - Python version
   - JAX version
   - Operating system

## Pull Request Process

1. Ensure your code passes all tests
2. Update documentation if needed
3. Add a clear description of your changes
4. Reference any related issues

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive experience for everyone.

## Questions?

If you have questions about contributing, feel free to:

- Open an issue on GitHub
- Reach out to the maintainers

Thank you for contributing to ProbJax!
