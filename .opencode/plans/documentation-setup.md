# ProbJax Documentation Website Setup Plan

## Overview
Set up a Sphinx-based documentation website for ProbJax similar to Jaxley's ReadTheDocs structure, starting with a minimal but expandable foundation.

## Goals
1. Create a professional documentation website using Sphinx
2. Set up ReadTheDocs integration for automatic builds
3. Start with minimal sections: Installation, Quick Start, API Reference
4. Link to existing Jupyter notebooks in `examples/` directory
5. Ensure documentation is maintainable and expandable

## Current State Analysis
- **Codebase**: 216 Python files, ~62k lines of code
- **Existing Documentation**: Only README.md with basic information
- **Examples**: Jupyter notebooks in `examples/` directory (6 subdirectories)
- **Docstrings**: Partial coverage, especially in `probjax.stats` module
- **Package Structure**: Well-organized with core, inference, nn, stats, utils modules

## Implementation Plan

### Phase 1: Basic Documentation Setup

#### 1.1 Create Documentation Directory Structure
```
docs/
├── _static/           # Static files (CSS, images)
├── _templates/        # Custom templates
├── conf.py           # Sphinx configuration
├── index.rst          # Main documentation page
├── installation.md    # Installation instructions
├── quickstart.md      # Quick start guide
├── api/              # API reference
│   ├── index.rst
│   ├── core.rst
│   ├── inference.rst
│   ├── nn.rst
│   ├── stats.rst
│   └── utils.rst
├── examples.rst      # Links to examples
├── contributing.md   # Contribution guidelines
├── Makefile          # Build commands
└── make.bat          # Windows build commands
```

#### 1.2 Configure Sphinx (conf.py)
Based on Jaxley's configuration:
- Use `sphinx_book_theme` for consistent look
- Enable extensions: autodoc, autosummary, napoleon, intersphinx, viewcode
- Support for MyST markdown (myst_nb extension)
- Configure autodoc for automatic API documentation
- Set up intersphinx linking to JAX, NumPy, Python docs

#### 1.3 Create Main Documentation Files
1. **index.rst**: Landing page with project overview, quick start, and navigation
2. **installation.md**: Detailed installation instructions (from README.md)
3. **quickstart.md**: Basic usage examples (from README.md)
4. **api/index.rst**: API reference overview
5. **api/*.rst**: Module-specific API documentation
6. **examples.rst**: Links to Jupyter notebooks in `examples/` directory
7. **contributing.md**: Contribution guidelines

#### 1.4 Set Up ReadTheDocs Integration
1. Create `.readthedocs.yaml` configuration file
2. Specify Python version (3.11)
3. Configure build dependencies
4. Set up automatic builds on push to main

### Phase 2: API Documentation Enhancement

#### 2.1 Improve Docstring Coverage
- Review and enhance docstrings in key modules
- Ensure all public functions have proper documentation
- Follow NumPy/Google docstring style

#### 2.2 Configure Autodoc Settings
- Set up `autosummary_generate = True`
- Configure `autodoc_typehints = "description"`
- Create module-specific RST files for better organization

### Phase 3: Build and Testing

#### 3.1 Local Build Testing
- Test Sphinx build with `make html`
- Verify all links work correctly
- Check API documentation generation

#### 3.2 ReadTheDocs Deployment
- Push to repository
- Verify automatic builds
- Test on readthedocs.io

## Key Files to Create

### 1. `docs/conf.py`
```python
# Sphinx configuration for ProbJax documentation
project = "ProbJax"
author = "Manuel Gloeckler"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_nb",
]
html_theme = "sphinx_book_theme"
# Additional configuration...
```

### 2. `.readthedocs.yaml`
```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
sphinx:
  configuration: docs/conf.py
python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .
```

### 3. `docs/requirements.txt`
```
sphinx>=7.0
sphinx-book-theme>=1.0
myst-nb>=1.0
sphinx-design
sphinx-math-dollar
```

## Dependencies to Add
- Sphinx and related packages
- ReadTheDocs configuration
- Documentation-specific requirements

## Success Criteria
1. Documentation builds successfully with `make html`
2. API documentation is automatically generated
3. ReadTheDocs deployment works
4. Basic documentation structure is in place
5. Easy to expand with more tutorials and examples

## Future Expansion
1. Add more tutorials (convert existing notebooks)
2. Add advanced tutorials section
3. Add how-to guides
4. Add FAQ section
5. Add changelog and release notes

## Timeline
- Phase 1: 2-3 hours (basic setup)
- Phase 2: 1-2 hours (API enhancement)
- Phase 3: 1 hour (testing and deployment)

## Maintenance
- Documentation updates with code changes
- Regular docstring improvements
- Tutorial additions as new features are developed
