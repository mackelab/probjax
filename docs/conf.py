# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to sys.path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "ProbJax"
copyright = "2024, Manuel Gloeckler"
author = "Manuel Gloeckler"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_nb",
]

# Support for both reStructuredText and MyST markdown
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
add_module_names = False

# -- Options for autosummary ------------------------------------------------

autosummary_generate = True

# -- Options for napoleon ----------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "flax": ("https://flax.readthedocs.io/en/latest", None),
}

# -- Options for MyST-NB -----------------------------------------------------

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "tasklist",
]

# Notebook execution settings
nb_execution_mode = "off"  # Use pre-existing outputs (don't execute)
nb_execution_timeout = 120  # 2 minute timeout if enabled
nb_execution_raise_on_error = False
nb_execution_allow_errors = True

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "ProbJax"
html_logo = None  # Add logo path if available

html_theme_options = {
    "repository_url": "https://github.com/probjax/probjax",
    "use_repository_button": True,
    "use_download_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "toc_title": "Navigation",
    "show_navbar_depth": 2,
    "show_toc_level": 3,
    "pygment_dark_style": "github-dark",
    "logo": {
        "text": "ProbJax",
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom CSS files
html_css_files = []

# -- Additional configuration ------------------------------------------------

# The master toctree document
master_doc = "index"
