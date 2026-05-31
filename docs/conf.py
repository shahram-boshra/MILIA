# =============================================================================
# docs/conf.py — MILIA Sphinx Configuration
# =============================================================================
# Sphinx configuration for building MILIA documentation.
#
# Source: Scientific Python Development Guide
#   (learn.scientific-python.org/development/tutorials/docs/)
# Source: pyOpenSci Python Package Guide
#   (pyopensci.org/python-package-guide/documentation/)
# Source: Sphinx 8.1 docs (sphinx-doc.org)
#
# Build instructions:
#   pip install -r docs/requirements.txt   # install doc dependencies
#   sphinx-build -b html docs docs/_build  # build HTML docs
#   # or: make docs                        # via Makefile target
#
# All metadata values sourced from pyproject.toml and
# milia_pipeline/__init__.py (single source of truth).
# =============================================================================

import importlib.metadata
import os
import sys

# -- Path setup ---------------------------------------------------------------
# Add the project root to sys.path so Sphinx autodoc can import milia_pipeline.
# This mirrors the editable install: `pip install -e .` adds project root to
# sys.path. Per Sphinx autodoc docs: "the module or the package must be in one
# of the directories on sys.path."
sys.path.insert(0, os.path.abspath(".."))

# -- Project information ------------------------------------------------------
# Dynamically read from installed package metadata (pyproject.toml is the
# canonical source via setuptools). Falls back to direct import if the package
# is not installed (e.g., during CI doc builds before `pip install`).
try:
    _metadata = importlib.metadata.metadata("milia")
    project = _metadata["Name"]
    author = _metadata["Author-email"].split("<")[0].strip()
    release = _metadata["Version"]
except importlib.metadata.PackageNotFoundError:
    # Fallback for hermetic documentation builds where the package is NOT
    # installed (e.g. Read the Docs, which installs only docs/requirements.txt).
    # Read the __version__ literal straight from the package source WITHOUT
    # importing milia_pipeline: importing it executes import-time code paths
    # (cli_manager -> config) that require the full runtime dependency stack
    # and a configs/ directory, neither of which exists in a docs-only env.
    # pyproject.toml declares the version via `attr = "milia_pipeline.__version__"`,
    # so the __init__.py literal is the single source of truth; we parse that
    # same literal here, keeping the build dynamic and dependency-free.
    import re
    from pathlib import Path

    project = "MILIA"
    author = "Asadollah (Shahram) Boshra"
    _init_path = Path(__file__).resolve().parent.parent / "milia_pipeline" / "__init__.py"
    _version_match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        _init_path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    release = _version_match.group(1) if _version_match else "unknown"

version = ".".join(release.split(".")[:2])  # short X.Y version
copyright = f"2026-present, {author}"  # noqa: A001

# -- General configuration ----------------------------------------------------
extensions = [
    # --- Built-in Sphinx extensions ---
    # autodoc: auto-generate API docs from docstrings
    # Source: Scientific Python Dev Guide, pyOpenSci
    "sphinx.ext.autodoc",
    # autosummary: generate summary tables with links to full docs
    # Source: Scientific Python Dev Guide
    "sphinx.ext.autosummary",
    # napoleon: support NumPy-style docstrings (standard in scientific Python)
    # Source: pyOpenSci — "If you are using NumPy style docstrings, be sure
    # to include the sphinx napoleon extension"
    "sphinx.ext.napoleon",
    # intersphinx: cross-reference to external project documentation
    # (Python, NumPy, PyTorch, PyG, RDKit)
    "sphinx.ext.intersphinx",
    # viewcode: add [source] links to API docs pointing to highlighted source
    "sphinx.ext.viewcode",
    # --- Third-party extensions ---
    # MyST parser: write docs in Markdown instead of reStructuredText
    # Source: Scientific Python Dev Guide — "MyST plugin"
    "myst_parser",
    # sphinx-copybutton: one-click copy for code blocks
    "sphinx_copybutton",
]

# -- Source file configuration ------------------------------------------------
# Support both Markdown (.md via MyST) and reStructuredText (.rst)
# Source: Scientific Python Dev Guide — source_suffix = [".rst", ".md"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The root document (without extension).
root_doc = "index"

# Patterns to exclude from source discovery.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST parser configuration ------------------------------------------------
# Source: myst-parser docs (myst-parser.readthedocs.io)
myst_enable_extensions = [
    "colon_fence",  # ::: directive syntax (cleaner than ``` for nesting)
    "deflist",  # definition lists
    "fieldlist",  # field lists
    "substitution",  # Jinja-like substitutions
]
myst_heading_anchors = 3  # auto-generate anchors for h1–h3

# -- autodoc configuration ----------------------------------------------------
# Source: Sphinx autodoc docs, Scientific Python Dev Guide
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# -- autosummary configuration ------------------------------------------------
autosummary_generate = False  # hermetic docs build: no autosummary directives,
# so no package import/introspection is triggered. API reference is curated
# (see docs/api/index.md). Re-enable only if the package is made import-safe
# (remove the import-time load_config() side effect) AND installed in the build.

# -- napoleon configuration ----------------------------------------------------
# Source: pyOpenSci — NumPy-style docstrings are standard in scientific Python
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_rtype = False  # inline return type with description

# -- intersphinx configuration ------------------------------------------------
# Source: Sphinx intersphinx docs — cross-reference external projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/stable/", None),
}

# -- HTML output configuration ------------------------------------------------
# Theme: pydata-sphinx-theme (standard in scientific Python ecosystem)
# Source: pyOpenSci — "the most common Sphinx themes used in the Python
# scientific community include pydata-sphinx-theme"
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/shahram-boshra/MILIA",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "shahram-boshra",
    "github_repo": "MILIA",
    "github_version": "main",
    "doc_path": "docs",
}

html_static_path = ["_static"]

# -- Options for sphinx-copybutton --------------------------------------------
# Strip common shell prompts from copied code blocks
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
