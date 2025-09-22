import os
import re
import sys
from pathlib import Path


# -- Path setup --------------------------------------------------------------

# Add the project's src directory to sys.path so autodoc can import modules
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


# -- Project information -----------------------------------------------------


def _read_pyproject_metadata(pyproject_path: Path):
    """Extract minimal metadata from pyproject.toml without external deps.

    Returns a dict with keys: name, version, description.
    """
    text = pyproject_path.read_text(encoding="utf-8")

    # Very small, tolerant regex extraction
    def _get(pattern, default=""):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else default

    name = _get(r"^name\s*=\s*\"([^\"]+)\"", "record")
    version = _get(r"^version\s*=\s*\"([^\"]+)\"", "2.1.0")
    description = _get(r"^description\s*=\s*\"([^\"]+)\"", "Project documentation")
    return {"name": name, "version": version, "description": description}


_meta = _read_pyproject_metadata(ROOT / "pyproject.toml")

project = _meta["name"] or "record"
copyright = "2025"
author = "RECORD project contributors"

# The short X.Y version and the full version, including alpha/beta/rc tags
version = _meta["version"]
release = _meta["version"]


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Mock heavy/optional GUI and visualization deps during autodoc imports
autodoc_mock_imports = [
    "PyQt6",
    "pyqtgraph",
    "vtk",
    "cv2",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

language = "en"


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]

# Example theme options (tweak or remove as desired)
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
