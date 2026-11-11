"""Sphinx configuration for the ManyPred documentation (built by Read the Docs)."""

from manypred import __version__

project = "ManyPred"
author = "DPIRD-DMA"
version = release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "undoc-members": False}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True  # "Attributes:" as fields, not objects autodoc lists again
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

html_theme = "sphinx_rtd_theme"
html_title = f"ManyPred {release}"
html_logo = "assets/logo-dark.svg"
html_favicon = "assets/icon-light.svg"
# the theme's blue header hides the logo's teal and navy
html_theme_options = {"logo_only": True, "style_nav_header_background": "#0F172A"}
html_show_copyright = False
