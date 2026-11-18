"""Sphinx configuration (pydata-sphinx-theme, as in scbirlab/chemprop-accel)."""

import os
from importlib import metadata

repo = os.environ.get("GITHUB_REPOSITORY", "scbirlab/unimol-accel")
owner, name = repo.split("/", 1)

project = "unimol-accel"
author = "Eachan Johnson"
copyright = "2026, Eachan Johnson"
try:
    release = metadata.version("unimol-accel")
except metadata.PackageNotFoundError:  # pragma: no cover
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath", "attrs_inline"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
suppress_warnings = ["myst.header"]
nitpicky = False

html_theme = "pydata_sphinx_theme"
html_title = f"unimol-accel {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "github_url": f"https://github.com/{repo}",
    "icon_links": [
        {
            "name": "Benchmark report",
            "url": f"https://{owner}.github.io/{name}/benchmarks/report/",
            "icon": "fa-solid fa-gauge-high",
            "type": "fontawesome",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
    "use_edit_page_button": True,
}
html_context = {
    "github_user": owner,
    "github_repo": name,
    "github_version": "main",
    "doc_path": "docs",
}
