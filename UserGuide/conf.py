# Configuration file for the Sphinx documentation builder.
#

# -- Project information -----------------------------------------------------

project = 'MindSpeed-MM'
release = 'v1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

myst_enable_extensions = [
    "tasklist",
    "deflist",
    "dollarmath",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_css_files = [
    'width.css',
]

html_static_path = ['_static']
