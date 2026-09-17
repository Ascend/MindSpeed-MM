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

# 把 ```mermaid 围栏交给 sphinxcontrib.mermaid 渲染，否则会退化成普通代码块
myst_fence_as_directive = ["mermaid"]

# 为标题生成 GitHub 风格锚点，使文档内 #章节名 链接在 RTD 上同样可跳转
myst_heading_anchors = 4

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'requirements.txt']


language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_css_files = [
    'width.css',
]

html_static_path = ['_static']
