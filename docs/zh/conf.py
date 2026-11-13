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


# -- GitHub 风格提示块 --------------------------------------------------------
# 文档中的提示块统一采用 GitHub 语法（> [!NOTE] 等），该写法在代码托管平台上原生
# 渲染为提示框，但 MyST 不识别，会把标记按字面量输出。此处补上解析，使其在生成的
# 站点上同样渲染为提示框，文档源文件无需改写。

from docutils import nodes
from docutils.transforms import Transform

GITHUB_ALERTS = {
    'NOTE': nodes.note,
    'TIP': nodes.tip,
    'IMPORTANT': nodes.important,
    'WARNING': nodes.warning,
    'CAUTION': nodes.caution,
}


class GithubAlertTransform(Transform):
    """把首段为 [!NOTE] 一类标记的引用块转换为对应的提示框。"""

    default_priority = 500

    def apply(self):
        for quote in list(self.document.findall(nodes.block_quote)):
            if not quote.children or not isinstance(quote.children[0], nodes.paragraph):
                continue
            text = quote.children[0].astext().strip()
            for tag, node_class in GITHUB_ALERTS.items():
                marker = '[!%s]' % tag
                if text != marker and not text.startswith(marker + '\n'):
                    continue
                alert = node_class()
                body = list(quote.children[1:])
                remainder = text[len(marker):].lstrip('\n')
                if remainder:
                    body.insert(0, nodes.paragraph(text=remainder))
                alert += body
                quote.replace_self(alert)
                break


def setup(app):
    app.add_transform(GithubAlertTransform)
    return {'parallel_read_safe': True}
