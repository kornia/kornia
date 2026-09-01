# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import builtins
import html
import importlib.util
import inspect
import os
import re
import sys
from datetime import UTC, datetime

# Monkey-patch for PyTorch compatibility with sphinx_autodoc_typehints
# Newer versions of PyTorch removed torch.jit.annotations.compiler_flag
import torch.jit.annotations

if not hasattr(torch.jit.annotations, "compiler_flag"):
    torch.jit.annotations.compiler_flag = None

# Let the library know it is being imported by the Sphinx build.
builtins.__sphinx_build__ = True

# --- Patch sphinx_autodoc_defaultargs to not crash on torchscript/pybind11 callables ---
try:
    import sphinx_autodoc_defaultargs

    _orig_process_docstring = sphinx_autodoc_defaultargs.process_docstring

    def _safe_process_docstring(app, what, name, obj, options, lines):
        try:
            return _orig_process_docstring(app, what, name, obj, options, lines)
        except ValueError as e:
            msg = str(e).lower()
            if "no signature found for builtin" in msg or "pybind11" in msg:
                return  # leave docstring unchanged
            raise

    sphinx_autodoc_defaultargs.process_docstring = _safe_process_docstring

except (ModuleNotFoundError, ImportError):
    # Optional dependency not installed in some environments.
    sphinx_autodoc_defaultargs = None

except AttributeError:
    # Extension API changed; don't patch.
    sphinx_autodoc_defaultargs = None


# ``generate_examples.main`` builds four pretrained models (KeyNet, DISK, ALIKED
# and XFeat), so this build fetches checkpoints from the same rate-limited hosts
# the test jobs do. CI restores the shared ``weights/`` cache for this job, but
# nothing here is running under ``conftest.py``, which is what points torch at
# it for the test and doctest runs -- so point at it here too, or the cache is
# restored and then ignored. Sphinx executes this file with the working
# directory set to its own folder, hence the path off ``__file__``.
import torch.hub  # noqa: E402

_weights_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "weights")
# Only when there is a restored cache to use and the developer has not pointed
# torch somewhere themselves: a local ``pixi run build-docs`` in a fresh clone has
# neither, and redirecting it would re-download all four checkpoints past a warm
# ``~/.cache/torch/hub``.
if os.path.isdir(_weights_cache) and not os.environ.get("TORCH_HOME"):
    torch.hub.set_dir(_weights_cache)

# readthedocs generated the whole documentation in an isolated environment
# by cloning the git repo. Thus, any on-the-fly operation will not effect
# on the resulting documentation. We therefore need to import and run the
# corresponding code here.
spec = importlib.util.spec_from_file_location("generate_examples", "../generate_examples.py")
generate_examples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_examples)

# Pre-generate the example images
generate_examples.main()

spec = importlib.util.spec_from_file_location("generate_benchmarks", "../generate_benchmarks.py")
generate_benchmarks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_benchmarks)

# Pre-generate the benchmark results page
generate_benchmarks.main()

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
current_path = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.append(current_path)

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_autodoc_defaultargs",
    "sphinx_copybutton",
    "sphinx.ext.linkcode",
    "sphinx.ext.githubpages",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.gtagjs",
    "sphinxcontrib.youtube",
    "sphinx_design",
    "notfound.extension",
]

# substitutes the default values
docstring_default_arg_substitution = "Default: "
autodoc_preserve_defaults = True

bibtex_bibfiles = ["references.bib"]
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = [".rst", ".ipynb"]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "Kornia"
author = f"{project} developers"
copyright = f"{datetime.now(tz=UTC).year}, {author}"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# version = 'master (' + kornia.__version__ + ' )'
version = ""

if "READTHEDOCS" not in os.environ:
    # if developing locally, use kornia.__version__ as version
    from kornia import __version__

    version = __version__

# release = 'master'
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", ".ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"
pygments_dark_style = "monokai"

# Prototype switch between the furo layout and a pydata-sphinx-theme layout with a top navbar
# (Guide / API / Models / Community); ``KORNIA_DOCS_THEME=pydata`` selects the latter.
DOCS_THEME = os.environ.get("KORNIA_DOCS_THEME", "furo")

html_theme = "pydata_sphinx_theme" if DOCS_THEME == "pydata" else "furo"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# TODO(jian): make to work with https://docs.kornia.org
html_baseurl = "https://kornia.readthedocs.io/en/latest/"

# Git ref that the "view/edit source" links and the ``linkcode`` extension point at.
rtd_version = os.environ.get("READTHEDOCS_VERSION")
if rtd_version and rtd_version not in {"latest", "stable"}:
    code_ref = rtd_version
else:
    code_ref = "main"

# Changing sidebar title to Kornia
html_title = "Kornia"

_GITHUB_ICON_SVG = (
    '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
    '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
    "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 "
    "1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 "
    "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 "
    "2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 "
    "3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 "
    '8c0-4.42-3.58-8-8-8z"></path></svg>'
)

_FURO_THEME_OPTIONS = {
    # 'analytics_id': 'G-RKS4WFXVHJ', # Unsupported by furo theme
    "light_logo": "img/kornia_logo_only_light.svg",
    "dark_logo": "img/kornia_logo_only_dark.svg",
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    # "View source" / "Edit on GitHub" buttons at the top of every page: the cheapest
    # way to turn a reader who spotted a typo into a contributor.
    "source_repository": "https://github.com/kornia/kornia/",
    "source_branch": code_ref,
    "source_directory": "docs/source/",
    "top_of_page_buttons": ["view", "edit"],
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/kornia/kornia",
            "html": _GITHUB_ICON_SVG,
            "class": "",
        },
    ],
    "light_css_variables": {
        "color-sidebar-background": "#3980F5",
        "color-sidebar-background-border": "#3980F5",
        "color-sidebar-caption-text": "white",
        "color-sidebar-link-text--top-level": "white",
        "color-sidebar-link-text": "white",
        "sidebar-caption-font-size": "normal",
        "color-sidebar-item-background--hover": " #5dade2",
    },
    "dark_css_variables": {
        "color-sidebar-background": "#1a1c1e",
        "color-sidebar-background-border": "#1a1c1e",
        "color-sidebar-caption-text": "white",
        "color-sidebar-link-text--top-level": "white",
    },
    # "announcement": """
    #     <a style=\"text-decoration: none; color: white;\"
    #        href=\"https://github.com/kornia/kornia\">
    #        <img src=\"https://github.com/kornia/data/raw/main/GitHub-Mark-Light-32px.png\" width=20 height=20/>
    #        Star Kornia on GitHub
    #     </a>
    # """,
}

_PYDATA_THEME_OPTIONS = {
    "logo": {
        "image_light": "_static/img/kornia_logo_only_light.svg",
        "image_dark": "_static/img/kornia_logo_only_dark.svg",
        "text": "Kornia",
    },
    # Top navbar: one entry per top-level toctree item of index.rst (Guide / API / Models / Community),
    # plus the external tutorials site. Each section gets its own left sidebar.
    # Navbar: logo + search on the left, flexible space, then the nav links,
    # theme switcher and icon links on the right.
    "navbar_align": "right",
    "navbar_start": ["navbar-logo", "search-button-field"],
    "navbar_center": [],
    "navbar_end": ["navbar-nav", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
    "header_links_before_dropdown": 6,
    "icon_links": [
        {"name": "GitHub", "url": "https://github.com/kornia/kornia", "icon": "fa-brands fa-github"},
        {"name": "Discord", "url": "https://discord.gg/HfnywwpBnD", "icon": "fa-brands fa-discord"},
        {"name": "Twitter", "url": "https://twitter.com/kornia_foss", "icon": "fa-brands fa-x-twitter"},
    ],
    "use_edit_page_button": True,
    "show_nav_level": 1,
    "navigation_depth": 4,
    "show_toc_level": 2,
    "collapse_navigation": False,
    # Right sidebar: page TOC and edit links, with the sponsor box (docs/source/_templates/sponsors.html)
    # below. The landing page is a designed hero page and carries no right rail at all.
    "secondary_sidebar_items": {
        "index": [],
        "**": ["page-toc", "edit-this-page", "sourcelink", "sponsors"],
    },
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
    "footer_start": [],
    "footer_end": [],
}

html_theme_options = _PYDATA_THEME_OPTIONS if DOCS_THEME == "pydata" else _FURO_THEME_OPTIONS

if DOCS_THEME == "pydata":
    # Feeds the "Edit this page" button and the source links.
    html_context = {
        "github_user": "kornia",
        "github_repo": "kornia",
        "github_version": code_ref,
        "doc_path": "docs/source",
    }
    # The landing page has its own card grid; a section sidebar next to it would be empty noise.
    html_sidebars = {"index": []}

# html_logo = '_static/img/kornia_logo.svg'
# html_logo = '_static/img/kornia_logo_only.png'
html_favicon = "_static/img/kornia_logo_favicon.png"

# Show the build date in the footer so readers can tell a stale mirror from the live docs.
html_last_updated_fmt = "%b %d, %Y"

# sphinx-copybutton: strip ``>>>`` / ``...`` / ``$`` prompts and skip output lines, so the
# clipboard receives runnable code instead of a transcript.
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_line_continuation_character = "\\"

# Config the `sphinxcontrib.gtagjs` extension
# NOTE: if this didn't work, we can remove the extension itself
gtagjs_ids = [
    "G-YSCFZB2WDV",  # Shouldn't be necessary if the readthedocs autoinjection work
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_extra_path = ["_extra"]

# Output file base name for HTML help builder.
htmlhelp_basename = "Kornia"
html_css_files = ["css/pydata.css" if DOCS_THEME == "pydata" else "css/main.css"]
html_js_files = ["js/custom.js"]

# Configure viewcode extension.
# based on https://github.com/readthedocs/sphinx-autoapi/issues/202
code_url = f"https://github.com/kornia/kornia/blob/{code_ref}"


def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    modname = info.get("module")
    fullname = info.get("fullname")
    if not modname or not fullname:
        return None

    try:
        mod = importlib.import_module(modname)
    except (ImportError, ModuleNotFoundError):
        return None

    obj = mod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    obj = inspect.unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
        src, start = inspect.getsourcelines(obj)
    except (TypeError, OSError, ValueError):
        return None

    if not fn:
        return None

    fn = os.path.abspath(fn).replace("\\", "/")
    marker = "/kornia/"
    idx = fn.rfind(marker)
    if idx == -1:
        return None

    file_rel = fn[idx + 1 :]  # -> "kornia/....py"
    end = start + len(src) - 1
    return f"{code_url}/{file_rel}#L{start}-L{end}"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [(master_doc, "kornia.tex", "Kornia", "manual")]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "Kornia", "Kornia Documentation", [author], 1)]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "kornia",
        "Kornia Documentation",
        author,
        "Kornia",
        "Differentiable Computer Vision in Pytorch.",
        "Miscellaneous",
    )
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# mock these modules and won't try to actually import them
autodoc_mock_imports = ["boxmot", "segmentation_models_pytorch"]


# -- Social / SEO metadata --------------------------------------------------

# Used when a page does not declare its own ``.. meta:: :name: description``.
_DEFAULT_DESCRIPTION = (
    "Kornia is a differentiable computer vision library for PyTorch: batched, GPU-ready, "
    "autograd-friendly image transforms, filters, color conversions, camera geometry, augmentations "
    "and curated deep learning models."
)
_SOCIAL_IMAGE = "https://github.com/kornia/data/raw/main/kornia_banner_pixie.png"
_META_TAG_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_META_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def _inject_social_metatags(app, pagename, templatename, context, doctree):
    """Add Open Graph / Twitter card tags (and a fallback description) to every HTML page.

    Search engines and chat apps render these when a docs link is shared; Sphinx and furo emit
    none of them by default. The description is taken from the page's own ``.. meta::``
    directive when it has one, so page authors keep control of the snippet.
    """
    metatags = context.get("metatags", "") or ""
    match = None
    for tag in _META_TAG_RE.findall(metatags):
        attrs = dict(_META_ATTR_RE.findall(tag))
        if attrs.get("name", "").lower() == "description" and attrs.get("content"):
            match = attrs["content"]
            break
    description = html.unescape(match).strip().strip('"') if match else _DEFAULT_DESCRIPTION
    description = " ".join(description.split())
    if len(description) > 300:
        description = description[:297].rsplit(" ", 1)[0] + "..."
    title = context.get("title") or project
    if pagename != master_doc:
        title = f"{title} - {project}"
    url = f"{html_baseurl}{pagename}.html"
    esc = html.escape

    extra = []
    if not match:
        extra.append(f'<meta name="description" content="{esc(description)}" />')
    extra += [
        '<meta property="og:type" content="website" />',
        f'<meta property="og:site_name" content="{esc(project)}" />',
        f'<meta property="og:title" content="{esc(title)}" />',
        f'<meta property="og:description" content="{esc(description)}" />',
        f'<meta property="og:url" content="{esc(url)}" />',
        f'<meta property="og:image" content="{_SOCIAL_IMAGE}" />',
        '<meta name="twitter:card" content="summary_large_image" />',
        '<meta name="twitter:site" content="@kornia_foss" />',
        f'<meta name="twitter:title" content="{esc(title)}" />',
        f'<meta name="twitter:description" content="{esc(description)}" />',
        f'<meta name="twitter:image" content="{_SOCIAL_IMAGE}" />',
    ]
    context["metatags"] = metatags + "\n" + "\n".join(extra)


def setup(app):
    app.connect("html-page-context", _inject_social_metatags)
