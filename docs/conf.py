# Copyright (c) 2026 Beestream Ltd.
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
# Configuration file for the Sphinx documentation builder.
# The pages are the Markdown files in this directory, read with MyST.

# -- Project information

project = "jitNN"
copyright = "2026, Beestream Ltd"
author = "Beestream Ltd"

# -- General configuration

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
]

source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Anchors for "FILE.md#section" links, as GitHub generates them.
myst_heading_anchors = 4
myst_enable_extensions = ["colon_fence"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
html_logo = "images/jitnn-logo-dark.svg"
html_favicon = "images/jitnn-icon.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {"logo_only": True}

# -- Options for LaTeX output (the PDF build)

latex_logo = None
latex_documents = [("index", "jitnn.tex", "jitNN Documentation", author, "manual")]
