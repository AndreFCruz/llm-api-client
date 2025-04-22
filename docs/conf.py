import os
import sys
sys.path.insert(0, os.path.abspath('../llm_api_client'))

# -- Project information -----------------------------------------------------
project = 'llm-api-client'
copyright = '2025, AndreFCruz'
author = 'AndreFCruz'

# Get the version from the package itself (if available)
# You might need to adjust this depending on how you manage your package version
try:
    from llm_api_client import __version__
    version = __version__
    release = version
except ImportError:
    version = '0.0.0'
    release = version

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Include documentation from docstrings
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
    'sphinx.ext.viewcode',     # Add links to source code
    'sphinx_copybutton',       # Add copy button to code blocks
    'sphinx.ext.doctest',      # Add doctest support for examples
    'sphinx_rtd_theme',        # Read the Docs theme
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# -- Options for autodoc extension -------------------------------------------
autodoc_member_order = 'bysource'   # Order members by source order
# You might need to mock imports if some dependencies are heavy or C extensions
# autodoc_mock_imports = ["dependency1", "dependency2"]

# -- Napoleon settings -----------------------------------------------------
napoleon_google_docstring = False
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

# -- sphinx_copybutton settings ----------------------------------------------
copybutton_prompt_text = '>>> '
copybutton_prompt_is_regexp = False
