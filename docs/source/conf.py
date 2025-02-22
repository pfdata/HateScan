# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import importlib.util
import sys
import os

# Dynamically resolve the path to the HateScan module
module_name = "HateScan"
module_path = os.path.abspath(os.path.join("..", ".."))  # Adjust the relative path as needed
spec = importlib.util.spec_from_file_location(module_name, os.path.join(module_path, "__init__.py"))

if spec and spec.loader:
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

project = "HateScan"
copyright = "2024, Pablo Freitas"
author = "Pablo Freitas"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
]

templates_path = ["_templates"]
exclude_patterns = ["tests/*", "HateScan.test*"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
