"""Top-level package for xscen."""
import warnings

# Import the submodules
from . import (
    aggregate,
    biasadjust,
    catalog,
    checkups,
    config,
    ensembles,
    extract,
    indicators,
    io,
    regrid,
    scripting,
    utils,
)

__author__ = """Gabriel Rondeau-Genesse"""
__email__ = "rondeau-genesse.gabriel@ouranos.ca"
__version__ = "0.2.4-beta"


# monkeypatch so that warnings.warn() doesn't mention itself
def warning_on_one_line(
    message: str, category: Warning, filename: str, lineno: int, file=None, line=None
):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = warning_on_one_line
