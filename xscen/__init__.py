"""Top-level package for xscen."""
import warnings

# Import the submodules
from . import (
    aggregate,
    biasadjust,
    catalog,
    checkups,
    common,
    config,
    extraction,
    finalize,
    indicators,
    io,
    regridding,
    scr_utils,
)

__author__ = """Gabriel Rondeau-Genesse"""
__email__ = "rondeau-genesse.gabriel@ouranos.ca"
__version__ = "0.2.6-beta"


# monkeypatch so that warnings.warn() doesn't mention itself
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = warning_on_one_line
