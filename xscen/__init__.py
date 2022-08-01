"""Top-level package for xscen."""
import warnings

# Import the submodules
from . import aggregate, catalog, checkups, config, indicators, io, scripting, utils

# Import top-level functions
from ._biasadjust import *
from ._ensembles import *
from ._extract import *
from ._ops import *
from ._regrid import *
from .utils import CV  # noqa

__author__ = """Gabriel Rondeau-Genesse"""
__email__ = "rondeau-genesse.gabriel@ouranos.ca"
__version__ = "0.2.4-beta"


# monkeypatch so that warnings.warn() doesn't mention itself
def warning_on_one_line(
    message: str, category: Warning, filename: str, lineno: int, file=None, line=None
):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = warning_on_one_line
