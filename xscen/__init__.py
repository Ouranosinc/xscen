"""Top-level package for xscen."""
import warnings

# Import the submodules
from . import catalog, checkups, config, extract, helpers, io, utils

# Import top-level functions
from ._biasadjust import *
from ._ensembles import *
from ._regrid import *
from .aggregate import *
from .catalog import DataCatalog, ProjectCatalog  # noqa
from .config import CONFIG  # noqa
from .extract import extract_dataset, search_data_catalogs  # noqa
from .indicators import compute_indicators  # noqa
from .io import save_to_netcdf, save_to_zarr  # noqa

__author__ = """Gabriel Rondeau-Genesse"""
__email__ = "rondeau-genesse.gabriel@ouranos.ca"
__version__ = "0.2.4-beta"


# monkeypatch so that warnings.warn() doesn't mention itself
def warning_on_one_line(
    message: str, category: Warning, filename: str, lineno: int, file=None, line=None
):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = warning_on_one_line
