"""Top-level package for xscen."""
import warnings

# Import the submodules
from . import (
    aggregate,
    biasadjust,
    catalog,
    catutils,
    config,
    diagnostics,
    ensembles,
    extract,
    indicators,
    io,
    reduce,
    regrid,
    scripting,
    spatial,
    utils,
)

# Import top-level functions
from .aggregate import *
from .biasadjust import *
from .catalog import DataCatalog, ProjectCatalog  # noqa
from .catutils import build_path, parse_directory
from .config import CONFIG, load_config  # noqa
from .diagnostics import properties_and_measures
from .ensembles import *
from .extract import (  # noqa
    extract_dataset,
    get_warming_level,
    search_data_catalogs,
    subset_warming_level,
)
from .indicators import compute_indicators  # noqa
from .io import save_to_netcdf, save_to_zarr  # noqa
from .reduce import build_reduction_data, reduce_ensemble
from .regrid import *
from .scripting import (
    TimeoutException,
    measure_time,
    move_and_delete,
    save_and_update,
    send_mail,
    send_mail_on_exit,
    timeout,
)
from .utils import clean_up

__author__ = """Gabriel Rondeau-Genesse"""
__email__ = "rondeau-genesse.gabriel@ouranos.ca"
__version__ = "0.7.1"


# monkeypatch so that warnings.warn() doesn't mention itself
def warning_on_one_line(
    message: str, category: Warning, filename: str, lineno: int, file=None, line=None
):  # noqa: D103
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = warning_on_one_line
