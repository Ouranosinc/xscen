"""A climate change scenario-building analysis framework, built with xclim/xarray."""

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
    testing,
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
from .io import save_to_netcdf, save_to_table, save_to_zarr  # noqa
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
__version__ = "0.8.3"


def warning_on_one_line(
    message: str, category: Warning, filename: str, lineno: int, file=None, line=None
):  # noqa: D103
    """Monkeypatch Reformat warning so that `warnings.warn` doesn't mention itself."""
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = warning_on_one_line

# FIXME: This is a temporary fix for the FutureWarning spam from intake-esm.
# Print FutureWarnings from intake-esm only once
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="intake_esm",
    message="The default of observed=False is deprecated and will be changed to True in a future version of pandas. "
    "Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="intake_esm",
    message="DataFrame.applymap has been deprecated. Use DataFrame.map instead.",
)
