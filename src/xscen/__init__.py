"""A climate change scenario-building analysis framework, built with xclim/xarray."""

###################################################################################
# Apache Software License 2.0
#
# Copyright (c) 2024, Gabriel Rondeau-Genesse, Pascal Bourgault, Juliette Lavoie, Trevor James Smith
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###################################################################################

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
    regrid,
    scripting,
    spatial,
    testing,
    utils,
)

# Import top-level functions
from .aggregate import *
from .biasadjust import *
from .catalog import DataCatalog, ProjectCatalog
from .catutils import build_path, parse_directory
from .config import CONFIG, load_config
from .diagnostics import properties_and_measures
from .ensembles import *
from .extract import (
    extract_dataset,
    get_period_from_warming_level,
    get_warming_level,
    get_warming_level_from_period,
    search_data_catalogs,
    subset_warming_level,
)
from .indicators import compute_indicators
from .io import save_to_netcdf, save_to_table, save_to_zarr
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
__version__ = "0.13.1"


# FIXME: file and line are unused
def warning_on_one_line(
    message: str,
    category: Warning,
    filename: str,
    lineno: int,
    file=None,  # noqa: F841
    line=None,  # noqa: F841
):
    """
    Monkeypatch Reformat warning so that `warnings.warn` doesn't mention itself.

    Parameters
    ----------
    message : str
        The warning message.
    category : Warning
        The warning category.
    filename : str
        The filename where the warning was raised.
    lineno : int
        The line number where the warning was raised.
    file : file
        The file where the warning was raised.
    line : str
        The line where the warning was raised.

    Returns
    -------
    str
        The reformatted warning message.
    """
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = warning_on_one_line

# FIXME: This is a temporary fix for the FutureWarning spam from intake-esm.
# Print FutureWarnings from intake-esm only once
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="intake_esm",
    message="The default of observed=False is deprecated and will be changed to True in a future version of pandas. "
    "Pass observed=False to retain current behavior or observed=True to adopt the future default "
    "and silence this warning.",
)
