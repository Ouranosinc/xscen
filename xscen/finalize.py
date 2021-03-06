import logging
import re
from typing import Optional

import numpy as np
import xarray as xr
from xclim.core import units
from xclim.core.calendar import convert_calendar, get_calendar

from .common import maybe_unstack, unstack_fill_nan
from .config import parse_config

logger = logging.getLogger(__name__)


@parse_config
def change_units(ds: xr.Dataset, variables_and_units: dict):
    """Changes units of Datasets to non-CF units.

    Parameters
    ----------
    ds : xr.Dataset
      Dataset to use
    variables_and_units : dict
      Description of the variables and units to output

    """

    with xr.set_options(keep_attrs=True):
        for v in variables_and_units:
            if (v in ds) and (
                units.units2pint(ds[v]) != units.units2pint(variables_and_units[v])
            ):
                time_in_ds = units.units2pint(ds[v]).dimensionality.get("[time]")
                time_in_out = units.units2pint(
                    variables_and_units[v]
                ).dimensionality.get("[time]")

                if time_in_ds == time_in_out:
                    ds[v] = units.convert_units_to(ds[v], variables_and_units[v])
                elif time_in_ds - time_in_out == 1:
                    # ds is an amount
                    ds[v] = units.amount2rate(ds[v], out_units=variables_and_units[v])
                elif time_in_ds - time_in_out == -1:
                    # ds is a rate
                    ds[v] = units.rate2amount(ds[v], out_units=variables_and_units[v])
                else:
                    raise NotImplementedError(
                        f"No known transformation between {ds[v].units} and {variables_and_units[v]} (temporal dimensionality mismatch)."
                    )

    return ds


def clean_up(
    ds: xr.Dataset,
    variables_and_units: Optional[dict] = None,
    maybe_unstack_dict: Optional[dict] = None,
    convert_calendar_kwargs: Optional[dict] = None,
    missing_by_var: Optional[dict] = None,
    attrs_to_remove: Optional[dict] = None,
    remove_all_attrs_except: Optional[dict] = None,
    add_attrs: Optional[dict] = None,
    change_attr_prefix: Optional[str] = None,
    to_level: Optional[str] = "cleanedup",
):
    """
    Clean up of the dataset. It can:
     - convert to the right units using xscen.finalize.change_units
     - call the xscen.common.maybe_unstack function
     - convert the calendar and interpolate over missing dates
     - remove a list of attributes
     - remove everything but a list of attributes
     - add attributes
     - change the prefix of the catalog attrs

     in that order.

    Parameters
    ----------
    ds: xr.Dataset
        Input dataset to clean up
    variables_and_units: dict
        Dictionary of variable to convert. eg. {'tasmax': 'degC', 'pr': 'mm d-1'}
    maybe_unstack_dict: dict
        Dictionary to pass to xscen.common.maybe_unstack fonction.
        The format should be: {'coords': path_to_coord_file, 'rechunk': {'time': -1 }, 'stack_drop_nans': True}.
    convert_calendar_kwargs: dict
        Dictionary of arguments to feed to xclim.core.calendar.convert_calendar. This will be the same for all variables.
        If missing_by_vars is given, it will override the 'missing' argument given here.
        Eg. {target': default, 'align_on': 'random'}
    missing_by_var: list
        Dictionary where the keys are the variables and the values are the argument to feed the `missing`
        parameters of the xclim.core.calendar.convert_calendar for the given variable.
        If missing_by_var == 'interpolate', the missing will be filled with NaNs, then linearly interpolated over time.
    attrs_to_remove: dict
        Dictionary where the keys are the variables and the values are a list of the attrs that should be removed.
        For global attrs, use the key 'global'.
        The element of the list can be exact matches for the attributes name
        or use the same substring matching rules as intake_esm:
        - ending with a '*' means checks if the substring is contained in the string
        - starting with a '^' means check if the string starts with the substring.
        eg. {'global': ['unnecessary note', 'cell*'], 'tasmax': 'old_name'}
    remove_all_attrs_except: dict
        Dictionary where the keys are the variables and the values are a list of the attrs that should NOT be removed,
        all other attributes will be deleted. If None (default), nothing will be deleted.
        For global attrs, use the key 'global'.
        The element of the list can be exact matches for the attributes name
        or use the same substring matching rules as intake_esm:
        - ending with a '*' means checks if the substring is contained in the string
        - starting with a '^' means check if the string starts with the substring.
        eg. {'global': ['necessary note', '^cat/'], 'tasmax': 'new_name'}
    add_attrs: dict
        Dictionary where the keys are the variables and the values are a another dictionary of attributes.
        For global attrs, use the key 'global'.
        eg. {'global': {'title': 'amazing new dataset'}, 'tasmax': {'note': 'important info about tasmax'}}
    change_attr_prefix: str
        Replace "cat/" in the catalogue global attrs by this new string
    to_level: str
        The processing level to assign to the output.

    Returns
    -------
    ds: xr.Dataset
        Cleaned up dataset
    """

    if variables_and_units:
        logger.info(f"Converting units: {variables_and_units}")
        ds = change_units(ds=ds, variables_and_units=variables_and_units)

    # unstack nans
    if maybe_unstack_dict:
        ds = maybe_unstack(ds, **maybe_unstack_dict)

    # convert calendar
    if convert_calendar_kwargs:
        ds_copy = ds.copy()

        # if missing_by_var exist make sure missing data are added to time axis
        if missing_by_var:
            convert_calendar_kwargs.setdefault("missing", np.nan)

        # make default `align_on`='`random` when the initial calendar is 360day
        if get_calendar(ds) == "360_day" and "align_on" not in convert_calendar_kwargs:
            convert_calendar_kwargs["align_on"] = "random"

        logger.info(f"Converting calendar with {convert_calendar_kwargs} ")
        ds = convert_calendar(ds, **convert_calendar_kwargs)

        # convert each variable individually
        if missing_by_var:
            # remove 'missing' argument to be replace by `missing_by_var`
            del convert_calendar_kwargs["missing"]
            for var, missing in missing_by_var.items():
                logging.info(f"Filling missing {var} with {missing}")
                if missing == "interpolate":
                    converted_var = convert_calendar(
                        ds_copy[var], **convert_calendar_kwargs, missing=np.nan
                    )
                    converted_var = converted_var.interpolate_na(
                        "time", method="linear"
                    )
                else:
                    converted_var = convert_calendar(
                        ds_copy[var], **convert_calendar_kwargs, missing=missing
                    )
                ds[var] = converted_var

    def _search(a, b):
        if a[-1] == "*":  # check if a is contained in b
            return a[:-1] in b
        elif a[0] == "^":
            return b.startswith(a[1:])
        else:
            return a == b

    ds.attrs["cat/processing_level"] = to_level

    # remove attrs
    if attrs_to_remove:
        for var, list_of_attrs in attrs_to_remove.items():
            obj = ds if var == "global" else ds[var]
            for ds_attr in list(obj.attrs.keys()):  # iter over attrs in ds
                for list_attr in list_of_attrs:  # check if we want to remove attrs
                    if _search(list_attr, ds_attr):
                        del obj.attrs[ds_attr]

    # delete all attrs, but the ones in the list
    if remove_all_attrs_except:
        for var, list_of_attrs in remove_all_attrs_except.items():
            obj = ds if var == "global" else ds[var]
            for ds_attr in list(obj.attrs.keys()):  # iter over attrs in ds
                delete = True  # assume we should delete it
                for list_attr in list_of_attrs:
                    if _search(list_attr, ds_attr):
                        delete = (
                            False  # if attr is on the list to not delete, don't delete
                        )
                if delete:
                    del obj.attrs[ds_attr]

    if add_attrs:
        for var, attrs in add_attrs.items():
            obj = ds if var == "global" else ds[var]
            for attrname, attrtmpl in attrs.items():
                obj.attrs[attrname] = attrtmpl

    if change_attr_prefix:
        for ds_attr in list(ds.attrs.keys()):
            new_name = ds_attr.replace("cat/", change_attr_prefix)
            if new_name:
                ds.attrs[new_name] = ds.attrs.pop(ds_attr)

    return ds
