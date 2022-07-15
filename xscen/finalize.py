import logging
from typing import Optional
import xarray as xr
import numpy as np
import re

from xclim.core import units
from xclim.core.calendar import convert_calendar


from .config import parse_config
from .common import unstack_fill_nan, maybe_unstack

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


def clean_up(ds: xr.Dataset,
             var_and_convert_units: Optional[dict] = None,
             maybe_unstack_dict: Optional[dict] = None,
             add_feb_29: bool =False,
             attrs_to_remove: Optional[dict]= None,
             only_attrs_to_keep: Optional[dict]= None,
             add_attrs: Optional[dict] = None,
             change_attr_prefix: Optional[str] = None,
             ):
    """
    Clean up of the dataset. It can:
     - convert to the right units using xscen.finalize.change_units
     - call the xscen.common.maybe_unstack fonction
     - add a february 29th by linear interpolation
     - remove a list of attributes
     - remove everything but a list of attributes
     - add attributes
     - change the prefix of the catalog attrs
     in that order
    Parameters
    ----------
    ds: xr.Dataset
        Input dataset to clean up
    var_and_convert_units: dict
        Dictionary of variable to convert. eg. {'tasmax': 'degC', 'pr': 'mm d-1'}
    maybe_unstack_dict: dict
        Dictionnary to pass to xscen.common.maybe_unstack fonction.
        The format should be: {'coords': path_to_coord_file, 'rechunk': {'time': -1 }, 'stack_drop_nans': True)}.
    add_feb_29: bool
        wheter to add back a february 29th (that was removed in a noleap calendar) by linear interpolation
    attrs_to_remove: dict
        Dictionary where the keys are the variables and the values are a list of the attrs that should be removed.
        The element of the list can be regular expression ( eg.'cat/*' to get rid of all the attrs that start by 'cat/')
        For global attrs, use the key 'global'.
        eg. {'global': ['unnecessary note', 'cat/*'], 'tasmax': 'old_name'}
    only_attrs_to_keep: dict
        Dictionary where the keys are the variables and the values are a list of the attrs that should NOT be removed.
        The element of the list can be regular expression ( eg.'cat/*' to only keep the attrs that start by 'cat/')
        For global attrs, use the key 'global'.
        eg. {'global': ['necessary note', 'cat/*'], 'tasmax': 'new_name'}
    add_attrs: dict
        Dictionary where the keys are the variables and the values are a another dictionary of attributes.
        For global attrs, use the key 'global'.
        eg. {'global': {'title': 'amazing new dataset'}, 'tasmax': {'note': 'important info about tasmax'}}
    change_attr_prefix: str
     Replace "cat/" in the catalogue attrs by this new string

    Returns
    -------
    ds: xr.Dataset
        Cleaned up dataset
    """
    if var_and_convert_units:
        ds = change_units(ds = ds , variables_and_units= var_and_convert_units)

    # unstack nans
    if maybe_unstack_dict:
        ds = maybe_unstack(ds, **maybe_unstack_dict)

    # put back feb 29th
    if add_feb_29:
        with_missing = convert_calendar(ds, 'standard', missing=np.NaN)
        ds = with_missing.interpolate_na('time', method='linear')

    # remove attrs
    if attrs_to_remove:
        for var, list_of_attrs in attrs_to_remove:
            obj = ds if var == 'global' else ds[var]
            for ds_attr in list(ds.attrs.keys()): #iter over attrs in ds
                for list_attr in list_of_attrs: # check if ds_attrs is in the list we want to remove
                    if re.search(list_attr, ds_attr):
                        del obj.attrs[list_attr]

    # delete all attrs, but the ones in the list
    if only_attrs_to_keep:
        for var, list_of_attrs in only_attrs_to_keep:
            obj = ds if var == 'global' else ds[var]
            for ds_attr in list(ds.attrs.keys()):  # iter over attrs in ds
                for list_attr in list_of_attrs:  # check if ds_attrs is in the list we want to remove
                    if not re.search(list_attr, ds_attr):
                        del obj.attrs[list_attr]

    if add_attrs:
        for var, attrs in add_attrs:
            obj = ds if var == 'global' else ds[var]
            for attrname, attrtmpl in attrs.items():
                obj.attrs[attrname] = attrtmpl

    if change_attr_prefix:
        for ds_attr in list(ds.attrs.keys()):
                ds[ds_attr]= ds_attr.replace('cat/',change_attr_prefix)

    return ds



