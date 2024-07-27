"""# noqa: SS01
Units Handling Submodule
========================
"""

import inspect
from copy import deepcopy
from functools import wraps

# this dependency is "necessary" for convert_units_to
# if we only do checks, we could get rid of it
import cf_xarray.units
import numpy as np
import pint
import xarray as xr

from .base import Quantified, copy_all_attrs

# shamelessly adapted from `cf-xarray` (which adopted it from MetPy and xclim itself)
units = deepcopy(cf_xarray.units.units)
# Switch this flag back to False. Not sure what that implies, but it breaks some tests.
units.force_ndarray_like = False  # noqa: F841
# Another alias not included by cf_xarray
units.define("@alias percent = pct")


# XC
def units2pint(value: xr.DataArray | str | units.Quantity) -> pint.Unit:
    """Return the pint Unit for the DataArray units.

    Parameters
    ----------
    value : xr.DataArray or str or pint.Quantity
        Input data array or string representing a unit (with no magnitude).

    Returns
    -------
    pint.Unit
        Units of the data array.
    """
    if isinstance(value, str):
        unit = value
    elif isinstance(value, xr.DataArray):
        unit = value.attrs["units"]
    elif isinstance(value, units.Quantity):
        # This is a pint.PlainUnit, which is not the same as a pint.Unit
        return cast(pint.Unit, value.units)
    else:
        raise NotImplementedError(f"Value of type `{type(value)}` not supported.")

    # Catch user errors undetected by Pint
    degree_ex = ["deg", "degree", "degrees"]
    unit_ex = [
        "C",
        "K",
        "F",
        "Celsius",
        "Kelvin",
        "Fahrenheit",
        "celsius",
        "kelvin",
        "fahrenheit",
    ]
    possibilities = [f"{d} {u}" for d in degree_ex for u in unit_ex]
    if unit.strip() in possibilities:
        raise ValidationError(
            "Remove white space from temperature units, e.g. use `degC`."
        )

    return units.parse_units(unit)


# XC
def str2pint(val: str) -> pint.Quantity:
    """Convert a string to a pint.Quantity, splitting the magnitude and the units.

    Parameters
    ----------
    val : str
        A quantity in the form "[{magnitude} ]{units}", where magnitude can be cast to a float and
        units is understood by `units2pint`.

    Returns
    -------
    pint.Quantity
        Magnitude is 1 if no magnitude was present in the string.
    """
    mstr, *ustr = val.split(" ", maxsplit=1)
    try:
        if ustr:
            return units.Quantity(float(mstr), units=units2pint(ustr[0]))
        return units.Quantity(float(mstr))
    except ValueError:
        return units.Quantity(1, units2pint(val))


# XC
# def ensure_delta(unit: str) -> str:
#     """Return delta units for temperature.

#     For dimensions where delta exist in pint (Temperature), it replaces the temperature unit by delta_degC or
#     delta_degF based on the input unit. For other dimensionality, it just gives back the input units.

#     Parameters
#     ----------
#     unit : str
#         unit to transform in delta (or not)
#     """
#     u = units2pint(unit)
#     d = 1 * u
#     #
#     delta_unit = pint2cfunits(d - d)
#     # replace kelvin/rankine by delta_degC/F
#     if "kelvin" in u._units:
#         delta_unit = pint2cfunits(u / units2pint("K") * units2pint("delta_degC"))
#     if "degree_Rankine" in u._units:
#         delta_unit = pint2cfunits(u / units2pint("°R") * units2pint("delta_degF"))
#     return delta_unit


def extract_units(arg):
    """Extract units from a string, DataArray, or scalar."""
    if not (isinstance(arg, (str, xr.DataArray)) or np.isscalar(arg)):
        print(arg)
        raise TypeError(
            f"Argument must be a str, DataArray, or scalar. Got {type(arg)}"
        )
    elif isinstance(arg, xr.DataArray):
        ustr = None if "units" not in arg.attrs else arg.attrs["units"]
    elif isinstance(arg, str):
        ustr = str2pint(arg).units
    else:  # (scalar case)
        ustr = None
    return ustr if ustr is None else pint.Quantity(1, ustr).units


def check_units(args_to_check):
    """Decorator to check that all arguments have the same units (or no units)."""

    # if no units are present (DataArray without units attribute or float), then no check is performed
    # if units are present, then check is performed
    # in mixed cases, an error is raised
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            # dictionnary {arg_name:arg} for all args of func
            arg_dict = dict(zip(inspect.getfullargspec(func).args, args))
            # Obtain units (or None if no units) of all args
            units = []
            for arg_name in args_to_check:
                if isinstance(arg_name, str):
                    value = arg_dict[arg_name]
                    key = arg_name
                if isinstance(
                    arg_name, dict
                ):  # support for Dataset, or a dict of thresholds
                    key, val = list(arg_name.keys())[0], list(arg_name.values())[0]
                    value = arg_dict[key][val]
                if value is None:  # optional argument, should be ignored
                    args_to_check.remove(arg_name)
                    continue
                if key not in arg_dict:
                    raise ValueError(
                        f"Argument '{arg_name}' not found in function arguments."
                    )
                units.append(extract_units(value))
            # Check that units are consistent
            if len(set(units)) > 1:
                raise ValueError(
                    f"{args_to_check} must have the same units (or no units). Got {units}"
                )
            return func(*args, **kwargs)

        return _wrapper

    return _decorator


# XC simplified
def convert_units_to(  # noqa: C901
    source: Quantified,
    target: Quantified | units.Unit,
) -> xr.DataArray | float:
    """Convert a mathematical expression into a value with the same units as a DataArray.

    If the dimensionalities of source and target units differ, automatic CF conversions
    will be applied when possible. See :py:func:`xclim.core.units.cf_conversion`.

    Parameters
    ----------
    source : str or xr.DataArray or units.Quantity
        The value to be converted, e.g. '4C' or '1 mm/d'.
    target : str or xr.DataArray or units.Quantity or units.Unit
        Target array of values to which units must conform.

    Returns
    -------
    xr.DataArray or float
        The source value converted to target's units.
        The outputted type is always similar to `source` initial type.
        Attributes are preserved unless an automatic CF conversion is performed,
        in which case only the new `standard_name` appears in the result.

    See Also
    --------
    cf_conversion
    amount2rate
    rate2amount
    amount2lwethickness
    lwethickness2amount
    """
    # Target units
    target_unit = extract_units(target)
    source_unit = extract_units(source)
    if target_unit == source_unit:
        return source if isinstance(source, str) is False else str2pint(source).m
    else:  # Convert units
        if isinstance(source, xr.DataArray):
            out = source.copy(data=units.convert(source.data, source_unit, target_unit))
            out = out.assign_attrs(units=target_unit)
        else:
            out = str2pint(source).to(target_unit)
        return out


def _fill_args_dict(args, kwargs, args_to_check, func):
    """Combine args and kwargs into a dict."""
    args_dict = {}
    signature = inspect.signature(func)
    for ik, (k, v) in enumerate(signature.parameters.items()):
        if ik < len(args):
            value = args[ik]
        if ik >= len(args):
            value = v.default if k not in kwargs else kwargs[k]
        args_dict[k] = value
    return args_dict


def _split_args_kwargs(args, func):
    """Assign Keyword only arguments to kwargs."""
    kwargs = {}
    signature = inspect.signature(func)
    indices_to_pop = []
    for ik, (k, v) in enumerate(signature.parameters.items()):
        if v.kind == inspect.Parameter.KEYWORD_ONLY:
            indices_to_pop.append(ik)
            kwargs[k] = v
    indices_to_pop.sort(reverse=True)
    for ind in indices_to_pop:
        args.pop(ind)
    return args, kwargs


# TODO: make it work with Dataset for real
# TODO: add a switch to prevent string from being converted to float?
def harmonize_units(args_to_check):
    """Check that units are compatible with dimensions, otherwise raise a `ValidationError`."""

    # if no units are present (DataArray without units attribute or float), then no check is performed
    # if units are present, then check is performed
    # in mixed cases, an error is raised
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            arg_names = inspect.getfullargspec(func).args
            args_dict = _fill_args_dict(list(args), kwargs, args_to_check, func)
            first_arg_name = args_to_check[0]
            first_arg = args_dict[first_arg_name]
            for arg_name in args_to_check[1:]:
                if isinstance(arg_name, str):
                    value = args_dict[arg_name]
                    key = arg_name
                if isinstance(
                    arg_name, dict
                ):  # support for Dataset, or a dict of thresholds
                    key, val = list(arg_name.keys())[0], list(arg_name.values())[0]
                    value = args_dict[key][val]
                if value is None:  # optional argument, should be ignored
                    args_to_check.remove(arg_name)
                    continue
                if key not in args_dict:
                    raise ValueError(
                        f"Argument '{arg_name}' not found in function arguments."
                    )
                args_dict[key] = convert_units_to(value, first_arg)
            args = list(args_dict.values())
            args, kwargs = _split_args_kwargs(args, kwargs, func)
            return func(*args, **kwargs)

        return _wrapper

    return _decorator


def _add_default_kws(params_dict, params_to_check, func):
    """Combine args and kwargs into a dict."""
    args_dict = {}
    signature = inspect.signature(func)
    for ik, (k, v) in enumerate(signature.parameters.items()):
        if k not in params_dict and k in params_to_check:
            if v.default != inspect._empty:
                params_dict[k] = v.default
    return params_dict


def harmonize_units(params_to_check):
    """Check that units are compatible with dimensions, otherwise raise a `ValidationError`."""

    # if no units are present (DataArray without units attribute or float), then no check is performed
    # if units are present, then check is performed
    # in mixed cases, an error is raised
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            params_func = inspect.signature(func).parameters.keys()
            if set(params_to_check).issubset(set(params_func)) is False:
                raise ValueError(
                    f"`harmonize_units' inputs `{params_to_check}` should be a subset of "
                    f"`{func.__name__}`'s arguments: `{params_func}` (arguments that can contain units)"
                )
            arg_names = inspect.getfullargspec(func).args
            args_dict = dict(zip(arg_names, args))
            params_dict = args_dict | {k: v for k, v in kwargs.items()}
            params_dict = {k: v for k, v in params_dict.items() if k in params_to_check}
            params_dict = _add_default_kws(params_dict, params_to_check, func)
            params_dict_keys = [k for k in params_dict.keys()]
            if set(params_dict.keys()) != set(params_to_check):
                raise ValueError(
                    f"{params_to_check} were passed but only {params_dict.keys()} were found "
                    f"in `{func.__name__}`'s arguments"
                )
            first_param = params_dict[params_to_check[0]]
            for param_name in params_dict.keys():
                if isinstance(param_name, str):
                    value = params_dict[param_name]
                    key = param_name
                if isinstance(
                    param_name, dict
                ):  # support for Dataset, or a dict of thresholds
                    key, val = list(param_name.keys())[0], list(param_name.values())[0]
                    value = params_dict[key][val]
                if value is None:  # optional argument, should be ignored
                    continue
                params_dict[key] = convert_units_to(value, first_param)
            for k in [k for k in params_dict.keys() if k not in args_dict.keys()]:
                kwargs[k] = params_dict[k]
                params_dict.pop(k)
            args = list(args)
            for iarg in range(len(args)):
                if arg_names[iarg] in params_dict.keys():
                    args[iarg] = params_dict[arg_names[iarg]]
            return func(*args, **kwargs)

        return _wrapper

    return _decorator
