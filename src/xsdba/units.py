"""# noqa: SS01
Units Handling Submodule
========================
"""

import inspect
from functools import wraps

import pint
import xarray as xr


def extract_units(arg):
    """Extract units from a string, DataArray, or scalar."""
    if not (isinstance(arg, (str, xr.DataArray)) or np.isscalar(arg)):
        raise TypeError("Argument must be a str, DataArray, or scalar.")
    elif isinstance(arg, xr.DataArray):
        ustr = None if "units" not in arg.attrs else arg.attrs["units"]
    elif isinstance(arg, str):
        # XC
        _, ustr = arg.split(" ", maxsplit=1)
    else:  # (scalar case)
        ustr = None
    return ustr if ustr is None else pint.Quantity(1, ustr).units


def check_units(args_to_check):
    """Decorator to check that all arguments have the same units (or no units)."""

    # if no units are present (DataArray without units attribute or float), then no check is performed
    # if units are present, then check is performed
    # in mixed cases, an error is raised
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # dictionnary {arg_name:arg} for all args of func
            arg_dict = dict(zip(inspect.getfullargspec(func).args, args))
            # Obtain units (or None if no units) of all args
            units = []
            for arg_name in args_to_check:
                if arg_name not in arg_dict:
                    raise ValueError(
                        f"Argument '{arg_name}' not found in function arguments."
                    )
                units.append(extract_units(arg_dict[arg_name]))
            # Check that units are consistent
            if len(set(units)) > 1:
                raise ValueError(
                    f"{args_to_check} must have the same units (or no units). Got {units}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
