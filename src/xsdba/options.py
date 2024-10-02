"""
Global or contextual options for xsdba, similar to xarray.set_options.
"""

# XC remove: metadata locales, do we need them?
from __future__ import annotations

from inspect import signature
from collections.abc import Callable

from boltons.funcutils import wraps

from .locales import _valid_locales
from .logging import ValidationError, raise_warn_or_log

METADATA_LOCALES = "metadata_locales"
DATA_VALIDATION = "data_validation"
CHECK_MISSING = "check_missing"
MISSING_OPTIONS = "missing_options"
RUN_LENGTH_UFUNC = "run_length_ufunc"
XSDBA_EXTRA_OUTPUT = "xsdba_extra_output"
SDBA_ENCODE_CF = "sdba_encode_cf"
KEEP_ATTRS = "keep_attrs"
AS_DATASET = "as_dataset"

MISSING_METHODS: dict[str, Callable] = {}

OPTIONS = {
    METADATA_LOCALES: [],
    DATA_VALIDATION: "raise",
    CHECK_MISSING: "any",
    MISSING_OPTIONS: {},
    RUN_LENGTH_UFUNC: "auto",
    XSDBA_EXTRA_OUTPUT: False,
    SDBA_ENCODE_CF: False,
    KEEP_ATTRS: "xarray",
    AS_DATASET: False,
}

_LOUDNESS_OPTIONS = frozenset(["log", "warn", "raise"])
_RUN_LENGTH_UFUNC_OPTIONS = frozenset(["auto", True, False])
_KEEP_ATTRS_OPTIONS = frozenset(["xarray", True, False])


def _valid_missing_options(mopts):
    for meth, opts in mopts.items():
        cls = MISSING_METHODS.get(meth, None)
        if (
            cls is None  # Method must be registered
            # All options must exist
            or any([opt not in OPTIONS[MISSING_OPTIONS][meth] for opt in opts.keys()])
            # Method option validator must pass, default validator is always True.
            or not cls.validate(**opts)
        ):
            return False
    return True


_VALIDATORS = {
    METADATA_LOCALES: _valid_locales,
    DATA_VALIDATION: _LOUDNESS_OPTIONS.__contains__,
    CHECK_MISSING: lambda meth: meth != "from_context" and meth in MISSING_METHODS,
    MISSING_OPTIONS: _valid_missing_options,
    RUN_LENGTH_UFUNC: _RUN_LENGTH_UFUNC_OPTIONS.__contains__,
    XSDBA_EXTRA_OUTPUT: lambda opt: isinstance(opt, bool),
    SDBA_ENCODE_CF: lambda opt: isinstance(opt, bool),
    KEEP_ATTRS: _KEEP_ATTRS_OPTIONS.__contains__,
    AS_DATASET: lambda opt: isinstance(opt, bool),
}


def _set_missing_options(mopts):
    for meth, opts in mopts.items():
        OPTIONS[MISSING_OPTIONS][meth].update(opts)


def _set_metadata_locales(locales):
    if isinstance(locales, str):
        OPTIONS[METADATA_LOCALES] = [locales]
    else:
        OPTIONS[METADATA_LOCALES] = locales


_SETTERS = {
    MISSING_OPTIONS: _set_missing_options,
    METADATA_LOCALES: _set_metadata_locales,
}


def register_missing_method(name: str) -> Callable:
    """Register missing method."""

    def _register_missing_method(cls):
        sig = signature(cls.is_missing)
        opts = {
            key: param.default if param.default != param.empty else None
            for key, param in sig.parameters.items()
            if key not in ["self", "null", "count"]
        }

        MISSING_METHODS[name] = cls
        OPTIONS[MISSING_OPTIONS][name] = opts
        return cls

    return _register_missing_method


def _run_check(func, option, *args, **kwargs):
    """Run function and customize exception handling based on option."""
    try:
        func(*args, **kwargs)
    except ValidationError as err:
        raise_warn_or_log(err, OPTIONS[option], stacklevel=4)


def datacheck(func: Callable) -> Callable:
    """Decorate functions checking data inputs validity."""

    @wraps(func)
    def run_check(*args, **kwargs):
        return _run_check(func, DATA_VALIDATION, *args, **kwargs)

    return run_check


class set_options:
    """Set options for xclim in a controlled context.

    Attributes
    ----------
    metadata_locales : list[Any]
        List of IETF language tags or tuples of language tags and a translation dict, or
        tuples of language tags and a path to a json file defining translation of attributes.
        Default: ``[]``.
    data_validation : {"log", "raise", "error"}
        Whether to "log", "raise" an error or 'warn' the user on inputs that fail the data checks in
        :py:func:`xclim.datachecks`. Default: ``"raise"``.
    check_missing : {"any", "wmo", "pct", "at_least_n", "skip"}
        How to check for missing data and flag computed indicators.
        Available methods are "any", "wmo", "pct", "at_least_n" and "skip".
        Missing method can be registered through the `xsdba.options.register_missing_method` decorator.
        Default: ``"any"``
    missing_options : dict
        Dictionary of options to pass to the missing method. Keys must the name of
        missing method and values must be mappings from option names to values.
    run_length_ufunc : str
      Whether to use the 1D ufunc version of run length algorithms or the dask-ready broadcasting version.
      Default is ``"auto"``, which means the latter is used for dask-backed and large arrays.
    xsdba_extra_output : bool
        Whether to add diagnostic variables to outputs of sdba's `train`, `adjust`
        and `processing` operations. Details about these additional variables are given in the object's
        docstring. When activated, `adjust` will return a Dataset with `scen` and those extra diagnostics
        For `processing` functions, see the doc, the output type might change, or not depending on the
        algorithm. Default: ``False``.
    sdba_encode_cf : bool
        Whether to encode cf coordinates in the ``map_blocks`` optimization that most adjustment methods are based on.
        This should have no impact on the results, but should run much faster in the graph creation phase.
    keep_attrs : bool or str
        Controls attributes handling in indicators. If True, attributes from all inputs are merged
        using the `drop_conflicts` strategy and then updated with xclim-provided attributes.
        If ``as_dataset`` is also True and a dataset was passed to the ``ds`` argument of the Indicator,
        the dataset's attributes are copied to the indicator's output.
        If False, attributes from the inputs are ignored. If "xarray", xclim will use xarray's `keep_attrs` option.
        Note that xarray's "default" is equivalent to False. Default: ``"xarray"``.
    as_dataset : bool
        If True, indicators output datasets. If False, they output DataArrays. Default :``False``.

    Examples
    --------
    You can use ``set_options`` either as a context manager:

    >>> import xclim
    >>> ds = xr.open_dataset(path_to_tas_file).tas
    >>> with xclim.set_options(metadata_locales=["fr"]):
    ...     out = xclim.atmos.tg_mean(ds)
    ...

    Or to set global options:

    .. code-block:: python

        import xsdba

        xsdba.set_options(missing_options={"pct": {"tolerance": 0.04}})
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    f"argument name {k!r} is not in the set of valid options {set(OPTIONS)!r}"
                )
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                raise ValueError(f"option {k!r} given an invalid value: {v!r}")

            self.old[k] = OPTIONS[k]

        self._update(kwargs)

    def __enter__(self):
        """Context management."""
        return

    @staticmethod
    def _update(kwargs):
        """Update values."""
        for k, v in kwargs.items():
            if k in _SETTERS:
                _SETTERS[k](v)
            else:
                OPTIONS[k] = v

    def __exit__(self, option_type, value, traceback):  # noqa: F841
        """Context management."""
        self._update(self.old)
