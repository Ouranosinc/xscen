# pylint: disable=missing-kwoa
"""# noqa: SS01
Pre- and Post-Processing Submodule
==================================
"""
from __future__ import annotations

import types
from collections.abc import Sequence
from typing import cast

import cftime
import dask.array as dsk
import numpy as np
import xarray as xr
from xarray.core import dtypes
from xarray.core.utils import get_temp_dimname

from xsdba._processing import _adapt_freq, _normalize, _reordering
from xsdba.base import Grouper, uses_dask
from xsdba.formatting import update_xsdba_history
from xsdba.nbutils import _escore
from xsdba.units import harmonize_units
from xsdba.utils import ADDITIVE, copy_all_attrs

__all__ = [
    "adapt_freq",
    "escore",
    "from_additive_space",
    "grouped_time_indexes",
    "jitter",
    "jitter_over_thresh",
    "jitter_under_thresh",
    "normalize",
    "reordering",
    "stack_periods",
    "stack_variables",
    "standardize",
    "to_additive_space",
    "uniform_noise_like",
    "unstack_periods",
    "unstack_variables",
    "unstandardize",
]


@update_xsdba_history
@harmonize_units(["ref", "sim", "thresh"])
def adapt_freq(
    ref: xr.DataArray,
    sim: xr.DataArray,
    *,
    group: Grouper | str,
    thresh: str = "0 mm d-1",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    r"""
    Adapt frequency of values under thresh of `sim`, in order to match ref.

    This is useful when the dry-day frequency in the simulations is higher than in the references. This function
    will create new non-null values for `sim`/`hist`, so that adjustment factors are less wet-biased.
    Based on :cite:t:`themesl_empirical-statistical_2012`.

    Parameters
    ----------
    ref : xr.Dataset
        Target/reference data, usually observed data, with a "time" dimension.
    sim : xr.Dataset
        Simulated data, with a "time" dimension.
    group : str or Grouper
        Grouping information, see base.Grouper.
    thresh : str
        Threshold below which values are considered zero, a quantity with units.

    Returns
    -------
    sim_adj : xr.DataArray
        Simulated data with the same frequency of values under threshold than ref.
        Adjustment is made group-wise.
    pth : xr.DataArray
        For each group, the smallest value of sim that was not frequency-adjusted.
        All values smaller were either left as zero values or given a random value between thresh and pth.
        NaN where frequency adaptation wasn't needed.
    dP0 : xr.DataArray
        For each group, the percentage of values that were corrected in sim.

    Notes
    -----
    With :math:`P_0^r` the frequency of values under threshold :math:`T_0` in the reference (ref) and
    :math:`P_0^s` the same for the simulated values, :math:`\\Delta P_0 = \\frac{P_0^s - P_0^r}{P_0^s}`,
    when positive, represents the proportion of values under :math:`T_0` that need to be corrected.

    The correction replaces a proportion :math:`\\Delta P_0` of the values under :math:`T_0` in sim by a uniform random
    number between :math:`T_0` and :math:`P_{th}`, where :math:`P_{th} = F_{ref}^{-1}( F_{sim}( T_0 ) )` and
    `F(x)` is the empirical cumulative distribution function (CDF).

    References
    ----------
    :cite:cts:`themesl_empirical-statistical_2012`
    """
    out = _adapt_freq(xr.Dataset(dict(sim=sim, ref=ref)), group=group, thresh=thresh)

    # Set some metadata
    copy_all_attrs(out, sim)
    out.sim_ad.attrs.update(sim.attrs)
    out.sim_ad.attrs.update(
        references="Themeßl et al. (2012), Empirical-statistical downscaling and error correction of regional climate "
        "models and its impact on the climate change signal, Climatic Change, DOI 10.1007/s10584-011-0224-4."
    )
    out.pth.attrs.update(
        long_name="Smallest value of the timeseries not corrected by frequency adaptation.",
        units=sim.units,
    )
    out.dP0.attrs.update(
        long_name=f"Proportion of values smaller than {thresh} in the timeseries corrected by frequency adaptation",
    )

    return out.sim_ad, out.pth, out.dP0


def jitter_under_thresh(x: xr.DataArray, thresh: str) -> xr.DataArray:
    """Replace values smaller than threshold by a uniform random noise.

    Warnings
    --------
    Not to be confused with R's jitter, which adds uniform noise instead of replacing values.

    Parameters
    ----------
    x : xr.DataArray
        Values.
    thresh : str
        Threshold under which to add uniform random noise to values, a quantity with units.

    Returns
    -------
    xr.DataArray.

    Notes
    -----
    If thresh is high, this will change the mean value of x.
    """
    j: xr.DataArray = jitter(x, lower=thresh, upper=None, minimum=None, maximum=None)
    return j


def jitter_over_thresh(x: xr.DataArray, thresh: str, upper_bnd: str) -> xr.DataArray:
    """Replace values greater than threshold by a uniform random noise.

    Warnings
    --------
    Not to be confused with R's jitter, which adds uniform noise instead of replacing values.

    Parameters
    ----------
    x : xr.DataArray
        Values.
    thresh : str
        Threshold over which to add uniform random noise to values, a quantity with units.
    upper_bnd : str
        Maximum possible value for the random noise, a quantity with units.

    Returns
    -------
    xr.DataArray.

    Notes
    -----
    If thresh is low, this will change the mean value of x.
    """
    j: xr.DataArray = jitter(
        x, lower=None, upper=thresh, minimum=None, maximum=upper_bnd
    )
    return j


@update_xsdba_history
@harmonize_units(["x", "lower", "upper", "minimum", "maximum"])
def jitter(
    x: xr.DataArray,
    lower: str | None = None,
    upper: str | None = None,
    minimum: str | None = None,
    maximum: str | None = None,
) -> xr.DataArray:
    """Replace values under a threshold and values above another by a uniform random noise.

    Warnings
    --------
    Not to be confused with R's `jitter`, which adds uniform noise instead of replacing values.

    Parameters
    ----------
    x : xr.DataArray
        Values.
    lower : str, optional
        Threshold under which to add uniform random noise to values, a quantity with units.
        If None, no jittering is performed on the lower end.
    upper : str, optional
        Threshold over which to add uniform random noise to values, a quantity with units.
        If None, no jittering is performed on the upper end.
    minimum : str, optional
        Lower limit (excluded) for the lower end random noise, a quantity with units.
        If None but `lower` is not None, 0 is used.
    maximum : str, optional
        Upper limit (excluded) for the upper end random noise, a quantity with units.
        If `upper` is not None, it must be given.

    Returns
    -------
    xr.DataArray
        Same as  `x` but values < lower are replaced by a uniform noise in range (minimum, lower)
        and values >= upper are replaced by a uniform noise in range [upper, maximum).
        The two noise distributions are independent.
    """
    # with units.context(infer_context(x.attrs.get("standard_name"))):
    out: xr.DataArray = x
    notnull = x.notnull()
    if lower is not None:
        jitter_lower = np.array(lower).astype(float)
        jitter_min = np.array(minimum if minimum is not None else 0).astype(float)
        jitter_min = jitter_min + np.finfo(x.dtype).eps
        if uses_dask(x):
            jitter_dist = dsk.random.uniform(
                low=jitter_min, high=jitter_lower, size=x.shape, chunks=x.chunks
            )
        else:
            jitter_dist = np.random.uniform(
                low=jitter_min, high=jitter_lower, size=x.shape
            )
        out = out.where(~((x < jitter_lower) & notnull), jitter_dist.astype(x.dtype))
    if upper is not None:
        if maximum is None:
            raise ValueError("If 'upper' is given, so must 'maximum'.")
        jitter_upper = np.array(upper).astype(float)
        jitter_max = np.array(maximum).astype(float)
        if uses_dask(x):
            jitter_dist = dsk.random.uniform(
                low=jitter_upper, high=jitter_max, size=x.shape, chunks=x.chunks
            )
        else:
            jitter_dist = np.random.uniform(
                low=jitter_upper, high=jitter_max, size=x.shape
            )
        out = out.where(~((x >= jitter_upper) & notnull), jitter_dist.astype(x.dtype))

    copy_all_attrs(out, x)  # copy attrs and same units
    return out


@update_xsdba_history
@harmonize_units(["data", "norm"])
def normalize(
    data: xr.DataArray,
    norm: xr.DataArray | None = None,
    *,
    group: Grouper | str,
    kind: str = ADDITIVE,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Normalize an array by removing its mean.

    Normalization if performed group-wise and according to `kind`.

    Parameters
    ----------
    data : xr.DataArray
        The variable to normalize.
    norm : xr.DataArray, optional
        If present, it is used instead of computing the norm again.
    group : str or Grouper
        Grouping information. See :py:class:`xsdba.base.Grouper` for details..
    kind : {'+', '*'}
        If `kind` is "+", the mean is subtracted from the mean and if it is '*', it is divided from the data.

    Returns
    -------
    xr.DataArray
        Groupwise anomaly.
    norm : xr.DataArray
        Mean over each group.
    """
    ds = xr.Dataset({"data": data})

    if norm is not None:
        ds = ds.assign(norm=norm)

    out = _normalize(ds, group=group, kind=kind)
    copy_all_attrs(out, ds)
    out.data.attrs.update(data.attrs)
    out.norm.attrs["units"] = data.attrs["units"]
    return out.data.rename(data.name), out.norm


def uniform_noise_like(
    da: xr.DataArray, low: float = 1e-6, high: float = 1e-3
) -> xr.DataArray:
    """Return a uniform noise array of the same shape as da.

    Noise is uniformly distributed between low and high.
    Alternative method to `jitter_under_thresh` for avoiding zeroes.
    """
    mod: types.ModuleType
    kw: dict
    if uses_dask(da):
        mod = dsk
        kw = {"chunks": da.chunks}
    else:
        mod = np
        kw = {}

    return da.copy(
        data=(high - low) * mod.random.random_sample(size=da.shape, **kw) + low
    )


@update_xsdba_history
def standardize(
    da: xr.DataArray,
    mean: xr.DataArray | None = None,
    std: xr.DataArray | None = None,
    dim: str = "time",
) -> tuple[xr.DataArray | xr.Dataset, xr.DataArray, xr.DataArray]:
    """Standardize a DataArray by centering its mean and scaling it by its standard deviation.

    Either of both of mean and std can be provided if need be.

    Returns
    -------
    out : xr.DataArray or xr.Dataset
        Standardized data.
    mean : xr.DataArray
        Mean.
    std : xr.DataArray
        Standard Deviation.
    """
    if mean is None:
        mean = da.mean(dim, keep_attrs=True)
    if std is None:
        std = da.std(dim, keep_attrs=True)
    out = (da - mean) / std
    copy_all_attrs(out, da)
    return out, mean, std


@update_xsdba_history
def unstandardize(da: xr.DataArray, mean: xr.DataArray, std: xr.DataArray):
    """Rescale a standardized array by performing the inverse operation of `standardize`."""
    out = (std * da) + mean
    copy_all_attrs(out, da)
    return out


@update_xsdba_history
def reordering(ref: xr.DataArray, sim: xr.DataArray, group: str = "time") -> xr.Dataset:
    """Reorder data in `sim` following the order of ref.

    The rank structure of `ref` is used to reorder the elements of `sim` along dimension "time", optionally doing the
    operation group-wise.

    Parameters
    ----------
    ref : xr.DataArray
        Array whose rank order sim should replicate.
    sim : xr.DataArray
        Array to reorder.
    group : str
        Grouping information. See :py:class:`xsdba.base.Grouper` for details.

    Returns
    -------
    xr.Dataset
        Sim reordered according to ref's rank order.

    References
    ----------
    :cite:cts:`cannon_multivariate_2018`.
    """
    ds = xr.Dataset({"sim": sim, "ref": ref})
    out: xr.Dataset = _reordering(ds, group=group).reordered
    copy_all_attrs(out, sim)
    return out


@update_xsdba_history
def escore(
    tgt: xr.DataArray,
    sim: xr.DataArray,
    dims: Sequence[str] = ("variables", "time"),
    N: int = 0,
    scale: bool = False,
) -> xr.DataArray:
    r"""Energy score, or energy dissimilarity metric, based on :cite:t:`szekely_testing_2004` and :cite:t:`cannon_multivariate_2018`.

    Parameters
    ----------
    tgt: xr.DataArray
        Target observations.
    sim: xr.DataArray
        Candidate observations. Must have the same dimensions as `tgt`.
    dims: sequence of 2 strings
        The name of the dimensions along which the variables and observation points are listed.
        `tgt` and `sim` can have different length along the second one, but must be equal along the first one.
        The result will keep all other dimensions.
    N : int
        If larger than 0, the number of observations to use in the score computation. The points are taken
        evenly distributed along `obs_dim`.
    scale : bool
        Whether to scale the data before computing the score. If True, both arrays as scaled according
        to the mean and standard deviation of `tgt` along `obs_dim`. (std computed with `ddof=1` and both
        statistics excluding NaN values).

    Returns
    -------
    xr.DataArray
        Return e-score with dimensions not in `dims`.

    Notes
    -----
    Explanation adapted from the "energy" R package documentation.
    The e-distance between two clusters :math:`C_i`, :math:`C_j` (tgt and sim) of size :math:`n_i,n_j`
    proposed by :cite:t:`szekely_testing_2004` is defined by:

    .. math::

        e(C_i,C_j) = \frac{1}{2}\frac{n_i n_j}{n_i + n_j} \left[2 M_{ij} − M_{ii} − M_{jj}\right]

    where

    .. math::

        M_{ij} = \frac{1}{n_i n_j} \sum_{p = 1}^{n_i} \sum_{q = 1}^{n_j} \left\Vert X_{ip} − X{jq} \right\Vert.

    :math:`\Vert\cdot\Vert` denotes Euclidean norm, :math:`X_{ip}` denotes the p-th observation in the i-th cluster.

    The input scaling and the factor :math:`\frac{1}{2}` in the first equation are additions of
    :cite:t:`cannon_multivariate_2018` to the metric. With that factor, the test becomes identical to the one
    defined by :cite:t:`baringhaus_new_2004`.
    This version is tested against values taken from Alex Cannon's MBC R package :cite:p:`cannon_mbc_2020`.

    References
    ----------
    :cite:cts:`baringhaus_new_2004,cannon_multivariate_2018,cannon_mbc_2020,szekely_testing_2004`.
    """
    pts_dim, obs_dim = dims

    if N > 0:
        # If N non-zero we only take around N points, evenly distributed
        sim_step = int(np.ceil(sim[obs_dim].size / N))
        sim = sim.isel({obs_dim: slice(None, None, sim_step)})
        tgt_step = int(np.ceil(tgt[obs_dim].size / N))
        tgt = tgt.isel({obs_dim: slice(None, None, tgt_step)})

    if scale:
        tgt, avg, std = standardize(tgt)
        sim, _, _ = standardize(sim, avg, std)

    # The dimension renaming is to allow different coordinates.
    # Otherwise, apply_ufunc tries to align both obs_dim together.
    new_dim = get_temp_dimname(tgt.dims, obs_dim)
    sim = sim.rename({obs_dim: new_dim})
    out: xr.DataArray = xr.apply_ufunc(
        _escore,
        tgt,
        sim,
        input_core_dims=[[pts_dim, obs_dim], [pts_dim, new_dim]],
        output_dtypes=[sim.dtype],
        dask="parallelized",
    )

    out.name = "escores"
    out = out.assign_attrs(
        {
            "long_name": "Energy dissimilarity metric",
            "description": f"Escores computed from {N or 'all'} points.",
            "references": "Székely, G. J. and Rizzo, M. L. (2004) Testing for Equal Distributions in High Dimension, InterStat, November (5)",
        }
    )
    return out


def _get_number_of_elements_by_year(time):
    """Get the number of elements in time in a year by inferring its sampling frequency.

    Only calendar with uniform year lengths are supported : 360_day, noleap, all_leap.
    """
    mult, freq, _, _ = parse_offset(xr.infer_freq(time))
    days_in_year = time.dt.days_in_year.max()
    elements_in_year = {"Q": 4, "M": 12, "D": days_in_year, "h": days_in_year * 24}
    N_in_year = elements_in_year.get(freq, 1) / mult
    if N_in_year % 1 != 0:
        raise ValueError(
            f"Sampling frequency of the data must be Q, M, D or h and evenly divide a year (got {mult}{freq})."
        )

    return int(N_in_year)


@update_xsdba_history
@harmonize_units(["data", "lower_bound", "upper_bound"])
def to_additive_space(
    data: xr.DataArray,
    lower_bound: str,
    upper_bound: str | None = None,
    trans: str = "log",
):
    r"""Transform a non-additive variable into an additive space by the means of a log or logit transformation.

    Based on :cite:t:`alavoine_distinct_2022`.

    Parameters
    ----------
    data : xr.DataArray
        A variable that can't usually be bias-adjusted by additive methods.
    lower_bound : str
        The smallest physical value of the variable, excluded, as a Quantity string.
        The data should only have values strictly larger than this bound.
    upper_bound : str, optional
        The largest physical value of the variable, excluded, as a Quantity string.
        Only relevant for the logit transformation.
        The data should only have values strictly smaller than this bound.
    trans : {'log', 'logit'}
        The transformation to use. See notes.

    Notes
    -----
    Given a variable that is not usable in an additive adjustment, this applies a transformation to a space where
    additive methods are sensible. Given :math:`X` the variable, :math:`b_-` the lower physical bound of that variable
    and :math:`b_+` the upper physical bound, two transformations are currently implemented to get :math:`Y`,
    the additive-ready variable. :math:`\ln` is the natural logarithm.

    - `log`

        .. math::

            Y = \ln\left( X - b_- \right)

        Usually used for variables with only a lower bound, like precipitation (`pr`,  `prsn`, etc)
        and daily temperature range (`dtr`). Both have a lower bound of 0.

    - `logit`

        .. math::

            X' = (X - b_-) / (b_+ - b_-)
            Y = \ln\left(\frac{X'}{1 - X'} \right)

        Usually used for variables with both a lower and a upper bound, like relative and specific humidity,
        cloud cover fraction, etc.

    This will thus produce `Infinity` and `NaN` values where :math:`X == b_-` or :math:`X == b_+`.
    We recommend using :py:func:`jitter_under_thresh` and :py:func:`jitter_over_thresh` to remove those issues.

    See Also
    --------
    from_additive_space : For the inverse transformation.
    jitter_under_thresh : Remove values exactly equal to the lower bound.
    jitter_over_thresh : Remove values exactly equal to the upper bound.

    References
    ----------
    :cite:cts:`alavoine_distinct_2022`.
    """
    # with units.context(infer_context(data.attrs.get("standard_name"))):
    lower_bound_array = np.array(lower_bound).astype(float)
    if upper_bound is not None:
        upper_bound_array = np.array(upper_bound).astype(float)

    with xr.set_options(keep_attrs=True), np.errstate(divide="ignore"):
        if trans == "log":
            out = cast(xr.DataArray, np.log(data - lower_bound_array))
        elif trans == "logit" and upper_bound is not None:
            data_prime = (data - lower_bound_array) / (
                upper_bound_array - lower_bound_array  # pylint: disable=E0606
            )
            out = cast(xr.DataArray, np.log(data_prime / (1 - data_prime)))
        else:
            raise NotImplementedError("`trans` must be one of 'log' or 'logit'.")

    # Attributes to remember all this.
    out = out.assign_attrs(xsdba_transform=trans)
    out = out.assign_attrs(xsdba_transform_lower=lower_bound_array)
    if upper_bound is not None:
        out = out.assign_attrs(xsdba_transform_upper=upper_bound_array)
    if "units" in out.attrs:
        out = out.assign_attrs(xsdba_transform_units=out.attrs.pop("units"))
        out = out.assign_attrs(units="")
    return out


@update_xsdba_history
@harmonize_units(["units", "lower_bound", "upper_bound"])
def from_additive_space(
    data: xr.DataArray,
    lower_bound: str | None = None,
    upper_bound: str | None = None,
    trans: str | None = None,
    units: str | None = None,
):
    r"""Transform back to the physical space a variable that was transformed with `to_additive_space`.

    Based on :cite:t:`alavoine_distinct_2022`.
    If parameters are not present on the attributes of the data, they must be all given are arguments.

    Parameters
    ----------
    data : xr.DataArray
        A variable that was transformed by :py:func:`to_additive_space`.
    lower_bound : str, optional
        The smallest physical value of the variable, as a Quantity string.
        The final data will have no value smaller or equal to this bound.
        If None (default), the `xsdba_transform_lower` attribute is looked up on `data`.
    upper_bound : str, optional
        The largest physical value of the variable, as a Quantity string.
        Only relevant for the logit transformation.
        The final data will have no value larger or equal to this bound.
        If None (default), the `xsdba_transform_upper` attribute is looked up on `data`.
    trans : {'log', 'logit'}, optional
        The transformation to use. See notes.
        If None (the default), the `xsdba_transform` attribute is looked up on `data`.
    units : str, optional
        The units of the data before transformation to the additive space.
        If None (the default), the `xsdba_transform_units` attribute is looked up on `data`.

    Returns
    -------
    xr.DataArray
        The physical variable. Attributes are conserved, even if some might be incorrect.
        Except units which are taken from `xsdba_transform_units` if available.
        All `xsdba_transform*` attributes are deleted.

    Notes
    -----
    Given a variable that is not usable in an additive adjustment, :py:func:`to_additive_space` applied a transformation
    to a space where additive methods are sensible. Given :math:`Y` the transformed variable, :math:`b_-` the
    lower physical bound of that variable and :math:`b_+` the upper physical bound, two back-transformations are
    currently implemented to get :math:`X`, the physical variable.

    - `log`

        .. math::

            X = e^{Y} + b_-

    - `logit`

        .. math::

            X' = \frac{1}{1 + e^{-Y}}
            X = X * (b_+ - b_-) + b_-

    See Also
    --------
    to_additive_space : For the original transformation.

    References
    ----------
    :cite:cts:`alavoine_distinct_2022`.
    """
    if trans is None and lower_bound is None and units is None:
        try:
            trans = data.attrs["xsdba_transform"]
            units = data.attrs["xsdba_transform_units"]
            lower_bound_array = np.array(data.attrs["xsdba_transform_lower"]).astype(
                float
            )
            if trans == "logit":
                upper_bound_array = np.array(
                    data.attrs["xsdba_transform_upper"]
                ).astype(float)
        except KeyError as err:
            raise ValueError(
                f"Attribute {err!s} must be present on the input data "
                "or all parameters must be given as arguments."
            ) from err
    elif (
        trans is not None
        and lower_bound is not None
        and units is not None
        and (upper_bound is not None or trans == "log")
    ):
        # FIXME: convert_units_to is causing issues since it can't handle all variations of Quantified here
        lower_bound_array = np.array(lower_bound).astype(float)
        if trans == "logit":
            upper_bound_array = np.array(upper_bound).astype(float)
    else:
        raise ValueError(
            "Parameters missing. Either all parameters are given as attributes of data, "
            "or all of them are given as input arguments."
        )

    with xr.set_options(keep_attrs=True):
        if trans == "log":
            out = np.exp(data) + lower_bound_array
        elif trans == "logit":
            out_prime = 1 / (1 + np.exp(-data))
            out = (
                out_prime
                * (upper_bound_array - lower_bound_array)  # pylint: disable=E0606
                + lower_bound_array
            )
        else:
            raise NotImplementedError("`trans` must be one of 'log' or 'logit'.")

    # Remove unneeded attributes, put correct units back.
    out.attrs.pop("xsdba_transform", None)
    out.attrs.pop("xsdba_transform_lower", None)
    out.attrs.pop("xsdba_transform_upper", None)
    out.attrs.pop("xsdba_transform_units", None)
    out = out.assign_attrs(units=units)
    return out


def stack_variables(ds: xr.Dataset, rechunk: bool = True, dim: str = "multivar"):
    """Stack different variables of a dataset into a single DataArray with a new "variables" dimension.

    Variable attributes are all added as lists of attributes to the new coordinate, prefixed with "_".
    Variables are concatenated in the new dimension in alphabetical order, to ensure
    coherent behaviour with different datasets.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    rechunk : bool
        If True (default), dask arrays are rechunked with `variables : -1`.
    dim : str
        Name of dimension along which variables are indexed.

    Returns
    -------
    xr.DataArray
        The transformed variable. Attributes are conserved, even if some might be incorrect, except for units,
        which are replaced with `""`. Old units are stored in `xsdba_transformation_units`.
        A `xsdba_transform` attribute is added, set to the transformation method. `xsdba_transform_lower` and
        `xsdba_transform_upper` are also set if the requested bounds are different from the defaults.

        Array with variables stacked along `dim` dimension. Units are set to "".
    """
    # Store original arrays' attributes
    attrs: dict = {}
    # sort to have coherent order with different datasets
    data_vars = sorted(ds.data_vars.items(), key=lambda e: e[0])
    nvar = len(data_vars)
    for i, (_nm, var) in enumerate(data_vars):
        for name, attr in var.attrs.items():
            attrs.setdefault(f"_{name}", [None] * nvar)[i] = attr

    # Special key used for later `unstacking`
    attrs["is_variables"] = True
    var_crd = xr.DataArray([nm for nm, vr in data_vars], dims=(dim,), name=dim)

    da = xr.concat([vr for nm, vr in data_vars], var_crd, combine_attrs="drop")

    if uses_dask(da) and rechunk:
        da = da.chunk({dim: -1})

    da.attrs.update(ds.attrs)
    da.attrs["units"] = ""
    da[dim].attrs.update(attrs)
    return da.rename("multivariate")


def unstack_variables(da: xr.DataArray, dim: str | None = None) -> xr.Dataset:
    """Unstack a DataArray created by `stack_variables` to a dataset.

    Parameters
    ----------
    da : xr.DataArray
        Array holding different variables along `dim` dimension.
    dim : str, optional
        Name of dimension along which the variables are stacked.
        If not specified (default), `dim` is inferred from attributes of the coordinate.

    Returns
    -------
    xr.Dataset
        Dataset holding each variable in an individual DataArray.
    """
    if dim is None:
        for _dim, _crd in da.coords.items():
            if _crd.attrs.get("is_variables"):
                dim = str(_dim)
                break
        else:
            raise ValueError("No variable coordinate found, were attributes removed?")

    ds = xr.Dataset(
        {name.item(): da.sel({dim: name.item()}, drop=True) for name in da[dim]},
        attrs=da.attrs,
    )
    del ds.attrs["units"]

    # Reset attributes
    for name, attr_list in da[dim].attrs.items():
        if not name.startswith("_"):
            continue
        for attr, var in zip(attr_list, da[dim], strict=False):
            if attr is not None:
                ds[var.item()].attrs[name[1:]] = attr

    return ds


def grouped_time_indexes(times, group):
    """Time indexes for every group blocks

    Time indexes can be used to implement a pseudo-"numpy.groupies" approach to grouping.

    Parameters
    ----------
    times : xr.DataArray
        Time dimension in the dataset of interest.
    group : str or Grouper
        Grouping information, see base.Grouper.

    Returns
    -------
    g_idxs : xr.DataArray
        Time indexes of the blocks (only using `group.name` and not `group.window`).
    gw_idxs : xr.DataArray
        Time indexes of the blocks (built with a rolling window of `group.window` if any).
    """

    def _get_group_complement(da, group):
        # complement of "dayofyear": "year", etc.
        gr = group if isinstance(group, str) else group.name
        if gr == "time.dayofyear":
            return da.time.dt.year
        if gr == "time.month":
            return da.time.dt.strftime("%Y-%d")
        raise NotImplementedError(f"Grouping {gr} not implemented.")

    # does not work with group == "time.month"
    group = group if isinstance(group, Grouper) else Grouper(group)
    gr, win = group.name, group.window
    # get time indices (0,1,2,...) for each block
    timeind = xr.DataArray(np.arange(times.size), coords={"time": times})
    win_dim0, win_dim = (
        get_temp_dimname(timeind.dims, lab) for lab in ["win_dim0", "win_dim"]
    )
    if gr == "time.dayofyear":
        # time indices for each block with window = 1
        g_idxs = timeind.groupby(gr).apply(
            lambda da: da.assign_coords(time=_get_group_complement(da, gr)).rename(
                {"time": "year"}
            )
        )
        # time indices for each block with general window
        da = timeind.rolling(time=win, center=True).construct(window_dim=win_dim0)
        gw_idxs = da.groupby(gr).apply(
            lambda da: da.assign_coords(time=_get_group_complement(da, gr)).stack(
                {win_dim: ["time", win_dim0]}
            )
        )
        gw_idxs = gw_idxs.transpose(..., win_dim)
    elif gr == "time":
        gw_idxs = timeind.rename({"time": win_dim}).expand_dims({win_dim0: [-1]})
        g_idxs = gw_idxs.copy()
    else:
        raise NotImplementedError(f"Grouping {gr} not implemented.")
    gw_idxs.attrs["group"] = (gr, win)
    gw_idxs.attrs["time_dim"] = win_dim
    gw_idxs.attrs["group_dim"] = [d for d in g_idxs.dims if d != win_dim][0]
    return g_idxs, gw_idxs


# XC: calendar
def parse_offset(freq: str) -> tuple[int, str, bool, str | None]:
    """Parse an offset string.

    Parse a frequency offset and, if needed, convert to cftime-compatible components.

    Parameters
    ----------
    freq : str
        Frequency offset.

    Returns
    -------
    multiplier : int
        Multiplier of the base frequency. "[n]W" is always replaced with "[7n]D",
        as xarray doesn't support "W" for cftime indexes.
    offset_base : str
        Base frequency.
    is_start_anchored : bool
        Whether coordinates of this frequency should correspond to the beginning of the period (`True`)
        or its end (`False`). Can only be False when base is Y, Q or M; in other words, xsdba assumes frequencies finer
        than monthly are all start-anchored.
    anchor : str, optional
        Anchor date for bases Y or Q. As xarray doesn't support "W",
        neither does xsdba (anchor information is lost when given).
    """
    # Useful to raise on invalid freqs, convert Y to A and get default anchor (A, Q)
    offset = pd.tseries.frequencies.to_offset(freq)
    base, *anchor = offset.name.split("-")
    anchor = anchor[0] if len(anchor) > 0 else None
    start = ("S" in base) or (base[0] not in "AYQM")
    if base.endswith("S") or base.endswith("E"):
        base = base[:-1]
    mult = offset.n
    if base == "W":
        mult = 7 * mult
        base = "D"
        anchor = None
    return mult, base, start, anchor


# XC : calendar
def compare_offsets(freqA: str, op: str, freqB: str) -> bool:
    """Compare offsets string based on their approximate length, according to a given operator.

    Offset are compared based on their length approximated for a period starting
    after 1970-01-01 00:00:00. If the offsets are from the same category (same first letter),
    only the multiplier prefix is compared (QS-DEC == QS-JAN, MS < 2MS).
    "Business" offsets are not implemented.

    Parameters
    ----------
    freqA : str
        RHS Date offset string ('YS', '1D', 'QS-DEC', ...).
    op : {'<', '<=', '==', '>', '>=', '!='}
        Operator to use.
    freqB : str
        LHS Date offset string ('YS', '1D', 'QS-DEC', ...).

    Returns
    -------
    bool
        Return freqA op freqB.
    """
    # Get multiplier and base frequency
    t_a, b_a, _, _ = parse_offset(freqA)
    t_b, b_b, _, _ = parse_offset(freqB)

    if b_a != b_b:
        # Different base freq, compare length of first period after beginning of time.
        t = pd.date_range("1970-01-01T00:00:00.000", periods=2, freq=freqA)
        t_a = (t[1] - t[0]).total_seconds()
        t = pd.date_range("1970-01-01T00:00:00.000", periods=2, freq=freqB)
        t_b = (t[1] - t[0]).total_seconds()
    # else Same base freq, compare multiplier only.

    return get_op(op)(t_a, t_b)


# XC: calendar
def construct_offset(mult: int, base: str, start_anchored: bool, anchor: str | None):
    """Reconstruct an offset string from its parts.

    Parameters
    ----------
    mult : int
        The period multiplier (>= 1).
    base : str
        The base period string (one char).
    start_anchored : bool
        If True and base in [Y, Q, M], adds the "S" flag, False add "E".
    anchor : str, optional
        The month anchor of the offset. Defaults to JAN for bases YS and QS and to DEC for bases YE and QE.

    Returns
    -------
    str
        An offset string, conformant to pandas-like naming conventions.

    Notes
    -----
    This provides the mirror opposite functionality of :py:func:`parse_offset`.
    """
    start = ("S" if start_anchored else "E") if base in "YAQM" else ""
    if anchor is None and base in "AQY":
        anchor = "JAN" if start_anchored else "DEC"
    return (
        f"{mult if mult > 1 else ''}{base}{start}{'-' if anchor else ''}{anchor or ''}"
    )


# XC: calendar
# Names of calendars that have the same number of days for all years
uniform_calendars = ("noleap", "all_leap", "365_day", "366_day", "360_day")


# XC: calendar
def _month_is_first_period_month(time, freq):
    """Return True if the given time is from the first month of freq."""
    if isinstance(time, cftime.datetime):
        frq_monthly = xr.coding.cftime_offsets.to_offset("MS")
        frq = xr.coding.cftime_offsets.to_offset(freq)
        if frq_monthly.onOffset(time):
            return frq.onOffset(time)
        return frq.onOffset(frq_monthly.rollback(time))
    # Pandas
    time = pd.Timestamp(time)
    frq_monthly = pd.tseries.frequencies.to_offset("MS")
    frq = pd.tseries.frequencies.to_offset(freq)
    if frq_monthly.is_on_offset(time):
        return frq.is_on_offset(time)
    return frq.is_on_offset(frq_monthly.rollback(time))


# XC: calendar
# TODO: implement needed functions in stack_periods
# move to processing
def stack_periods(
    da: xr.Dataset | xr.DataArray,
    window: int = 30,
    stride: int | None = None,
    min_length: int | None = None,
    freq: str = "YS",
    dim: str = "period",
    start: str = "1970-01-01",
    align_days: bool = True,
    pad_value=dtypes.NA,
):
    """Construct a multi-period array.

    Stack different equal-length periods of `da` into a new 'period' dimension.

    This is similar to ``da.rolling(time=window).construct(dim, stride=stride)``, but adapted for arguments
    in terms of a base temporal frequency that might be non-uniform (years, months, etc.).
    It is reversible for some cases (see `stride`).
    A rolling-construct method will be much more performant for uniform periods (days, weeks).

    Parameters
    ----------
    da : xr.Dataset or xr.DataArray
        An xarray object with a `time` dimension.
        Must have a uniform timestep length.
        Output might be strange if this does not use a uniform calendar (noleap, 360_day, all_leap).
    window : int
        The length of the moving window as a multiple of ``freq``.
    stride : int, optional
        At which interval to take the windows, as a multiple of ``freq``.
        For the operation to be reversible with :py:func:`unstack_periods`, it must divide `window` into an odd number of parts.
        Default is `window` (no overlap between periods).
    min_length : int, optional
        Windows shorter than this are not included in the output.
        Given as a multiple of ``freq``. Default is ``window`` (every window must be complete).
        Similar to the ``min_periods`` argument of  ``da.rolling``.
        If ``freq`` is annual or quarterly and ``min_length == ``window``, the first period is considered complete
        if the first timestep is in the first month of the period.
    freq : str
        Units of ``window``, ``stride`` and ``min_length``, as a frequency string.
        Must be larger or equal to the data's sampling frequency.
        Note that this function offers an easier interface for non-uniform period (like years or months)
        but is much slower than a rolling-construct method.
    dim : str
        The new dimension name.
    start : str
        The `start` argument passed to :py:func:`xarray.date_range` to generate the new placeholder
        time coordinate.
    align_days : bool
        When True (default), an error is raised if the output would have unaligned days across periods.
        If `freq = 'YS'`, day-of-year alignment is checked and if `freq` is "MS" or "QS", we check day-in-month.
        Only uniform-calendar will pass the test for `freq='YS'`.
        For other frequencies, only the `360_day` calendar will work.
        This check is ignored if the sampling rate of the data is coarser than "D".
    pad_value : Any
        When some periods are shorter than others, this value is used to pad them at the end.
        Passed directly as argument ``fill_value`` to :py:func:`xarray.concat`,
        the default is the same as on that function.

    Return
    ------
    xr.DataArray
        A DataArray with a new `period` dimension and a `time` dimension with the length of the longest window.
        The new time coordinate has the same frequency as the input data but is generated using
        :py:func:`xarray.date_range` with the given `start` value.
        That coordinate is the same for all periods, depending on the choice of ``window`` and ``freq``, it might make sense.
        But for unequal periods or non-uniform calendars, it will certainly not.
        If ``stride`` is a divisor of ``window``, the correct timeseries can be reconstructed with :py:func:`unstack_periods`.
        The coordinate of `period` is the first timestep of each window.
    """
    # Import in function to avoid cyclical imports
    from xsdba.units import (  # pylint: disable=import-outside-toplevel
        infer_sampling_units,
        units2str,
    )

    stride = stride or window
    min_length = min_length or window
    if stride > window:
        raise ValueError(
            f"Stride must be less than or equal to window. Got {stride} > {window}."
        )

    srcfreq = xr.infer_freq(da.time)
    cal = da.time.dt.calendar
    use_cftime = da.time.dtype == "O"

    if (
        # TODO: Can we remove compare_offsets, only used here ...
        # if srcfreq in ("D", "h", "min", "s", "ms", "us", "ns")
        compare_offsets(srcfreq, "<=", "D")
        and align_days
        and (
            (freq.startswith(("Y", "A")) and cal not in uniform_calendars)
            or (freq.startswith(("Q", "M")) and window > 1 and cal != "360_day")
        )
    ):
        if freq.startswith(("Y", "A")):
            u = "year"
        else:
            u = "month"
        raise ValueError(
            f"Stacking {window}{freq} periods will result in unaligned day-of-{u}. "
            f"Consider converting the calendar of your data to one with uniform {u} lengths, "
            "or pass `align_days=False` to disable this check."
        )

    # Convert integer inputs to freq strings
    mult, *args = parse_offset(freq)
    # TODO: remove construct?  (hard code  construct-offset)
    win_frq = construct_offset(mult * window, *args)
    strd_frq = construct_offset(mult * stride, *args)
    minl_frq = construct_offset(mult * min_length, *args)

    # The same time coord as da, but with one extra element.
    # This way, the last window's last index is not returned as None by xarray's grouper.
    time2 = xr.DataArray(
        xr.date_range(
            da.time[0].item(),
            freq=srcfreq,
            calendar=cal,
            periods=da.time.size + 1,
            use_cftime=use_cftime,
        ),
        dims=("time",),
        name="time",
    )

    periods = []
    # longest = 0
    # Iterate over strides, but recompute the full window for each stride start
    for strd_slc in da.resample(time=strd_frq).groups.values():
        win_resamp = time2.isel(time=slice(strd_slc.start, None)).resample(time=win_frq)
        # Get slice for first group
        win_slc = list(win_resamp.groups.values())[0]
        if min_length < window:
            # If we ask for a min_length period instead is it complete ?
            min_resamp = time2.isel(time=slice(strd_slc.start, None)).resample(
                time=minl_frq
            )
            min_slc = list(min_resamp.groups.values())[0]
            open_ended = min_slc.stop is None
        else:
            # The end of the group slice is None if no outside-group value was found after the last element
            # As we added an extra step to time2, we avoid the case where a group ends exactly on the last element of ds
            open_ended = win_slc.stop is None
        if open_ended:
            # Too short, we got to the end
            break
        if (
            strd_slc.start == 0
            and parse_offset(freq)[1] in "YAQ"
            and min_length == window
            and not _month_is_first_period_month(da.time[0].item(), freq)
        ):
            # For annual or quarterly frequencies (which can be anchor-based),
            # if the first time is not in the first month of the first period,
            # then the first period is incomplete but by a fractional amount.
            continue
        periods.append(
            slice(
                strd_slc.start + win_slc.start,
                (
                    (strd_slc.start + win_slc.stop)
                    if win_slc.stop is not None
                    else da.time.size
                ),
            )
        )

    # Make coordinates
    lengths = xr.DataArray(
        [slc.stop - slc.start for slc in periods],
        dims=(dim,),
        attrs={"long_name": "Length of each period"},
    )
    longest = lengths.max().item()
    # Length as a pint-ready array : with proper units, but values are not usable as indexes anymore
    m, u = infer_sampling_units(da)
    lengths = lengths * m
    lengths.attrs["units"] = units2str(u)
    # Start points for each period and remember parameters for unstacking
    starts = xr.DataArray(
        [da.time[slc.start].item() for slc in periods],
        dims=(dim,),
        attrs={
            "long_name": "Start of the period",
            # Save parameters so that we can unstack.
            "window": window,
            "stride": stride,
            "freq": freq,
            "unequal_lengths": int(len(np.unique(lengths)) > 1),
        },
    )
    # The "fake" axis that all periods share
    fake_time = xr.date_range(
        start, periods=longest, freq=srcfreq, calendar=cal, use_cftime=use_cftime
    )
    # Slice and concat along new dim. We drop the index and add a new one so that xarray can concat them together.
    out = xr.concat(
        [
            da.isel(time=slc)
            .drop_vars("time")
            .assign_coords(time=np.arange(slc.stop - slc.start))
            for slc in periods
        ],
        dim,
        join="outer",
        fill_value=pad_value,
    )
    out = out.assign_coords(
        time=(("time",), fake_time, da.time.attrs.copy()),
        **{f"{dim}_length": lengths, dim: starts},
    )
    out.time.attrs.update(long_name="Placeholder time axis")
    return out


# XC: calendar
def unstack_periods(da: xr.DataArray | xr.Dataset, dim: str = "period"):
    """Unstack an array constructed with :py:func:`stack_periods`.

    Can only work with periods stacked with a ``stride`` that divides ``window`` in an odd number of sections.
    When ``stride`` is smaller than ``window``, only the center-most stride of each window is kept,
    except for the beginning and end which are taken from the first and last windows.

    Parameters
    ----------
    da : xr.DataArray
        As constructed by :py:func:`stack_periods`, attributes of the period coordinates must have been preserved.
    dim : str
        The period dimension name.

    Notes
    -----
    The following table shows which strides are included (``o``) in the unstacked output.

    In this example, ``stride`` was a fifth of ``window`` and ``min_length`` was four (4) times ``stride``.
    The row index ``i`` the period index in the stacked dataset,
    columns are the stride-long section of the original timeseries.

    .. table:: Unstacking example with ``stride < window``.

        === === === === === === === ===
         i   0   1   2   3   4   5   6
        === === === === === === === ===
         3               x   x   o   o
         2           x   x   o   x   x
         1       x   x   o   x   x
         0   o   o   o   x   x
        === === === === === === === ===
    """
    from xsdba.units import (  # pylint: disable=import-outside-toplevel
        infer_sampling_units,
    )

    try:
        starts = da[dim]
        window = starts.attrs["window"]
        stride = starts.attrs["stride"]
        freq = starts.attrs["freq"]
        unequal_lengths = bool(starts.attrs["unequal_lengths"])
    except (AttributeError, KeyError) as err:
        raise ValueError(
            f"`unstack_periods` can't find the window, stride and freq attributes on the {dim} coordinates."
        ) from err

    if unequal_lengths:
        try:
            lengths = da[f"{dim}_length"]
        except KeyError as err:
            raise ValueError(
                f"`unstack_periods` can't find the `{dim}_length` coordinate."
            ) from err
        # Get length as number of points
        m, _ = infer_sampling_units(da.time)
        lengths = lengths // m
    else:
        # It is acceptable to lose "{dim}_length" if they were all equal
        lengths = xr.DataArray([da.time.size] * da[dim].size, dims=(dim,))

    # Convert from the fake axis to the real one
    time_as_delta = da.time - da.time[0]
    if da.time.dtype == "O":
        # cftime can't add with np.timedelta64 (restriction comes from numpy which refuses to add O with m8)
        time_as_delta = pd.TimedeltaIndex(
            time_as_delta
        ).to_pytimedelta()  # this array is O, numpy complies
    else:
        # Xarray will return int when iterating over datetime values, this returns timestamps
        starts = pd.DatetimeIndex(starts)

    def _reconstruct_time(_time_as_delta, _start):
        times = _time_as_delta + _start
        return xr.DataArray(times, dims=("time",), coords={"time": times}, name="time")

    # Easy case:
    if window == stride:
        # just concat them all
        periods = []
        for i, (start, length) in enumerate(
            zip(starts.values, lengths.values, strict=False)
        ):
            real_time = _reconstruct_time(time_as_delta, start)
            periods.append(
                da.isel(**{dim: i}, drop=True)
                .isel(time=slice(0, length))
                .assign_coords(time=real_time.isel(time=slice(0, length)))
            )
        return xr.concat(periods, "time")

    # Difficult and ambiguous case
    if (window / stride) % 2 != 1:
        raise NotImplementedError(
            "`unstack_periods` can't work with strides that do not divide the window into an odd number of parts."
            f"Got {window} / {stride} which is not an odd integer."
        )

    # Non-ambiguous overlapping case
    Nwin = window // stride
    mid = (Nwin - 1) // 2  # index of the center window

    mult, *args = parse_offset(freq)
    strd_frq = construct_offset(mult * stride, *args)

    periods = []
    for i, (start, length) in enumerate(
        zip(starts.values, lengths.values, strict=False)
    ):
        real_time = _reconstruct_time(time_as_delta, start)
        slices = list(real_time.resample(time=strd_frq).groups.values())
        if i == 0:
            slc = slice(slices[0].start, min(slices[mid].stop, length))
        elif i == da.period.size - 1:
            slc = slice(slices[mid].start, min(slices[Nwin - 1].stop or length, length))
        else:
            slc = slice(slices[mid].start, min(slices[mid].stop, length))
        periods.append(
            da.isel(**{dim: i}, drop=True)
            .isel(time=slc)
            .assign_coords(time=real_time.isel(time=slc))
        )

    return xr.concat(periods, "time")
