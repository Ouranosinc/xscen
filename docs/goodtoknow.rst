============
Good to know
============

.. _opening-data:

Which function to use when opening data
---------------------------------------
There are many ways to open data in xscen workflows. The list below tries to make the differences clear:

:Search and extract: Using :py:func:`~xscen.extract.search_data_catalogs` + :py:func:`~xscen.extract.extract_dataset`.
	This is the main method recommended to parse catalogs of "raw" data, data not yet modified by your worflow. It has features meant to ease the aggregation and extraction of raw files :

	* variable conversion and resampling of subdaily data
	* spatial and temporal subsetting
	* matching historical and future runs for simulations

	``search_data_catalogs``  returns a dictionary with a specific catalog for each of the unique ``id`` found in the search. One should then iterate over this dictionary and call
	``extract_dataset`` on each item. This then returns a dictionary with a single dataset for each ``xrfreq``. You thus end up with one dataset per frequency and ``id``.

:to_dataset_dict: Using :py:meth:`~xscen.catalog.DataCatalog.to_dataset_dict`.
    When all the data you need is in a single catalog (for example, your :py:meth:`~xscen.catalog.ProjectCatalog`) and you don't need any of the features listed above. Note that this can be combined to a simple `.search` beforehand, to subset on parts of the catalog.
    As explained in :doc:`columns`, it creates a dictionary with a single Dataset for each combination of ``id``, ``domain``, ``processing_level`` and ``xrfreq`` unless different aggregation rules were called during the catalog creation.

:to_dataset: Using :py:meth:`~xscen.catalog.DataCatalog.to_dataset`.
    Similar to ``to_dataset_dict``, but only returns a single dataset. If the catalog has more than one, the call will fail. It behaves like :py:meth:`~intake_esm.core.esm_datastore.to_dask`, but exposes options to add aggregations.
    This is useful when constructing an ensemble dataset that would otherwise result in distinct entries in the output of ``to_dataset_dict``. It can usually be used in
    replacement of a combination of ``to_dataset_dict`` and :py:func:`~xclim.ensembles.create_ensemble`.

:open_dataset: Of course, xscen workflows can still use the conventional :py:func:`~xarray.open_dataset`. Just be aware that datasets opened this way will lack the attributes
    automatically added by the previous functions, which will then result in poorer metadata or even failure for some `xscen` functions. Same thing for :py:func:`~xarray.open_mfdataset`. If one has data listed in a catalog,
    the functions above will usually provide what you need, i.e. : ``xr.open_mfdataset(cat.df.path)`` is very rarely optimal.

:create_ensemble: With :py:meth:`~xscen.catalog.DataCatalog.to_dataset` or :py:func:`~xscen.ensembles.ensemble_stats`, you should usually find what you need. :py:func:`~xclim.ensembles.create_ensemble` is not needed in xscen workflows.


Which function to use when resampling data
------------------------------------------

:extract_dataset: :py:func:`~xscen.extract.extract_dataset` has resampling capabilities to provide daily data from finer sources.

:xclim indicators: Through :py:func:`~xscen.indicators.compute_indicators`, xscen workflows can easily use `xclim indicators <https://xclim.readthedocs.io/en/stable/indicators.html>`_
	to go from daily data to coarser (monthly, seasonal, annual).

What is currently not covered by either `xscen` or `xclim` is a method to resample from data coarser than daily, where the base period is non-uniform (ex: resampling from monthly to annual data, taking into account the number of days per month).
