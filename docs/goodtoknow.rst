============
Good to know
============

.. _opening-data:

Which function to use when opening data
---------------------------------------
There are many ways to open data in xscen workflows. The list below tries to make the differences clear:

:Search and extract: Using :py:func:`~xscen.extract.search_data_catalogs` + :py:func:`~xscen.extract.extract_dataset`.
    This is the main method recommended to parse catalogs of "raw" data, data not yet modified by your workflow. It has features meant to ease the aggregation and extraction of raw files :

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

:extract_dataset: :py:func:`~xscen.extract.extract_dataset`'s resampling capabilities are meant to provide daily data from finer sources.

:resample: :py:func`xscen.extract.resample` extends xarray's `resample` methods with support for weighted resampling when starting from data coarser than daily and for handling of missing timesteps or values.

:xclim indicators: Through :py:func:`~xscen.indicators.compute_indicators`, xscen workflows can easily use `xclim indicators <https://xclim.readthedocs.io/en/stable/indicators.html>`_
    to go from daily data to coarser (monthly, seasonal, annual), with missing values handling. This option will add more metadata than the two firsts.


Metadata translation
--------------------

xscen itself does not add many translatable attributes, but when it does, it will look into xclim's options for which locales to translate them to. Similar to xclim, it will always add a particular attribute in english and then translations with the same attribute name suffixed by "_XX", where "XX" is the two-letter language code, as set in the `ISO-639-1 <https://en.wikipedia.org/wiki/ISO_639-1>`_ standard. For example, if a function adds a `long_name` and Inuktitut translation is activated, the function will also add a `long_name_iu` attribute.

In a config file, activating French translations for both xclim's indicators and xscen (and figanos) is done with :

.. code-block:: yaml

    xclim:
        metadata_locales:
            - fr

Which can also be activated in the code using :py:func:`xclim.core.options.set_options`. Note that this only applies to attributes that are *added* to a dataset. Some xscen functions will instead update an existing attribute. For example, when calculating the climatology of a variable with `long_name` `Mean temperature`, :py:func:`climatological_mean` will update the `long_name` as `30-year average of Mean temperature`. This automatic update is done for all locales available in the variable, no matter what xclim option is activated. For example, if a `long_name_eu` exists in the variable and a Basque translation catalog exists in that xscen instance, then the attribute will be translated, no matter what xclim's ``metadata_locales`` is set to.

Translation is of course not automatic but relies on manually populated `gettext <https://docs.python.org/3/library/gettext.html?highlight=gettext#module-gettext>`_ catalogs. xscen ships with a catalog of french (fr) translations. See :ref:`translating-xscen` to learn how to add translations to xscen. xclim's documentation of the same subject is `here <https://xclim.readthedocs.io/en/stable/internationalization.html>`_.

If your xscen is installed in "editable" mode in its source directory (``pip install -e .``), you should run ``make translate`` each time you pull changes from the upstream source.

Module-wide options
-------------------

As seen above, it can be useful to use the "special" sections of the config file to set some module-wide options. For example:

.. code-block:: yaml

    logging:
        # same arguments as python's logging.config.dictConfig
    xarray:
        keep_attrs: True
    xclim:
        metadata_locales:
            - fr
        check_missing: "skip"
    warning:
        # warning_category : filter_action
        all: ignore


Global warming dataset
----------------------
The :py:func:`xscen.extract.get_warming_level` and :py:func:`xscen.extract.subset_warming_level` functions use a custom made database of global temperature averages to find the global warming levels of known climate simulations. The database is stored as a netCDF file `inside the package itself <https://github.com/Ouranosinc/xscen/blob/main/xscen/data/IPCC_annual_global_tas.nc>`_. It stores the global temperature average (land and ocean) from 1850 to 2100 for multiple simulations (not all simulations cover the entire temporal range). Simulations are defined through 4 fields:

- ``mip_era`` : "CMIP6", "CMIP5" or "obs" (see below)
- ``source`` : The model name for GCM (same as the `source` column) and the driving model name for RCM (`driving_model` column)
- ``experiment`` : The CMIP experiment name of the run. The "historical" and "pre-industrial" experiments have been merged into each future experiment (similar to what ``match_hist_and_fut`` does in :py:func:`search_data_catalogs`)
- ``member`` : The realization variant label of the run (same as the `member` column)

An extra ``data_source`` field is also available and describes how the data has been obtained:

- "IPCC Atlas" : The timeseries was copied directly from the public data of the `IPCC Atlas <https://github.com/IPCC-WG1/Atlas/tree/main/datasets-aggregated-regionally/data>`_'
- "From Amon" : The monthly temperature average was resampled annually and averaged over the globe using a cos-lat weighting
- "From Amon with xscen" : Same, xscen was used to perform the computation.

In addition to the climate simulations, a few "observational" datasets are made available in the database. The choice of datasets and the methodology was adapted from the WMO's `State of the Global Climate 2021 <https://library.wmo.int/idurl/4/56300>`_. However, to have some consistency between these and the simulated series, an estimated 1850-1900 mean temperature was added to the WMO-compliant anomalies to get absolute values. Keep in mind that this is only an estimation, the timeseries should only be used to compute anomalies. The observational series have a short dataset name in the ``source`` field, "obs" in ``mip_era`` and ``experiment``,  and an empty ``member`` (`""`). The ``data_source`` is noted : "Computed following WMO guidelines".
