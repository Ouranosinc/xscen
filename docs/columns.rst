Columns
=======

This section presents a definition and examples for each column of a xscen DataCatalog.
The entries for the columns are based on CMIP6 metadata and the ES-DOC controlled vocabulary (https://github.com/ES-DOC/pyessv-archive).
Some columns might be left empty (with a ``NaN``), but ``id``, ``domain``,  ``processing_level`` and ``xrfreq`` are mandatory.
These four columns are what xscen uses by default to guess which entries can be merged together : all entries with the same unique combination of
the four columns will be combined in a single Dataset when any of the first three functions listed :ref:`here <opening-data>` are used.

- ``id``: Unique Dataset ID generated by ``xscen`` based on a subset of columns. By default, it is based on ``xscen.catalog.ID_COLUMNS``.
    - E.g. "ERA_ecmwf_ERA5_ERA5-Land_NAM", "CMIP6_ScenarioMIP_CMCC_CMCC-ESM2_ssp245_r1i1p1f1_global"

- ``type``: Type of data.
    - E.g. "forecast", "station-obs", "gridded-obs", "reconstruction", "simulation"

- ``processing_level``: Level of post-processing reached.
    - E.g. "raw", "extracted", "regridded", "biasadjusted"

- ``bias_adjust_institution``: Institution that computed the bias adjustment.
    - E.g. "Ouranos", "PCIC"

- ``bias_adjust_project``: Name of the project that computed the bias adjustment.
    - E.g. "ESPO-R5", "BCCAQv2"

- ``mip_era``: CMIP generation associated with the data.
    - E.g. "CMIP6", "CMIP5"

- ``activity``: Model Intercomparison Project (MIP) associated with the data. This is the same as ``activity_id`` in CMIP6 data. CMIP is the activity for the historical experiment and the DECK experiments. It can also be "MRCC5" for simulations that are not part of CORDEX.
    - E.g. "CMIP", "CORDEX", "HighResMIP"

- ``driving_model``: For RCM data, name of the driver. For CMIP6, this is only includes the model name. In CMIP5, following the ``driving_model`` convention from ES-DOC, this was in the format "institution-model".
    - E.g. "CanESM5", "CCCma-CanESM2"

- ``driving_member``:  For RCM data, member (realisation) of the driver. This looks like "rAiBpCfD" for CMIP6 (without the "fD" for CMIP5)..
    - E.g. "r1i1p1f1"

- ``institution``: Institution associated with the source.
    - E.g. "CCCma", "Ouranos", "ECMWF"

- ``source``: For simulation types, this is the model. For GCMs, this is the name of the model (`source_id` in CMIP6 and `rcm_name` in ES-DOC for CORDEX-CMIP5). For reconstruction types, this is the name of the dataset.
    - E.g. "CanESM5", "CRCM5", "ERA5", "ERA5-Land, ERA5-Preliminary"

- ``experiment``: Name of the experiment of the model.
    - E.g. "historical", "ssp245", "rcp85"

- ``member``: Name of the realisation. For CMIP6 RCMs, this is the RCM member "rX", the realization part of the ``version_realization`` CORDEX-CMIP6 spec. In CMIP5, this is the member associated with the driver (i.e. ``driving_member``).
    - E.g. "r1i1p1f1"

- ``xrfreq``: Pandas/xarray frequency.
    - E.g. "YS", "QS-DEC"

- ``frequency``: Frequency in letters (CMIP6 format).
    - E.g. "yr","qtr"

- ``variable``: Variables in the dataset. It can be a Tuple.
    - E.g. "tasmax", ("tasmax", "tasmin", "pr")

- ``domain``: Name of the region covered by the dataset. It can also contain information on the grid.
    - E.g. "global", "NAM", "ARC-44",  "ARC-22"

- ``date_start``: First date of the dataset. This usually is a Datetime object with a ms resolution.
    - E.g. "2022-06-03 00:00:00"

- ``date_end``: Last date of the dataset. This usually is a Datetime object with a ms resolution.
    - E.g. "2022-06-03 00:00:00"

- ``version``: Version of the dataset. For RCMs, this is the version of the ``version_realization`` CORDEX-CMIP6 spec.
    - E.g. "1.0" "v1"

- ``format``: Format of the dataset.
    - E.g. "zarr", "nc"

- ``path``: Path to the dataset.
    - E.g. "/some/path/to/the/data.zarr"
