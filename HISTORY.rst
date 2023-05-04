=======
History
=======

v0.6.0 (2023-05-04)
-------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Juliette Lavoie (:user:`juliettelavoie`), Pascal Bourgault (:user:`aulemahal`), Gabriel Rondeau-Genesse (:user:`RondeauG`).

Announcements
^^^^^^^^^^^^^
* `xscen` is now offered as a conda package available through Anaconda.org. Refer to the installation documentation for more information. (:issue:`149`, :pull:`171`).
* Deprecation: Release 0.6.0 of `xscen` will be the last version to support ``xscen.extract.clisops_subset``. Use ``xscen.spatial.subset`` instead. (:pull:`182`, :pull:`184`).
* Deprecation: The argument `region`, used in multiple functions, has been slightly reformatted. Release 0.6.0 of `xscen` will be the last version to support the old format. (:issue:`99`, :issue:`101`, :pull:`184`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New 'cos-lat' averaging in `spatial_mean`. (:issue:`94`, :pull:`125`).
* Support for computing anomalies in `compute_deltas`.  (:pull:`165`).
* Add function `diagnostics.measures_improvement_2d`. (:pull:`167`).
* Add function ``regrid.create_bounds_rotated_pole`` and automatic use in ``regrid_dataset`` and ``spatial_mean``. This is temporary, while we wait for a functionning method in ``cf_xarray``. (:pull:`174`, :issue:`96`).
* Add ``spatial`` submodule with functions ``creep_weights`` and ``creep_fill`` for filling NaNs using neighbours. (:pull:`174`).
* Allow passing ``GeoDataFrame`` instances in ``spatial_mean``'s ``region`` argument, not only geospatial file paths. (:pull:`174`).
* Allow searching for periods in `catalog.search`. (:issue:`123`, :pull:`170`).
* Allow searching and extracting multiple frequencies for a given variable. (:issue:`168`, :pull:`170`).
* New masking feature in ``extract_dataset``. (:issue:`180`, :pull:`182`).
* New function ``xs.spatial.subset`` to replace ``xs.extract.clisops_subset`` and add method "sel". (:issue:`180`, :pull:`182`).
* Add long_name attribute to diagnostics. ( :pull:`189`).
* Added a new YAML-centric notebook (:issue:`8`, :pull:`191`).
* New ``utils.standardize_periods`` to standardize that argument across multiple functions. (:issue:`87`, :pull:`192`).
* New `coverage_kwargs` argument added to ``search_data_catalogs`` to allow modifying the default values of ``subset_file_coverage``. (:issue:`87`, :pull:`192`).

Breaking changes
^^^^^^^^^^^^^^^^
* 'mean' averaging has been deprecated in `spatial_mean`. (:pull:`125`).
* 'interp_coord' has been renamed to 'interp_centroid' in `spatial_mean`. (:pull:`125`).
* The 'datasets' dimension of the output of ``diagnostics.measures_heatmap`` is renamed 'realization'. (:pull:`167`).
* `_subset_file_coverage` was renamed `subset_file_coverage` and moved to ``catalog.py`` to prevent circular imports. (:pull:`170`).
* `extract_dataset` doesn't fail when a variable is in the dataset, but not `variables_and_freqs`. (:pull:`185`).
* The argument `period`, used in multiple function, is now always a single list, while `periods` is more flexible. (:issue:`87`, :pull:`192`).
* The parameters `reference_period` and `simulation_period` of ``xscen.train`` and ``xscen.adjust`` were renamed `period/periods` to respect the point above. (:issue:`87`, :pull:`192`).

Bug fixes
^^^^^^^^^
* Forbid pandas v1.5.3 in the environment files, as the linux conda build breaks the data catalog parser. (:issue:`161`, :pull:`162`).
* Only return requested variables when using ``DataCatalog.to_dataset``. (:pull:`163`).
* ``compute_indicators`` no longer crashes if less than 3 timesteps are produced. (:pull:`125`).
* `xarray` is temporarily pinned below v2023.3.0 due to an API-breaking change. (:issue:`175`, :pull:`173`).
* `xscen.utils.unstack_fill_nan`` can now handle datasets that have non dimension coordinates. (:issue:`156`, :pull:`175`).
* `extract_dataset` now skips a simulation way earlier if the frequency doesn't match. (:pull:`170`).
* `extract_dataset` now correctly tries to extract in reverse timedelta order. (:pull:`170`).
* `compute_deltas` no longer creates all NaN values if the input dataset is in a non-standard calendar. (:pull:`188`).

Internal changes
^^^^^^^^^^^^^^^^
* `xscen` now manages packaging for PyPi and TestPyPI via GitHub workflows. (:pull:`159`).
* Pre-load coordinates in ``extract.clisops_subset`` (:pull:`163`).
* Minimal documentation for templates. (:pull:`163`).
* `xscen` is now indexed in `Zenodo <https://zenodo.org/>`_, under the `ouranos` community of projects. (:pull:`164`).
* Added a few relevant `Shields <https://shields.io/>`_ to the README.rst. (:pull:`164`).
* Better warning messages in ``_subset_file_coverage`` when coverage is insufficient. (:pull:`125`).
* The top-level Makefile now includes a `linkcheck` recipe, and the ReadTheDocs configuration no longer reinstalls the `llvmlite` compiler library. (:pull:`173`).
* The checkups on coverage and duplicates can now be skipped in `subset_file_coverage`. (:pull:`170`).
* Changed the `ProjectCatalog` docstrings to make it more obvious that it needs to be created empty. (:issue:`99`, :pull:`184`).
* Added parse_config to `creep_fill`, `creep_weights`, and `reduce_ensemble` (:pull:`191`).

v0.5.0 (2023-02-28)
-------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Juliette Lavoie (:user:`juliettelavoie`), Trevor James Smith (:user:`Zeitsperre`), Sarah Gammon (:user:`sg2475962`) and Pascal Bourgault (:user:`aulemahal`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Possibility of excluding variables read from file from the catalog produced by ``parse_directory``. (:pull:`107`).
* New functions ``extract.subset_warming_level`` and ``aggregate.produce_horizon``. (:pull:`93`).
* add `round_var` to `xs.clean_up`. (:pull:`93`).
* New "timeout_cleanup" option for ``save_to_zarr``, which removes variables that were in the process of being written when receiving a ``TimeoutException``. (:pull:`106`).
* New ``scripting.skippable`` context, allowing the use of CTRL-C to skip code sections. (:pull:`106`).
* Possibility of fields with underscores in the patterns of ``parse_directory``. (:pull:`111`).
* New ``utils.show_versions`` function for printing or writing to file the dependency versions of `xscen`. (:issue:`109`, :pull:`112`).
* Added previously private notebooks to the documentation. (:pull:`108`).
* Notebooks are now tested using `pytest` with `nbval`. (:pull:`108`).
* New ``restrict_warming_level`` argument for ``extract.search_data_catalogs`` to filter dataset that are not in the warming level csv. (:issue:`105`, :pull:`138`).
* Set configuration value programmatically through ``CONFIG.set``. (:pull:`144`).
* New ``to_dataset`` method on ``DataCatalog``. The same as ``to_dask``, but exposing more aggregation options. (:pull:`147`).
* New templates folder with one general template. (:issue:`151`, :pull:`158`).

Breaking changes
^^^^^^^^^^^^^^^^
* Functions that are called internally can no longer parse the configuration. (:pull:`133`).

Bug fixes
^^^^^^^^^
* ``clean_up`` now converts the calendar of variables that use "interpolate" in "missing_by_var" at the same time.
    - Hence, when it is a conversion from a 360_day calendar, the random dates are the same for all of the these variables. (:issue:`102`, :pull:`104`).
* ``properties_and_measures`` no longer casts month coordinates to string. (:pull:`106`).
* `search_data_catalogs` no longer crashes if it finds nothing. (:issue:`42`, :pull:`92`).
* Prevented fixed fields from being duplicated during `_dispatch_historical_to_future` (:issue:`81`, :pull:`92`).
* Added missing `parse_config` to functions in `reduce.py` (:pull:`92`).
* Added deepcopy before `skipna` is popped in `spatial_mean` (:pull:`92`).
* `subset_warming_level` now validates that the data exists in the dataset provided (:issue:`117`, :pull:`119`).
* Adapt `stack_drop_nan` for the newest version of xarray (2022.12.0). (:issue:`122`, :pull:`126`).
* Fix `stack_drop_nan` not working if intermediate directories don't exist (:issue:`128`).
* Fixed a crash when `compute_indicators` produced fixed fields (:pull:`139`).

Internal changes
^^^^^^^^^^^^^^^^
* ``compute_deltas`` skips the unstacking step if there is no time dimension and cast object dimensions to string. (:pull:`9`)
* Added the "2sem" frequency to the translations CVs. (:pull:`111`).
* Skip files we can't read in ``parse_directory``. (:pull:`111`).
* Fixed non-numpy-standard Docstrings. (:pull:`108`).
* Added more metadata to package description on PyPI. (:pull:`108`).
* Faster ``search_data_catalogs`` and ``extract_dataset`` through a faster ``DataCatalog.unique``, date parsing and a rewrite of the ``ensure_correct_time`` logic. (:pull:`127`).
* The ``search_data_catalogs`` function now accepts `str` or `pathlib.Path` variables (in addition to lists of either data type) for performing catalog lookups. (:pull:`121`).
* `produce_horizons` now supports fixed fields (:pull:`139`).
* Rewrite of ``unstack_dates`` for better performance with dask arrays. (:pull:`144`).

v0.4.0 (2022-09-28)
-------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Juliette Lavoie (:user:`juliettelavoie`), Trevor James Smith (:user:`Zeitsperre`) and Pascal Bourgault (:user:`aulemahal`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New functions ``diagnostics.properties_and_measures``, ``diagnostics.measures_heatmap`` and ``diagnostics.measures_improvement``. (:issue:`5`, :pull:`54`).
* Add argument `resample_methods` to `xs.extract.resample`. (:issue:`57`, :pull:`57`)
* Added a ReadTheDocs configuration to expose public documentation. (:issue:`65`, :pull:`66`).
* ``xs.utils.stack_drop_nans``/ ``xs.utils.unstack_fill_nan`` will now format the `to_file`/`coords` string to add the domain and the shape. (:issue:`59`, :pull:`67`).
* New unstack_dates function to "extract" seasons or months from a timeseries. (:pull:`68`).
* Better spatial_mean for cases using xESMF and a shapefile with multiple polygons. (:pull:`68`).
* Yet more changes to parse_directory: (:pull:`68`).
    - Better parallelization by merging the finding and name-parsing step in the same dask tree.
    - Allow cvs for the variable columns.
    - Fix parsing the variable names from datasets.
    - Sort the variables in the tuples (for a more consistent output)
* In extract_dataset, add option ``ensure_correct_time`` to ensure the time coordinate matches the expected freq. Ex: monthly values given on the 15th day are moved to the 1st, as expected when asking for "MS". (:issue: `53`).
* In regrid_dataset: (:pull:`68`).
    * Allow passing skipna to the regridder kwargs.
    * Do not fail for any grid mapping problem, includin if a grid_mapping attribute mentions a variable that doesn't exist.
* Default email sent to the local user. (:pull:`68`).
* Special accelerated pathway for parsing catalogs with all dates within the datetime64[ns] range. (:pull:`75`).
* New functions ``reduce_ensemble`` and ``build_reduction_data`` to support kkz and kmeans clustering. (:issue:`4`, :pull:`63`).
* `ensemble_stats` can now loop through multiple statistics, support functions located in `xclim.ensembles._robustness`, and supports weighted realizations. (:pull:`63`).
* New function `ensemble_stats.generate_weights` that estimates weights based on simulation metadata. (:pull:`63`).
* New function `catalog.unstack_id` to reverse-engineer IDs. (:pull:`63`).
* `generate_id` now accepts Datasets. (:pull:`63`).
* Add `rechunk` option to `properties_and_measures` (:pull:`76`).
* Add `create` argument to `ProjectCatalog` (:issue:`11`, :pull:`77`).
* Add percentage deltas to `compute_deltas` (:issue:`82`, :pull:`90`).

Breaking changes
^^^^^^^^^^^^^^^^
* `statistics / stats_kwargs` have been changed/eliminated in `ensemble_stats`, respectively. (:pull:`63`).

Bug fixes
^^^^^^^^^
* Add a missing dependencies to the env (`pyarrow`, for faster string handling in catalogs). (:pull:`68`).
* Allow passing ``compute=False`` to `save_to_zarr`. (:pull:`68`).

Internal changes
^^^^^^^^^^^^^^^^
* Small bugfixes in `aggregate.py`. (:pull:`55`, :pull:`56`).
* Default method of `xs.extract.resample` now depends on frequency. (:issue:`57`, :pull:`58`).
* Bugfix for `_restrict_by_resolution` with CMIP6 datasets (:pull:`71`).
* More complete check of coverage in ``_subset_file_coverage``. (:issue:`70`, :pull:`72`)
* The code that performs ``common_attrs_only`` in `ensemble_stats` has been moved to `clean_up`. (:pull:`63`).
* Removed the default ``to_level`` in `clean_up`. (:pull:`63`).
* `xscen` now has an official logo. (:pull:`69`).
* Use numpy max and min in `properties_and_measures` (:pull:`76`).
* Cast catalog date_start and date_end to "%4Y-%m-%d %H:00" when writing to disk. (:issue:`83`, :pull:`79`)
* Skip test of coverage on the sum if the list of select files is empty. (:pull:`79`)
* Added missing CMIP variable names in conversions.yml and added the ability to provide a custom file instead (:issue:`86`, :pull:`88`)
* Changed 'allow_conversion' and 'allow_resample' default to False in search_data_catalogs (:issue:`86`, :pull:`88`)

v0.3.0 (2022-08-23)
-------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Juliette Lavoie (:user:`juliettelavoie`), Trevor James Smith (:user:`Zeitsperre`) and Pascal Bourgault (:user:`aulemahal`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New function ``clean_up`` added. (:issue:`22`, :pull:`25`).
* `parse_directory`: Fixes to `xr_open_kwargs` and support for wildcards (*) in the directories. (:pull:`19`).
* New function ``xscen.ensemble.ensemble_stats`` added. (:issue:`3`, :pull:`28`).
* New functions ``spatial_mean``, ``climatological_mean`` and ``deltas`` added. (:issue:`4`, :pull:`35`).
* Add argument ``intermediate_reg_grids`` to ``xscen.regridding.regrid``. (:issue:`34`, :pull:`39`).
* Add argument ``moving_yearly_window`` to ``xscen.biasadjust.adjust``. (:pull:`39`).
* Many adjustments to ``parse_directory``: better wildcards (:issue:`24`), allow custom columns, fastpaths for ``parse_from_ds``, and more (:pull:`30`).
* Documentation now makes better use of autodoc to generate package index. (:pull:`41`).
* `periods` argument added to `compute_indicators` to support datasets with jumps in time (:pull:`35`).

Breaking changes
^^^^^^^^^^^^^^^^
* Patterns in ``parse_directory`` start at the end of the paths in ``directories``. (:pull:`30`).
* Argument ``extension`` of ``parse_directory`` has been renamed ``globpattern``. (:pull:`30`).
* The ``xscen`` API and filestructure have been significantly refactored. (:issue:`40`, :pull:`41`). The following functions are available from the top-level:
    - ``adjust``, ``train``, ``ensemble_stats``, ``clisops_subset``, ``dispatch_historical_to_future``, ``extract_dataset``, ``resample``, ``restrict_by_resolution``, ``restrict_multimembers``, ``search_data_catalogs``, ``save_to_netcdf``, ``save_to_zarr``, ``rechunk``, ``compute_indicators``, ``regrid_dataset``, and ``create_mask``.
* xscen now requires geopandas and shapely (:pull:`35`).
* Following a change in intake-esm xscen now uses "cat:" to prefix the dataset attributes extracted from the catalog. All catalog-generated attributes should now be valid when saving to netCDF. (:issue:`13`, :pull:`51`).

Internal changes
^^^^^^^^^^^^^^^^
* `parse_directory`: Fixes to `xr_open_kwargs`. (:pull:`19`).
* Fix for indicators removing the 'time' dimension. (:pull:`23`).
* Security scanning using CodeQL and GitHub Actions is now configured for the repository. (:pull:`21`).
* Bumpversion action now configured to automatically augment the version number on each merged pull request. (:pull:`21`).
* Add ``align_on = 'year'`` argument in bias adjustment converting of calendars. (:pull:`39`).
* GitHub Actions using Ubuntu-22.04 images are now configured for running testing ensemble using `tox-conda`. (:pull:`44`).
* `import xscen` smoke test is now run on all pull requests. (:pull:`44`).
* Fix for `create_mask` removing attributes (:pull:`35`).

v0.2.0 (first official release)
-------------------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Pascal Bourgault (:user:`aulemahal`), Trevor James Smith (:user:`Zeitsperre`), Juliette Lavoie (:user:`juliettelavoie`).

Announcements
^^^^^^^^^^^^^
* This is the first official release for xscen!

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Supports workflows with YAML configuration files for better transparency, reproducibility, and long-term backups.
* Intake_esm-based catalog to find and manage climate data.
* Climate dataset extraction, subsetting, and temporal aggregation.
* Calculate missing variables through Intake-esm's DerivedVariableRegistry.
* Regridding with xESMF.
* Bias adjustment with xclim.

Breaking changes
^^^^^^^^^^^^^^^^
* N/A

Internal changes
^^^^^^^^^^^^^^^^
* N/A
