=======
History
=======

v0.4.0 (unreleased)
-------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Juliette Lavoie (:user:`juliettelavoie`), Trevor James Smith (:user:`Zeitsperre`) and Pascal Bourgault (:user:`aulemahal`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New functions ``diagnostics.properties_and_measures``, ``diagnostics.measures_heatmap`` and ``diagnostics.measures_improvement``. (:issue:`5`, :pull:`54`)
* Add argument `resample_methods` to `xs.extract.resample`. (:issue:`57`, :pull:`57`)
* Added a ReadTheDocs configuration to expose public documentation. (:issue:`65`, :pull:`66`).
* ``xs.utils.stack_drop_nans``/ ``xs.utils.unstack_fill_nan`` will now format the `to_file`/`coords` string to add the domain and the shape. (:issue:`59`, :pull:`67`)
* New unstack_dates function to "extract" seasons or months from a timeseries. (:pull:`68`).
* Better spatial_mean for cases using xESMF and a shapefile with multiple polygons. (:pull:`68`).
* Yet more changes to parse_directory: (:pull:`68`).
    - Better parallelization by merging the finding and name-parsing step in the same dask tree.
    - Allow cvs for the variable columns
    - Fix parsing the variable names from datasets
    - Sort the variables in the tuples (for a more consistent output)
* In extract_dataset, add option ``ensure_correct_time`` to ensure the time coordinate matches the expected freq. Ex: monthly values given on the 15th day are moved to the 1st, as expected when asking for "MS". (:issue: `53`).
* In regrid_dataset: (:pull:`68`).
    * Allow passing skipna to the regridder kwargs.
    * Do not fail for any grid mapping problem, includin if a grid_mapping attribute mentions a variable that doesn't exist.
* Default email sent to the local user. (:pull:`68`).
* Special accelerated pathway for parsing catalogs with all dates within the datetime64[ns] range (:pull:`75`).
* New functions ``reduce_ensemble`` and ``build_reduction_data`` to support kkz and kmeans clustering (:issue:`4`, :pull:`63`)
* `ensemble_stats` can now loop through multiple statistics, support functions located in `xclim.ensembles._robustness`, and supports weighted realizations (:pull:`63`).
* New function `ensemble_stats.generate_weights` that estimates weights based on simulation metadata (:pull:`63`).
* New function `catalog.unstack_id` to reverse-engineer IDs (:pull:`63`).
* `generate_id` now accepts Datasets (:pull:`63`).

Breaking changes
^^^^^^^^^^^^^^^^
* `statistics / stats_kwargs` have been changed/eliminated in `ensemble_stats`, respectively (:pull:`63`).

Bug fixes
^^^^^^^^^
* Add a missing dependencies to the env. (pyarrow for faster string handling in catalogs) (:pull:`68`).
* Allow passing compute=False to save_to_zarr. (:pull:`68`).

Internal changes
^^^^^^^^^^^^^^^^
* Small bugfixes in aggregate.py (:pull:`55`, :pull:`56`).
* Default method of `xs.extract.resample` now depends on frequency. (:issue:`57`, :pull:`58`).
* Bugfix for `_restrict_by_resolution` with CMIP6 datasets (:pull:`71`).
* More complete check of coverage in ``_subset_file_coverage`` (:issue: `70`, :pull: `72`)
* The code that performs `common_attrs_only` in `ensemble_stats` has been moved to `clean_up` (:pull:`63`).
* Removed the default `to_level` in `clean_up` (:pull:`63`).


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
