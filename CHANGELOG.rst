=========
Changelog
=========

..
    `Unreleased <https://github.com/Ouranosinc/xscen>`_ (latest)
    ------------------------------------------------------------
    Contributors:

    Changes
    ^^^^^^^
    * No change.

    Fixes
    ^^^^^
    * No change.

.. _changes_0.13.1:

`v0.13.1 <https://github.com/Ouranosinc/xscen/tree/0.13.1>`_ (2025-10-21)
-------------------------------------------------------------------------
Contributors to this version: Juliette Lavoie (:user:`juliettelavoie`), Trevor James Smith (:user:`Zeitsperre`), Gabriel Rondeau-Genesse (:user:`RondeauG`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New official column ``bias_adjust_reference``. When a bias_adjust_project has multiple references (e.g., CanLead, ESPO6 v2.0), the information is stored in this column. (:pull:`643`, :pull:`644`).
* Add ``clip_var`` option in ``xs.clean_up`` to clip values outside a given range. (:pull:`645`).

Bug fixes
^^^^^^^^^
* Fixed a bug in ``ProjectCatalog.update`` where the function would crash on Windows systems. (:pull:`656`).

Breaking changes
^^^^^^^^^^^^^^^^
* Pins have been added for `h5py` (``>=3.12.1``) and `h5netcdf` (``>=1.5.0``) to ensure that modern versions are preferably installed. (:pull:`658`).

Internal changes
^^^^^^^^^^^^^^^^
* Updated the cookiecutter template to the latest version. (:pull:`651`):
    * Updated the Contributor Covenant agreement to v3.0.
    * Replaced `black`, `blackdoc`, and `isort` with `ruff`.
    * Added a `CITATION.cff` file.
    * `pyproject.toml` is now `PEP 639 <https://peps.python.org/pep-0639/>`_-compliant.
* Pinned `pydantic` below v2.12.0 for compatibility issues with `intake-esm`
* When running tests with `tox`, the `h5py` library is now compiled from source. This requires the `hdf5` (conda) or `libhdf5-dev` (system) packages to be installed. (:pull:`658`).

.. _changes_0.13.0:

v0.13.0 (2025-09-03)
--------------------
Contributors to this version: Juliette Lavoie (:user:`juliettelavoie`), Pascal Bourgault (:user:`aulemahal`), Artem Buyalo (:user:`ArtemBuyalo`), Éric Dupuis (:user:`coxipi`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Add `additive_space` option to ``xs.train``. (:pull:`603`).
* Modify the xclim modules definition of ``relative_humidity_from_dewpoint`` to include ``invalid_values='clip'``.(:pull:`616`).
* Add possibility to "creep fill" iteratively with argument ``steps`` in ``xs.spatial.creep_weights``. (:pull:`594`).
* Ability to save and load sparse arrays like the creep or regridding weights to disk with ``xs.io.save_sparse``  and ``xs.io.load_sparse``. (:pull:`594`).
* Generalize ``xs.regrid.create_bounds_gridmapping`` to include dataset with `crs`. (:pull:`628`, :issue:`627`).
* Ability to return multiple periods if passed multiple warming levels in ``xs.extract.get_period_from_warming_level``. (:pull:`630`, :issue:`629`).
* Update xscen to xclim 0.58 (:pull:`634`).
* New function ``xs.spatial.rotate_vectors`` to rotate vectors from/to their native grid axes to/from real west-east/south-north axes. (:pull:`635`).
* New function ``xs.spatial.get_crs`` to get a cartopy crs from a grid mapping variable (only Rotated Pole and Oblique Mercator) (:pull:`635`).

Bug fixes
^^^^^^^^^
* Add standard_name to dtr definition in conversions. (:pull:`611`).
* Better handling of attributes in ``xs.train``. (:pull:`608`, :issue:`607`)
* Fix dimension renaming in ``xs.spatial_mean``. (:pull:`620`)

Bug fixes
^^^^^^^^^
* Fixed `xs.utils.xclim_convert_units_to` context patching. (:pull:`604`).

.. _changes_0.12.3:

v0.12.3 (2025-05-26)
--------------------
Contributors to this version: Juliette Lavoie (:user:`juliettelavoie`), Sarah Gammon (:user:`SarahG-579462`).

Bug fixes
^^^^^^^^^
* Fixed bugs in template 1. (:pull:`595`).

Internal changes
^^^^^^^^^^^^^^^^
* Updated ``xclim`` to v0.57.0. (:pull:`596`).

.. _changes_0.12.2:

v0.12.2 (2025-05-16)
--------------------
Contributors to this version: Juliette Lavoie (:user:`juliettelavoie`), Éric Dupuis (:user:`coxipi`), Gabriel Rondeau-Genesse (:user:`RondeauG`).

Breaking changes
^^^^^^^^^^^^^^^^
* Remove `adapt_freq` argument from `xs.train`. This argument should be passed to xsdba directly. (:pull:`589`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Add annual global tas timeseries for CMIP6's model UKESM1-0-LL ssp585 r4i1p1f2 (:pull:`573`).
* Add `strict_units` option to health checks. (:pull:`574`, :issue:`574`).
* Add 10yr and 30yr frequencies support in CVs. (:pull:`576`).
* New converters `hurslogit_from_hurs` and `hurs_from_hurslogit`. (:pull:`590`).

Bug fixes
^^^^^^^^^
* ``xs.utils.change_units`` will now always check that the units returned are exactly the same as the requested units. (:pull:`578`).
* Fixed a bug in ``xs.catalog.subset_file_coverage`` where the function could not process dates after 2262. (:issue:`587`, :pull:`591`).

.. _changes_0.12.1:

v0.12.1 (2025-04-07)
--------------------
Contributors to this version: Juliette Lavoie (:user:`juliettelavoie`), Gabriel Rondeau-Genesse (:user:`RondeauG`), Éric Dupuis (:user:`coxipi`), Pascal Bourgault (:user:`aulemahal`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* `xscen` officially supports Python 3.13. (:pull:`551`).

Breaking changes
^^^^^^^^^^^^^^^^
* Make `strip_cat_metadata` False by default in ``xs.save_to_zarr`` and ``xs.save_to_netcdf``. (:pull:`556`, :issue:`555`).
* New official column ``driving_member``. For RCMs, this should store the driver's realisation number, while the ``member`` column should now store the RCM simulation's realisation number, noted as "rX". This ``member`` should approximately map to the "realization" part of CORDEX-CMIP6's "version_realization" facet (the version part mapping to the already existing ``version`` column). The member restricting feature of ``search_data_catalogs`` has been adapted, but continues to work with catalogs missing the ``driving_member`` column. (:pull:`559`).
* Also adapted from the CORDEX-CMIP6 specifications, the ``driving_model`` column does not need to indicate the driver's institution name anymore. (:pull:`559`).
* For Python 3.13 support, `xscen` now requires `clisops>=0.16.1` and `xsdba>=0.4.0`. (:pull:`551`).
* Minimum required `intake-esm` has been updated to `>=2025.2.3`. (:pull:`551`).
* Temporarily pinned `numcodecs` to `<0.16.0` for compatibility with `zarr`. (:pull:`571`).

Bug fixes
^^^^^^^^^
* Fixed the default for ``xs.utils.maybe_unstack``. (:pull:`553`).
* Patch ``xsdba.units.convert_units_to`` with ``xclim.core.units.convert_units_to`` with `context="infer"` locally in ``xs.train`` and ``xs.adjust`` instead of using ``xc.core.settings.context``. (:pull:`552`).
* Fixed a bug in ``xs.utils.clean_up`` where attributes would be dropped when using the `missing_by_vars` argument. (:pull:`569`, :issue:`568`).
* Allow undetectable frequencies in ``xs.extract.resample``. (:pull:`567`).

Internal changes
^^^^^^^^^^^^^^^^
* Added the ability to test `xESMF`-related functions with `tox / pip`. (:pull:`554`).
* Updated the pins for `xclim`, `xarray`, `dask`, and `rechunker`. (:pull:`570`).
* More accurate listing of dependencies for the project in `pyproject.toml` and `environment*.yml`. (:pull:`557`).
* `sphinx` dependencies are more streamlined in the `docs` environment. (:pull:`557`).
* Added `codespell`, `deptry`, `vulture`, and `yamllint` to the linting checks. (:pull:`557`).

.. _changes_0.12.0:

v0.12.0 (2025-03-10)
--------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Pascal Bourgault (:user:`aulemahal`), Juliette Lavoie (:user:`juliettelavoie`), Sarah Gammon (:user:`SarahG-579462`), Éric Dupuis (:user:`coxipi`).

Breaking changes
^^^^^^^^^^^^^^^^
* `xscen` now uses `flit` as its build-engine and no longer uses `setuptools`, `setuptools-scm`, or `wheel`. (:pull:`519`).
* Update to support Python3.13 and `xclim` v0.55.0 (:pull:`532`).
* `xscen` now requires the `xsdba` package for bias adjustment functionality (replacement for `xclim.sdba`). (:pull:`530`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Include station-obs and forecasts in the derived schema for `build_path`. (:pull:`534`).
* Project catalog now allows `check_valid` and `drop_duplicates` keyword arguments. (:pull:`536`, :issue:`535`).
* Add annual global tas timeseries for CMIP6's models CanESM5 r1i1p2f1 (ssp126, ssp245, ssp370, ssp585), MPI-ESM1-2-LR ssp370 (r2i1p1f1, r3i1p1f1, r4i1p1f1, r5i1p1f1) (:pull:`544`).
* Allow ``pd.Timestamp`` and more precise datetime strings for ``xs.search_data_catalogs`` and ``dc.search``. (:pull:`547`, :issue:`546`).
* ``xscen.train`` now accepts a list of two or more variables, in which case, the variables are stacked. ``xsdba.MBCn`` is supported. (:pull:`548`).

Bug fixes
^^^^^^^^^
* Docstrings and documentation have been updated to remove several small grammatical errors. (:pull:`527`).

Internal changes
^^^^^^^^^^^^^^^^
* Updated the cookiecutter template to the latest commit. (:pull:`527`):
    * Updated versions of many GitHub Actions and Python dependencies.
    * Removed `coveralls` from the CI dependencies.
    * Added `pre-commit` hooks for `vulture` (dead code) and `codespell` (typos).
* The minimum supported `clisops` version has been raised to v0.15.0. (:pull:`533`).
* Dependency pins have been synchronized across the repository. (:pull:`533`).
* GitHub Workflows for conda builds now use the `coverallsapp/github-action` action for coverage reporting. (:pull:`533`).
* `xsdba` is now used instead of `xclim.sdba`. (:pull:`530`).

.. _changes_0.11.0:

v0.11.0 (2025-01-23)
--------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Juliette Lavoie (:user:`juliettelavoie`), Trevor James Smith (:user:`Zeitsperre`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Improve ``xs.ensembles.build_partition_data``. (:pull:`504`).

Breaking changes
^^^^^^^^^^^^^^^^
* ``xs.utils.publish_release_notes`` and ``xs.utils.show_versions`` have been moved to ``xs.testing``. (:pull:`492`).
* The previously-deprecated ``xs.reduce`` module has been removed. Refer to ``xs.ensembles.make_criteria`` and ``xs.ensembles.reduce_ensemble`` for replacement functionality. (:pull:`517`).

Bug fixes
^^^^^^^^^
* Added a missing library (``openpyxl``) to the requirements. (:pull:`492`).
* Fixed a bug in ``xs.io.subset_maxsize`` where the function would drop the last year. (:pull:`492`).
* Fixed a bug in ``xs.io.clean_incomplete`` where the `.zmetadata` file was not removed. (:pull:`492`).
* Fixed a bug in the saving of datasets where encoding was sometimes not applied, resulting for example in rechunking not being respected. (:pull:`492`).
* Fixed multiple bugs in ``xs.io.save_to_zarr`` with `mode='a'`. (:pull:`492`).
* Fixed a few minor bugs in ``xs.io.save_to_table``. (:pull:`492`).

Internal changes
^^^^^^^^^^^^^^^^
* Added a new parameter `latest` to ``xs.testing.publish_release_notes`` to only print the latest release notes. (:pull:`492`).
* The estimation method in ``xs.io.estimate_chunks`` has been improved. (:pull:`492`).
* A new parameter `incomplete` has been added to ``xs.io.clean_incomplete`` to remove incomplete variables. (:pull:`492`).
* Continued work on adding tests. (:pull:`492`).
* Modified a CI build to test against the oldest supported version of `xclim`. (:pull:`505`).
* Updated the cookiecutter template version: (:pull:`507`)
    * Added `vulture` to pre-commit hooks (finding dead code blocks).
    * Added `zizmor` to the pre-commit hooks (security analysis for CI workflows).
    * Secured token usages on all workflows (using `zizmor`).
    * Simplified logic in ``bump-version.yml``.
    * Synchronized a few dependencies.
* Fixed a few socket blocks and configuration issues in the CI workflows. (:pull:`512`).
* Added Open Source Security Foundation Best Practices badge, Zenodo DOI badge, FOSSA license compliance badge to the README. (:pull:`514`).
* Several deprecated usages within the code base have been addressed. The number of warnings emitted from the test suite have been significantly reduced. (:issue:`515`, :pull:`516`).

.. _changes_0.10.1:

v0.10.1 (2024-11-04)
--------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Pascal Bourgault (:user:`aulemahal`), Éric Dupuis (:user:`coxipi`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``xs.io.make_toc`` now includes the global attributes of the dataset after the information about the variables. (:pull:`473`).
* New function ``xs.get_warming_level_from_period`` to get the warming level associated with a given time horizon. (:pull:`474`).
* Added ability to skip whole folders to ``xs.parse_directory`` with argument ``skip_dirs``. (:pull:`478`, :pull:`479`).
* `diagnostics.measures_improvement` now accepts `dim`, which specifies `dimension(s)` on which the proportion of improved pixels are computed. (:pull:`416`)
* The argument `indicators` in ``xs.produce_horizon`` is now optional. Added an argument `op` to control the climatological operation. (:pull:`483`).

Breaking changes
^^^^^^^^^^^^^^^^
* ``xs.get_warming_level`` has been renamed to ``xs.get_period_from_warming_level``. Its argument `return_horizon` was reversed and renamed `return_central_year` (:pull:`474`).
* Removed support for the deprecated `xclim` function `change_significance` in `ensemble_stats`. (:pull:`482`).
* The argument `indicators` in ``xs.produce_horizon`` is no longer positional. (:pull:`483`).

Bug fixes
^^^^^^^^^
* ``xs.io.save_to_table`` now correctly handles the case where the input is a `DataArray` or a `Dataset` with a single variable. (:pull:`473`).
* Fixed a bug in ``xs.utils.change_units`` where the original dataset was also getting modified. (:pull:`482`).
* Fixed a bug in ``xs.compute_indicators`` where the `cat:variable` attribute was not correctly set. (:pull:`483`).
* Fixed a bug in ``xs.climatological_op`` where kwargs were not passed to the operation function. (:pull:`486`).
* Fixed a bug in ``xs.climatological_op`` where `min_periods` was not passed when the operation was `linregress`. (:pull:`486`).

Internal changes
^^^^^^^^^^^^^^^^
* Include CF convention for temperature differences and on scale (:pull:`428`, :issue:`428`).
* Bumped the version of `xclim` to 0.53.2. (:pull:`482`).
* More tests added. (:pull:`486`).
* Fixed a bug in ``xs.testing.datablock_3d`` where some attributes of the rotated pole got reversed half-way through the creation of the dataset. (:pull:`486`).
* The function ``xs.regrid._get_grid_mapping`` was moved to ``xs.spatial.get_grid_mapping`` and is now a public function. (:pull:`486`).

.. _changes_0.10.0:

v0.10.0 (2024-09-30)
--------------------
Contributors to this version: Juliette Lavoie (:user:`juliettelavoie`), Pascal Bourgault (:user:`aulemahal`), Gabriel Rondeau-Genesse (:user:`RondeauG`), Trevor James Smith (:user:`Zeitsperre`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* The `mask` argument in ``stack_drop_nans`` can now be a list of dimensions. In that case, a `dropna(how='all')` operation will be used to create the mask on-the-fly. (:pull:`450`).
* Few changes to ``clean_up``:
    * The `convert_calendar` function now uses `xarray` instead of `xclim`. (:pull:`450`).
    * The `attrs_to_remove` and `remove_all_attrs_except` arguments now use real regex. (:pull:`450`).
    * Multiple entries can now be given for `change_attr_prefix`. (:pull:`450`).
* ``minimum_calendar`` now accepts a list as input. (:pull:`450`).
* More calendars are now recognized in ``translate_time_chunk``. (:pull:`450`).
* `new_dim` in ``unstack_dates`` is now None by default and changes depending on the frequency. It becomes `month` if the data is exactly monthly, and keep the old default of `season` otherwise. (:pull:`450`).
* Updated the list of libraries in `show_versions` to reflect our current environment. (:pull:`450`).
* New ``xscen.catutils.patterns_from_schema`` to generate all possible patterns from a given schema (or one of xscen's default), to use with :py:func:`parse_directory`. (:pull:`431`).
* New ``DataCatalog.copy_files`` to copy all files of catalog to a new destination, unzipping if needed and returning a new catalog. (:pull:`431`).
* Convenience functions ``xs.io.zip_directory`` and ``xs.io.unzip_directory`` (for zarrs). (:pull:`431`).
* New argument ``compute_indicators``: ``rechunk_input`` to rechunk the inputs to resample-appropriate chunks before calling xclim. (:pull:`431`).
* New ``xs.indicators.get_indicator_outputs`` to retrieve what variable name(s) and frequency to expect from an xclim indicator. (:pull:`431`).
* `xscen` now supports launches tests from `pytest` with the `--numprocesses` option. See the `pytest-xdist documentation <https://pytest-xdist.readthedocs.io/en/stable/>`_ for more information. (:pull:`464`).
* Conservative regridding now supports oblique mercator projections. (:pull:`467`).
* The automatic name for the weight file in ``regrid_dataset`` is now more explicit to avoid errors, but now requires `cat:id` and `cat:domain` arguments for both the source and target datasets. (:pull:`467`).

Breaking changes
^^^^^^^^^^^^^^^^
* Version facet is now optional in default filepath schemas for non-simulations a with "source_version" level. (:issue:`500`, :pull:`501`).
* Catalog attributes are removed by default in ``save_to_zarr`` and ``save_to_netcdf``. Catalog attributes are those added from the catalog columns by ``to_dataset``, ``to_dataset_dict`` and ``extract_dataset``, which have names prefixed with ``cat:``. (:issue:`499`, :pull:`501`).

Bug fixes
^^^^^^^^^
* Fixed bug with reusing weights. (:issue:`411`, :pull:`414`).
* Fixed bug in `update_from_ds` when "time" is a coordinate, but not a dimension. (:pull: `417`).
* Avoid modification of mutable arguments in ``search_data_catalogs`` (:pull:`413`).
* ``ensure_correct_time`` now correctly handles cases where timesteps are missing. (:pull:`440`).
* If using the argument `tile_buffer` with a `shape` method in ``spatial.subset``, the shapefile will now be reprojected to a WGS84 grid before the buffer is applied. (:pull:`440`).
* ``maybe_unstack`` now works if the dimension name is not the default. (:pull:`450`).
* ``unstack_fill_nan`` now works if given a dictionary that contains both dimensions and coordinates. (:pull:`450`).
* ``clean_up`` no longer modifies the original dataset. (:pull:`450`).
* ``unstack_dates`` now works correctly for yearly datasets when `winter_starts_year=True`, as well as multi-year datasets. (:pull:`450`).
* Fix ``xs.catalog.concat_data_catalogs`` for catalogs that have not been search yet. (:pull:`431`).
* Fix indicator computation using ``freq=2Q*`` by assuming this means a semiannual frequency anchored at the given month (pandas assumes 2 quarter steps, any of them anchored at the given month). (:pull:`431`).
* ``create_bounds_rotated_pole`` now uses the default value if the dataset has no `north_pole_grid_longitude` attribute, instead of crashing. (:pull:`455`).
* Rewrote the global tas data file with latest HDF5/h5py to avoid errors when using h5py 3.11 and hdf5 1.14.2. (:pull:`1861`).
* Remove reference of deprecated xclim functions (``convert_calendar``, ``get_calendar``) and adapt the code for supporting xclim 0.52.2 and its subsequent development version. (:pull:`465`).

Breaking changes
^^^^^^^^^^^^^^^^
* `convert_calendar` in ``clean_up`` now uses `xarray` instead of `xclim`. Keywords aren't compatible between the two, but given that `xclim` will abandon its function, no backwards compatibility was sought. (:pull:`450`).
* `attrs_to_remove` and `remove_all_attrs_except` in ``clean_up`` now use real regex. It should not be too breaking since a `fullmatch()` is used, but `*` is now `.*`. (:pull:`450`).
* Python 3.9 is no longer supported. (:pull:`456`).
* Functions and arguments that were deprecated in `xscen` v0.8.0 or earlier have been removed. (:pull:`461`).
* `pytest-xdist` is now a development dependency. (:pull:`464`).
* ``xs.regrid.create_bounds_rotated_pole`` has been renamed to ``xs.regrid.create_bounds_gridmapping``. (:pull:`467`).
* The `weights_location` argument in ``regrid_dataset`` is no longer positional. (:pull:`467`).
* The ``xs.regrid.create_mask`` function now requires explicit arguments instead of a dictionary. (:pull:`467`).

Internal changes
^^^^^^^^^^^^^^^^
* ``DataCatalog.to_dataset`` can now accept a ``preprocess`` argument even if ``create_ensemble_on`` is given. The user assumes calendar handling. (:pull:`431`).
* Include domain in `weight_location` in ``regrid_dataset``. (:pull:`414`).
* Added pins to `xarray`, `xclim`, `h5py`, and `netcdf4`. (:pull:`414`).
* Add ``.zip`` and ``.zarr.zip`` as possible file extensions for Zarr datasets. (:pull:`426`).
* Explicitly assign coords of multiindex in `xs.unstack_fill_nan`. (:pull:`427`).
* French translations are compiled offline. A new check ensures no PR are merged with missing messages. (:issue:`342`, :pull:`443`).
* Continued work to add tests. (:pull:`450`).
* Updated the cookiecutter template via `cruft`: (:pull:`452`)
    * GitHub Workflows that use rely on `PyPI`-based dependencies now use commit hashes.
    * `Dependabot` will now group updates by type.
    * Dependencies have been updated and synchronized.
    * Contributor guidance documentation has been adjusted.
    * `numpydoc-validate` has been added to the linting tools.
    * Linting checks are more reliant on `ruff` suggestions and stricter.
    * `flake8-alphabetize` has been replaced by `ruff`.
    * License information has been updated in the library top-level `__init__.py`.
* Docstrings have been adjusted to meet the `numpydoc` standard. (:pull:`452`).

CI changes
^^^^^^^^^^
* The `bump-version.yml` workflow now uses the Ouranosinc GitHub Helper Bot to sign bump version commits. (:pull:`462`).

.. _changes_0.9.1:

v0.9.1 (2024-06-04)
-------------------
Contributors to this version: Pascal Bourgault (:user:`aulemahal`), Trevor James Smith (:user:`Zeitsperre`), Juliette Lavoie (:user:`juliettelavoie`).

Breaking changes
^^^^^^^^^^^^^^^^
* `xscen` now uses a `src layout <https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/>`_ in lieu of a flat layout. (:pull:`407`).

Bug fixes
^^^^^^^^^
* Fixed defaults for ``xr_combine_kwargs`` in ``extract_dataset`` (:pull:`402`).
* Fixed bug with `xs.utils.update_attr`(:issue:`404`, :pull:`405`).
* Fixed template 1 bugs due to changes in dependencies. ( :pull:`405`).

Internal changes
^^^^^^^^^^^^^^^^
* `cartopy` has been pinned above version '0.23.0' in order to address a licensing issue. (:pull:`403`).
* The cookiecutter template has been updated to the latest commit via `cruft`. (:pull:`407`).
    * GitHub Workflows now point to commits rather than tags.
    * `Dependabot` will now only update on a monthly schedule.
    * Dependencies have been updated and synchronized.
    * ``CHANGES.rst`` is now ``CHANGELOG.rst`` (see: ` KeepAChangelog <https://keepachangelog.com/en/1.0.0/>`_).
    * The ``CODE_OF_CONDUCT.rst`` file adapted to `Contributor Covenant v2.1 <https://www.contributor-covenant.org/version/2/1/code_of_conduct/>`_.
    * Maintainer-specific directions are now found under ``releasing.rst``

.. _changes_0.9.0:

v0.9.0 (2024-05-07)
-------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Pascal Bourgault (:user:`aulemahal`), Gabriel Rondeau-Genesse (:user:`RondeauG`), Juliette Lavoie (:user:`juliettelavoie`), Marco Braun (:user:`vindelico`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``xs.reduce_ensemble`` will now call ``xclim.ensembles.create_ensemble`` and ``xclim.ensembles.make_critera`` if required. (:pull:`386`).

Breaking changes
^^^^^^^^^^^^^^^^
* Removed support for the old instances of the `region` argument in ``spatial_mean``, ``extract_dataset``, and ``subset``. (:pull:`367`).
* Removed ``xscen.extract.clisops_subset``. (:pull:`367`).
* ``dtr`` (the function) was renamed to ``dtr_from_minmax`` to avoid confusion with the `dtr` variable. (:pull:`372`).
* The ``xscen.reduce`` module has been abandoned. (:pull:`386`).
    * ``build_reduction_data`` has been made redundant by ``xclim.ensembles.make_critera`` and will be removed in a future release.
    * ``xscen.reduce.reduce_ensemble`` has been moved to ``xscen.ensembles.reduce_ensemble``, as a module was no longer necessary.

Internal changes
^^^^^^^^^^^^^^^^
* Modified ``xscen.utils.change_unit`` to always adopt the name from the `variables_and_units dictionary` if the physical units are equal but their names are not (ex. degC <-> ˚C) (:pull:`373`).
* Updated the `cookiecutter` template to the latest version. (:pull:`358`):
    * Addresses a handful of misconfigurations in the GitHub Workflows.
    * Added a few free `grep`-based hooks for finding unwanted artifacts in the code base.
    * Updated `ruff` to v0.2.0 and `black` to v24.2.0.
* Added more tests. (:pull:`366`, :pull:`367`, :pull:`372`).
* Refactored ``xs.spatial.subset`` into smaller functions. (:pull:`367`).
* An `encoding` argument was added to ``xs.config.load_config``. (:pull:`370`).
* Various small fixes to the code to address FutureWarnings. (:pull:`380`).
* ``xs.spatial.subset`` will try to guess CF coordinate if it can't find "latitude" or "longitude" in ``ds.cf``. (:pull:`384`).
* ``xs.extract_dataset`` and ``xs.DataCatalog.to_dataset`` will now default to opening datasets with option ``chunks={}``, which tries to respect chunking on disk. (:pull:`398`, :issue:`368`).

Bug fixes
^^^^^^^^^
* Fix ``unstack_dates`` for the new frequency syntax introduced by pandas v2.2. (:pull:`359`).
* ``subset_warming_level`` will not return partial subsets if the warming level is reached at the end of the timeseries. (:issue:`360`, :pull:`359`).
* Loading of training in `adjust` is now done outside of the periods loop. (:pull:`366`).
* Fixed bug for adding the preprocessing attributes inside the `adjust` function. (:pull:`366`).
* Fixed a bug to accept `group = False` in `adjust` function. (:pull:`366`).
* `creep_weights` now correctly handles the case where the grid is small, `n` is large, and `mode=wrap`. (:issue:`367`).
* Fixed a bug in ``tasmin_from_dtr`` and ``tasmax_from_dtr``, when `dtr` units differed from tasmin/max. (:pull:`372`).
* Fixed a bug where the requested chunking would be ignored when saving a dataset (:pull:`379`).
* The missing value check in ``health_checks`` will no longer crasg if a variable has no time dimension. (:pull:`382`).

.. _changes_0.8.3:

v0.8.3 (2024-02-28)
-------------------
Contributors to this version: Juliette Lavoie (:user:`juliettelavoie`), Trevor James Smith (:user:`Zeitsperre`), Gabriel Rondeau-Genesse (:user:`RondeauG`), Pascal Bourgault (:user:`aulemahal`).

Announcements
^^^^^^^^^^^^^
* `xscen` now has a `security disclosure policy <https://github.com/Ouranosinc/xscen/tree/main?tab=security-ov-file#security-ov-file>`_. (:pull:`353`).
* Various frequency-related changes to match the new `pandas` naming conventions. (:pull:`351`).

Internal changes
^^^^^^^^^^^^^^^^
* Added tests for diagnostics. (:pull:`352`).
* Added a `SECURITY.md` file to the repository and the documentation. (:pull:`353`).
* Added `tox` modifier for testing builds against the `main` development branch of `xclim`. (:pull:`351`, :pull:`355`).
* Added a `requirements_upstream.txt` file to the repository to track the development branches of relevant dependencies. (:pull:`355`).
* Added a dedicated GitHub Workflow to evaluate compatibility with upstream dependencies. (:pull:`355`).

Breaking changes
^^^^^^^^^^^^^^^^
* `xscen` now requires `pandas` >= 2.2 and `xclim` >= 0.48.2. (:pull:`351`).
* Functions that output a dict with keys as xrfreq (such as ``extract_dataset``, ``compute_indicators``) will now return the new nomenclature (e.g. ``"YS-JAN"`` instead of ``"AS-JAN"``). (:pull:`351`).
* Going from `xrfreq` to frequencies or timedeltas will still work, but the opposite (frequency --> xrfreq/timedelta) will now only result in the new `pandas` nomenclature. (:pull:`351`).

.. _changes_0.8.2:

v0.8.2 (2024-02-12)
-------------------
Contributors to this version: Trevor James Smith (:user:`Zeitsperre`), Pascal Bourgault (:user:`aulemahal`)

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Added a new argument ``indicators_kw`` to ``xs.ensembles.build_partition_data``. (:pull:`315`).
* `xscen` is `Semantic Versioning 2.0.0 <https://semver.org/spec/v2.0.0.html>`_ compliant. (:pull:`319`).
* `xesmf` made an optional dependency, making `xscen` easier to install with `pip`. (:pull:`337`).

Internal changes
^^^^^^^^^^^^^^^^
* Granular permissions and dependency scanning actions have been added to all GitHub CI Workflows. (:pull:`313`).
* Updated the list of dependencies to add missing requirements. (:pull:`314`).
* The `cookiecutter` template has been updated to the latest commit via `cruft`. (:pull:`319`):
    * `actions-versions-updater.yml` has been replaced with `Dependabot <https://docs.github.com/en/code-security/dependabot/working-with-dependabot>`_ (it's just better).
    * The OpenSSF `scorecard.yml` workflow has been added to the GitHub workflows to evaluate package security.
    * Code formatting tools (`black`, `blackdoc`, `isort`) are now hard-pinned. These need to be kept in sync with changes from `pre-commit`. (Dependabot should perform this task automatically.)
    * The versioning system has been updated to follow the Semantic Versioning 2.0.0 standard.
* Fixed an issue with `pytest -m "not requires_netcdf"` not working as expected. (:pull:`345`).

.. _changes_0.8.0:

v0.8.0 (2024-01-16)
-------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Pascal Bourgault (:user:`aulemahal`), Juliette Lavoie (:user:`juliettelavoie`), Sarah-Claude Bourdeau-Goulet (:user:`sarahclaude`), Trevor James Smith (:user:`Zeitsperre`), Marco Braun (:user:`vindelico`).

Announcements
^^^^^^^^^^^^^
* `xscen` now adheres to PEPs 517/518/621 using the `setuptools` and `setuptools-scm` backend for building and packaging. (:pull:`292`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New function ``xscen.indicators.select_inds_for_avail_vars`` to filter the indicators that can be calculated with the variables available in a ``xarray.Dataset``. (:pull:`291`).
* Replaced aggregation function ``climatological_mean()`` with ``climatological_op()`` offering more types of operations to aggregate over climatological periods. (:pull:`290`)
* Added the ability to search for simulations that reach a given warming level. (:pull:`251`).
* ``xs.spatial_mean`` now accepts the ``region="global"`` keyword to perform a global average (:issue:`94`, :pull:`260`).
* ``xs.spatial_mean`` with ``method='xESMF'`` will also automatically segmentize polygons (down to a 1° resolution) to ensure a correct average (:pull:`260`).
* Added documentation for `require_all_on` in `search_data_catalogs`. (:pull:`263`).
* ``xs.save_to_table`` and ``xs.io.to_table`` to transform datasets and arrays to DataFrames, but with support for multi-columns, multi-sheets and localized table of content generation.
* Better ``xs.extract.resample`` : support for weighted resampling operations when starting with frequencies coarser than daily and missing timesteps/values handling. (:issue:`80`, :issue:`93`, :pull:`265`).
* New argument ``attribute_weights`` to ``generate_weights`` to allow for custom weights. (:pull:`252`).
* ``xs.io.round_bits`` to round floating point variable up to a number of bits, allowing for a better compression. This can be combined with the saving step through argument ``"bitround"`` of ``save_to_netcdf`` and ``save_to_zarr``. (:pull:`266`).
* Added annual global tas timeseries for CMIP6's models CMCC-ESM2 (ssp245, ssp370, ssp585), EC-Earth3-CC (ssp245, ssp585), KACE-1-0-G (ssp245, ssp370, ssp585) and TaiESM1 (ssp245, ssp370). Moved global tas database to a netCDF file. (:issue:`268`, :pull:`270`).
* Implemented support for multiple levels and models in ``xs.subset_warming_level``. Better support for `DataArray` and `DataFrame` in ``xs.get_warming_level``. (:pull:`270`).
* Added the ability to directly provide an ensemble dataset to ``xs.ensemble_stats``. (:pull:`299`).
* Added support in ``xs.ensemble_stats`` for the new robustness-related functions available in `xclim`. (:pull:`299`).
* New function ``xs.ensembles.get_partition_input`` (:pull:`289`).

Breaking changes
^^^^^^^^^^^^^^^^
* ``climatological_mean()`` has been replaced with ``climatological_op()`` and will be abandoned in a future version. (:pull:`290`)
* ``experiment_weights`` argument in ``generate_weights`` was renamed to ``balance_experiments``. (:pull:`252`).
* New argument ``attribute_weights`` to ``generate_weights`` to allow for custom weights. (:pull:`252`).
* For a sequence of models, the output of ``xs.get_warming_level`` is now a list. Revert to a dictionary with ``output='selected'`` (:pull:`270`).
* The global average temperature database is now a netCDF, custom databases must follow the same format (:pull:`270`).

Bug fixes
^^^^^^^^^
* Fixed a bug in ``xs.search_data_catalogs`` when searching for fixed fields and specific experiments/members. (:pull:`251`).
* Fixed a bug in the documentation build configuration that prevented stable/latest and tagged documentation builds from resolving on ReadTheDocs. (:pull:`256`).
* Fixed ``get_warming_level`` to avoid incomplete matches. (:pull:`269`).
* `search_data_catalogs` now eliminates anything that matches any entry in `exclusions`. (:issue:`275`, :pull:`280`).
* Fixed a bug in ``xs.scripting.save_and_update`` where ``build_path_kwargs`` was ignored when trying to guess the file format. (:pull:`282`).
* Add a warning to ``xs.extract._dispatch_historical_to_future``. (:issue:`286`, :pull:`287`).
* Modify use_cftime for the calendar conversion in ``to_dataset``. (:issue:`303`, :pull:`289`).

Internal changes
^^^^^^^^^^^^^^^^
* Continued work on adding tests. (:pull:`251`).
* Fixed `pre-commit`'s `pretty-format-json` hook so that it ignores notebooks. (:pull:`254`).
* Fixed the labeler so docs/CI isn't automatically added for contributions by new collaborators. (:pull:`254`).
* Made it so that `tests` are no longer treated as an installable package. (:pull:`248`).
* Renamed the pytest marker from ``requires_docs`` to ``requires_netcdf``. (:pull:`248`).
* Included the documentation in the source distribution, while excluding the NetCDF files. (:pull:`248`).
* Reduced the size of the files in ``/docs/notebooks/samples`` and changed the notebooks and tests accordingly. (:issue:`247`, :pull:`248`).
* Added a new `xscen.testing` module with the `datablock_3d` function previously located in ``/tests/conftest.py``. (:pull:`248`).
* New function `xscen.testing.fake_data` to generate fake data for testing. (:pull:`248`).
* xESMF 0.8 Regridder and SpatialAverager argument ``out_chunks`` is now accepted by ``xs.regrid_dataset``  and ``xs.spatial_mean``. (:pull:`260`).
* Testing, Packaging, and CI adjustments. (:pull:`274`):
    * `xscen` builds now install in a `tox` environment with `conda`-provided `ESMF` in GitHub Workflows.
    * `tox` now offers a method for installing esmpy from a tag/branch (via ESMF_VERSION environment variable).
    * `$ make translate` is now called on ReadTheDocs and within `tox`.
    * Linters are now called by order of most common failures first, to speed up the CI.
    * `Manifest.in` is much more specific about what is installed.
    * Re-adds a dev recipe to the `setup.py`.
* Multiple improvements to the docstrings and type annotations. (:pull:`282`).
* `pip check` in conda builds in GitHub workflows have been temporarily set to always pass. (:pull:`288`).
* The `cookiecutter` template has been updated to the latest commit via `cruft`. (:pull:`292`):
    * `setup.py` has been mostly hollowed-out, save for the `babel`-related translation function.
    * `pyproject.toml` has been added, with most package configurations migrated into it.
    * `HISTORY.rst` has been renamed to `CHANGES.rst`.
    * `actions-version-updater.yml` has been added to automate the versioning of the package.
    * `pre-commit` hooks have been updated to the latest versions; `check-toml` and `toml-sort` have been added to cleanup the `pyproject.toml` file, and `check-json-schema` has been added to ensure GitHub and ReadTheDocs workflow files are valid.
    * `ruff` has been added to the linting tools to replace most `flake8` and `pydocstyle` verifications.
    * `tox` builds are more pure Python environment/PyPI-friendly.
    * `xscen` now uses `Trusted Publishing` for TestPyPI and PyPI uploads.
* Linting checks now examine the testing folder, function complexity, and alphabetical order of `__all__` lists. (:pull:`292`).
* ``publish_release_notes`` now uses better logic for finding and reformatting the `CHANGES.rst` file. (:pull:`292`).
* ``bump2version`` version-bumping utility was replaced by ``bump-my-version``. (:pull:`292`).
* Documentation build checks no longer fail due to broken external links; Notebooks are now nested and numbered. (:pull:`304`).

.. _changes_0.7.1:

v0.7.1 (2023-08-23)
-------------------
* Update dependencies by removing ``pygeos``, pinning ``shapely>=2`` and ``intake-esm>=2023.07.07`` as well as other small fixes to the environment files. (:pull:`243`).
* Fix ``xs.aggregate.spatial_mean`` with method ``cos-lat`` when the data is on a rectilinear grid. (:pull:`243`).

Internal changes
^^^^^^^^^^^^^^^^
* Added a workflow that removes obsolete GitHub Workflow caches from merged pull requests. (:pull:`250`).
* Added a workflow to perform automated labeling of pull requests, dependent on the files changed. (:pull:`250`).

.. _changes_0.7.0:

v0.7.0 (2023-08-22)
-------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Pascal Bourgault (:user:`aulemahal`), Trevor James Smith (:user:`Zeitsperre`), Juliette Lavoie (:user:`juliettelavoie`), Marco Braun (:user:`vindelico`).

Announcements
^^^^^^^^^^^^^
* Dropped support for Python 3.8, added support for 3.11. (:pull:`199`, :pull:`222`).
* `xscen` is now available on `conda-forge <https://anaconda.org/conda-forge/xscen>`_, and can be installed with ``conda install -c conda-forge xscen``. (:pull:`241`)

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* `xscen` now tracks code coverage using `coveralls <https://coveralls.io/>`_. (:pull:`187`).
* New function `get_warming_level` to search within the IPCC CMIP global temperatures CSV without requiring data. (:issue:`208`, :pull:`210`).
* File re-structuration from catalogs with ``xscen.catutils.build_path``. (:pull:`205`, :pull:`237`).
* New scripting functions `save_and_update` and `move_and_delete`. (:pull:`214`).
* Spatial dimensions can be generalized as X/Y when rechunking and will be mapped to rlon/rlat or lon/lat accordingly. (:pull:`221`).
* New argument `var_as_string` for `get_cat_attrs` to return variable names as strings. (:pull:`233`).
* New argument `copy` for `move_and_delete`. (:pull:`233`).
* New argument `restrict_year` for `compute_indicators`. (:pull:`233`).
* Add more comments in the template. (:pull:`233`, :issue:`232`).
* ``generate_weights`` now allows to split weights between experiments, and make them vary along the time/horizon axis. (:issue:`108`, :pull:`231`).
* New independence_level, `institution`, added to ``generate_weights``. (:pull:`231`).
* Updated ``produce_horizon`` so it can accept multiple periods or warming levels. (:pull:`231`, :pull:`240`).
* Add more comments in the template. (:pull:`233`, :pull:`235`, :issue:`232`).
* New function ``diagnostics.health_checks`` that can perform multiple checkups on a dataset. (:pull:`238`).

Breaking changes
^^^^^^^^^^^^^^^^
* Columns ``date_start`` and ``date_end`` now use a ``datetime64[ms]`` dtype. (:pull:`222`).
* The default output of ``date_parser`` is now ``pd.Timestamp`` (``output_dtype='datetime'``). (:pull:`222`).
* ``date_parser(date, end_of_period=True)`` has time "23:59:59", instead of "23:00". (:pull:`222`, :pull:`237`).
* ``driving_institution`` was removed from the "default" xscen columns. (:pull:`222`).
* Folder parsing utilities (``parse_directory``) moved to ``xscen.catutils``. Signature changed : ``globpattern`` removed, ``dirglob`` added, new ``patterns`` specifications. See doc for all changes. (:pull:`205`).
* ``compute_indicators`` now returns all outputs produced by indicators with multiple outputs (such as `rain_season`). (:pull:`228`).
* In ``generate_weights``, independence_level `all` was renamed `model`. (:pull:`231`).
* In response to a bugfix, results for ``generate_weights(independence_level='GCM')`` are significantly altered. (:issue:`230`, :pull:`231`).
* Legacy support for `stats_kwargs` in ``ensemble_stats`` was dropped. (:pull:`231`).
* `period` in ``produce_horizon`` has been deprecated and replaced with `periods`. (:pull:`231`).
* Some automated `to_level` were updated to reflect more recent changes. (:pull:`231`).
* Removed ``diagnostics.fix_unphysical_values``. (:pull:`238`).

Bug fixes
^^^^^^^^^
* Fix bug in ``unstack_dates`` with seasonal climatological mean. (:issue:`202`, :pull:`202`).
* Added NotImplemented errors when trying to call `climatological_mean` and `compute_deltas` with daily data. (:pull:`187`).
* Minor documentation fixes. (:issue:`223`, :pull:`225`).
* Fixed a bug in ``unstack_dates`` where it failed for anything other than seasons. (:pull:`228`).
* ``cleanup`` with `common_attrs_only` now works even when no `cat` attribute is present in the datasets. (:pull:`231`).

Internal changes
^^^^^^^^^^^^^^^^
* Removed the pin on xarray's version. (:issue:`175`, :pull:`199`).
* Folder parsing utilities now in pure python, platform independent. New dependency ``parse``. (:pull:`205`).
* Updated ReadTheDocs configuration to prevent ``--eager`` installation of xscen (:pull:`209`).
* Implemented a template to be used for unit tests. (:pull:`187`).
* Updated GitHub Actions to remove deprecation warnings. (:pull:`187`).
* Updated the cookiecutter used to generate boilerplate documentation and code via `cruft`. (:pull:`212`).
* A few changes to `subset_warming_level` so it doesn't need `driving_institution`. (:pull:`215`).
* Added more tests. (:pull:`228`).
* In ``compute_indicators``, the logic to manage indicators returning multiple outputs was simplified. (:pull:`228`).

.. _changes_0.6.0:

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
* Add function ``regrid.create_bounds_rotated_pole`` and automatic use in ``regrid_dataset`` and ``spatial_mean``. This is temporary, while we wait for a functioning method in ``cf_xarray``. (:pull:`174`, :issue:`96`).
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

.. _changes_0.5.0:

v0.5.0 (2023-02-28)
-------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Juliette Lavoie (:user:`juliettelavoie`), Trevor James Smith (:user:`Zeitsperre`), Sarah Gammon (:user:`SarahG-579462`) and Pascal Bourgault (:user:`aulemahal`).

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

.. _changes_0.4.0:

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
    * Do not fail for any grid mapping problem, including if a grid_mapping attribute mentions a variable that doesn't exist.
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

.. _changes_0.3.0:

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

.. _changes_0.2.0:

v0.2.0 (2021-01-26)
-------------------
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
