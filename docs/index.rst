
Welcome to xscen's documentation!
=================================

xscen: A climate change scenario-building analysis framework, built with Intake-esm catalogs and xarray-based packages such as xclim and xESMF.

Features
========
* Supports workflows with YAML configuration files for better transparency, reproducibility, and long-term backups.
* Intake_esm-based catalog to find and manage climate data.
* Climate dataset extraction, subsetting, and temporal aggregation.
* Calculate missing variables through Intake-esm's DerivedVariableRegistry.
* Regridding with xESMF.
* Bias adjustment with xclim.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    readme
    installation
    usage
    notebooks/catalog
    notebooks/getting_started
    notebooks/config_usage
    columns
    api
    contributing
    authors
    history
