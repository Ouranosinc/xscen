
Welcome to xscen's documentation!
=================================

xscen: A climate change scenario-building analysis framework, built with Intake-esm catalogs and xarray-based packages such as xclim and xESMF.

The documentation hosted in this repository is only partial. For the full documentation, please consult: https://scenario.gitlab.ouranos.ca/scenarios_main/

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
    columns
    api
    contributing
    authors
    history
