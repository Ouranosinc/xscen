
Welcome to xscen's documentation!
=================================

xscen: A climate change scenario-building analysis framework, built with Intake-esm catalogs and xarray-based packages such as xclim and xESMF.

Need help?
==========
* Ouranos employees can ask questions on the Ouranos private StackOverflow where you can tag subjects and people. (https://stackoverflow.com/c/ouranos/questions ).
* Potential bugs can be reported as an issue on github (https://github.com/Ouranosinc/xscen/issues ).
* To be aware of changes in xscen, you can "watch" the github repo. You can customize the watch function to notify you of new releases. (https://github.com/Ouranosinc/xscen )

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
    notebooks/diagnostics
    columns
    api
    contributing
    authors
    history

.. toctree::
   :maxdepth: 1
   :caption: Public Modules

   xscen

.. toctree::
   :maxdepth: 2
   :caption: Package Structure

   modules
