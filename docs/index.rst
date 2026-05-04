
Welcome to xscen's documentation!
=================================

**xscen**: A climate change scenario-building analysis framework, built with Intake-esm catalogs and xarray-based packages such as xclim and xESMF.

Need help?
^^^^^^^^^^
* Ouranos employees can ask questions on the Ouranos private StackOverflow where you can tag subjects and people. (https://stackoverflow.com/c/ouranos/questions)
* Potential bugs in xscen can be reported as an issue here: https://github.com/Ouranosinc/xscen/issues
* Problems with data on Ouranos' servers can be reported as an issue here: https://github.com/Ouranosinc/miranda/issues
* To be aware of changes in xscen, you can "watch" the GitHub repository. You can customize the watch function to notify you of new releases. (https://github.com/Ouranosinc/xscen)

Features
^^^^^^^^
* Supports workflows with YAML configuration files for better transparency, reproducibility, and long-term backups
* Intake-esm-based catalog to find and manage climate data
* Climate dataset extraction, subsetting, and temporal aggregation
* Calculate missing variables through Intake-esm's DerivedVariableRegistry
* Regridding with xESMF
* Bias adjustment with xsdba

Acknowledgments
^^^^^^^^^^^^^^^
`xscen` development is funded through `Ouranos <https://www.ouranos.ca/>`_, a not-for-profit collaborative innovation organization fostering resilient adaptation to climate change, based in Québec, Canada.

.. image:: https://raw.githubusercontent.com/Ouranosinc/.github/refs/heads/main/images/logo-ouranos-horizontal-couleur.svg
    :target: https://www.ouranos.ca/
    :align: center

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    readme
    installation
    goodtoknow
    notebooks/index
    columns
    templates
    api
    contributing
    releasing
    authors
    changelog
    security

.. toctree::
    :maxdepth: 2
    :caption: Package Structure

    apidoc/modules

.. toctree::
    :caption: GitHub Repository

    Ouranosinc/xscen <https://github.com/Ouranosinc/xscen>
