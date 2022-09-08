============
xscen |logo|
============

|build| |docs| |black| |pre-commit|

A climate change scenario-building analysis framework, built with Intake-esm catalogs and xarray-based packages such as xclim and xESMF.

For documentation concerning the public API of `xscen`, see: https://xscen.readthedocs.io/en/latest/

For the full documentation with notebooks/examples, please consult: https://scenario.gitlab.ouranos.ca/xscen/ (available only via the Ouranos intranet)

Features
--------
* Supports workflows with YAML configuration files for better transparency, reproducibility, and long-term backups.
* Intake_esm-based catalog to find and manage climate data.
* Climate dataset extraction, subsetting, and temporal aggregation.
* Calculate missing variables through Intake-esm's DerivedVariableRegistry.
* Regridding with xESMF.
* Bias adjustment with xclim.

Acknowledgments
---------------
This package was created with Cookiecutter_ and the `Ouranosinc/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyfeldroy/cookiecutter-pypackage
.. _`Ouranosinc/cookiecutter-pypackage`: https://github.com/Ouranosinc/cookiecutter-pypackage

.. |logo| image:: https://raw.githubusercontent.com/Ouranosinc/xclim/master/docs/_static/_images/xscen-logo-small.png
        :target: https://github.com/Ouranosinc/xscen

.. |build| image:: https://github.com/Ouranosinc/xscen/actions/workflows/main.yml/badge.svg
        :target: https://github.com/Ouranosinc/xscen/actions/workflows/main.yml
        :alt: Build Status

.. |docs| image:: https://readthedocs.org/projects/xscen/badge/?version=latest
        :target: https://xscen.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black
        :alt: Python Black

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/Ouranosinc/xscen/main.svg
        :target: https://results.pre-commit.ci/latest/github/Ouranosinc/xscen/main
        :alt: pre-commit.ci status
