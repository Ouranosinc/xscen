============
xscen |logo|
============

+----------------------------+-----------------------------------------------------+
| Versions                   | |pypi| |conda| |versions|                           |
+----------------------------+-----------------------------------------------------+
| Documentation and Support  | |docs|                                              |
+----------------------------+-----------------------------------------------------+
| Open Source                | |license| |ossf|                                    |
+----------------------------+-----------------------------------------------------+
| Coding Standards           | |black| |ruff| |pre-commit|                         |
+----------------------------+-----------------------------------------------------+
| Development Status         | |status| |build| |coveralls|                        |
+----------------------------+-----------------------------------------------------+

A climate change scenario-building analysis framework, built with Intake-esm catalogs and xarray-based packages such as xclim and xESMF.

For documentation concerning `xscen`, see: https://xscen.readthedocs.io/en/latest/

Features
--------
* Supports workflows with YAML configuration files for better transparency, reproducibility, and long-term backups.
* `Intake-esm`_-based catalog to find and manage climate data.
* Climate dataset extraction, subsetting, and temporal aggregation.
* Calculate missing variables through `intake-esm`'s `DerivedVariableRegistry`_.
* Regridding powered by `xESMF`_.
* Bias adjustment tools provided by `xclim`_.

Installation
------------

Please refer to the `installation docs`_.

Acknowledgments
---------------
This package was created with Cookiecutter_ and the `Ouranosinc/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/cookiecutter/cookiecutter
.. _Ouranosinc/cookiecutter-pypackage: https://github.com/Ouranosinc/cookiecutter-pypackage
.. _installation docs: https://xscen.readthedocs.io/en/latest/installation.html
.. _Intake-esm: https://intake-esm.readthedocs.io/
.. _DerivedVariableRegistry: https://intake-esm.readthedocs.io/en/latest/how-to/define-and-use-derived-variable-registry.html
.. _xclim: https://xclim.readthedocs.io/
.. _xESMF: https://xesmf.readthedocs.io/

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black
        :alt: Python Black

.. |build| image:: https://github.com/Ouranosinc/xscen/actions/workflows/main.yml/badge.svg
        :target: https://github.com/Ouranosinc/xscen/actions/workflows/main.yml
        :alt: Build Status

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/xscen.svg
        :target: https://anaconda.org/conda-forge/xscen
        :alt: Conda Build

.. |coveralls| image:: https://coveralls.io/repos/github/Ouranosinc/xscen/badge.svg
        :target: https://coveralls.io/github/Ouranosinc/xscen
        :alt: Coveralls

.. |docs| image:: https://readthedocs.org/projects/xscen/badge/?version=latest
        :target: https://xscen.readthedocs.io/en/latest
        :alt: Documentation Status

.. |license| image:: https://img.shields.io/pypi/l/figanos
        :target: https://github.com/Ouranosinc/figanos/blob/main/LICENSE
        :alt: License

.. |logo| image:: https://raw.githubusercontent.com/Ouranosinc/xscen/main/docs/_static/_images/xscen-logo-small.png
        :target: https://github.com/Ouranosinc/xscen
        :alt: xscen Logo

.. |ossf| image:: https://api.securityscorecards.dev/projects/github.com/Ouranosinc/xscen/badge
        :target: https://securityscorecards.dev/viewer/?uri=github.com/Ouranosinc/xscen
        :alt: OpenSSF Scorecard

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/Ouranosinc/xscen/main.svg
        :target: https://results.pre-commit.ci/latest/github/Ouranosinc/xscen/main
        :alt: pre-commit.ci status

.. |pypi| image:: https://img.shields.io/pypi/v/xscen.svg
        :target: https://pypi.python.org/pypi/xscen
        :alt: Python Package Index

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
        :target: https://github.com/astral-sh/ruff
        :alt: Ruff

.. |status| image:: https://www.repostatus.org/badges/latest/active.svg
        :target: https://www.repostatus.org/#active
        :alt: Project Status: Active  The project has reached a stable, usable state and is being actively developed.

.. |versions| image:: https://img.shields.io/pypi/pyversions/xscen.svg
        :target: https://pypi.python.org/pypi/xscen
        :alt: Supported Python Versions
