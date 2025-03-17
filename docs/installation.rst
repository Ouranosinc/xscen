============
Installation
============

We strongly recommend installing xscen in an Anaconda Python environment.
Furthermore, due to the complexity of some packages, the default dependency solver can take a long time to resolve the environment.
If `mamba` is not already your default solver, consider running the following commands in order to speed up the process:

    .. code-block:: console

        conda install -n base conda-libmamba-solver
        conda config --set solver libmamba

If you don't have `pip`_ installed, this `Python installation guide`_ can guide you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Official Sources
----------------

Because of some packages being absent from PyPI (such as `xESMF`), we strongly recommend installing `xscen` in an Anaconda Python environment.

`xscen` can be installed directly from conda-forge:

.. code-block:: console

    conda install -c conda-forge xscen

This is the preferred method to install xscen, as it will always install the most recent stable release.

.. note::

    If you are unable to install the package due to missing dependencies, ensure that `conda-forge` is listed as a source in your `conda` configuration: `conda config --add channels conda-forge`!

If for some reason you wish to install the `PyPI` version of `xscen` into an existing Anaconda environment (*not recommended*), this can be performed with:

.. code-block:: console

    python -m pip install xscen

This will install the latest stable release from `PyPI`, but will not include `xESMF` and other packages that are not available on `PyPI`. Measures have been taken to ensure that `xscen` will not break if `xESMF` fails to import, but some functionalities will be disabled.

Development Installation (Anaconda + pip)
-----------------------------------------

For development purposes, we provide the means for generating a conda environment with the latest dependencies in an `environment.yml` file at the top-level of the `Github repo <https://github.com/Ouranosinc/xscen>`_.

The sources for xscen can be downloaded from the `Github repo`_.

#. Download the source code from the `Github repo`_ using one of the following methods:

    * Clone the public repository:

        .. code-block:: console

            git clone git@github.com:Ouranosinc/xscen.git

    * Download the `tarball <https://github.com/Ouranosinc/xscen/tarball/main>`_:

        .. code-block:: console

            curl -OJL https://github.com/Ouranosinc/xscen/tarball/main

#. Once you have a copy of the source, you can install it with:

    .. code-block:: console

        conda env create -f environment-dev.yml
        conda activate xscen-dev
        make dev

    If you are on Windows, replace the ``make dev`` command with the following:

    .. code-block:: console

        python -m pip install -e .[dev]

    Even if you do not intend to contribute to `xscen`, we favor using `environment-dev.yml` over `environment.yml` because it includes additional packages that are used to run all the examples provided in the documentation.
    If for some reason you wish to install the `PyPI` version of `xscen` into an existing Anaconda environment (*not recommended if requirements are not met*), only run the last command above.

#. When new changes are made to the `Github repo`_, if using a clone, you can update your local copy using the following commands from the root of the repository:

    .. code-block:: console

        git fetch
        git checkout main
        git pull origin main
        conda env update -n xscen-dev -f environment-dev.yml
        conda activate xscen-dev
        make dev

    These commands should work most of the time, but if big changes are made to the repository, you might need to remove the environment and create it again.

#. Finally, in order to compile the translation catalogs, run the following command from the root of the repository:

    .. code-block:: console

        python -m pip install -e .
        make translate

.. _Github repo: https://github.com/Ouranosinc/xscen
