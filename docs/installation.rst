============
Installation
============

Official Sources
----------------

Because of some packages being absent from PyPI (such as `xESMF`), we strongly recommend installing `xscen` in an Anaconda Python environment.

`xscen` can be installed directly from conda-forge:

.. code-block:: console

    $ conda install -c conda-forge xscen

.. note::

    If you are unable to install the package due to missing dependencies, ensure that `conda-forge` is listed as a source in your `conda` configuration: `$ conda config --add channels conda-forge`!

If for some reason you wish to install the `PyPI` version of `xscen` into an existing Anaconda environment (*not recommended*), this can be performed with:

.. code-block:: console

    $ python -m pip install xscen

Development Installation (Anaconda + pip)
-----------------------------------------

For development purposes, we provide the means for generating a conda environment with the latest dependencies in an `environment.yml` file at the top-level of the `Github repo`_.

In order to get started, first clone the repo locally:

.. code-block:: console

    $ git clone git@github.com:Ouranosinc/xscen.git

Then you can create the environment and install the package:

.. code-block:: console

    $ cd xscen
    $ conda env create -f environment.yml

Finally, perform an `--editable` install of xscen and compile the translation catalogs:

.. code-block:: console

    $ python -m pip install -e .
    $ make translate

.. _Github repo: https://github.com/Ouranosinc/xscen
