============
Installation
============

From sources
------------

Because of some packages being absent from pypi (`xESMF`), we highly recommand installing
xscen in a conda environment. An environment file is available on the `Github repo`_.

.. Warning::
    The current version of the code requires `the master version of intake-esm <https://github.com/intake/intake-esm>`_.
    If you are not using the provided environment file, it can be installed from their Github repository following similar instructions.

Because the repo is private, you must clone it locally:

.. code-block:: console

    $ git clone git@github.com:Ouranosinc/xscen.git

Then you can create the environment and install the package:

.. code-block:: console

    $ cd xscen
    $ conda env create -f environment.yml
    $ pip install .


.. _Github repo: https://github.com/Ouranosinc/xscen
