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

It's easier to do all this by first cloning the repo locally:

.. code-block:: console

    $ git clone --recurse-submodules git@github.com:Ouranosinc/xscen.git

.. Warning::
    If you are NOT executing this from within the Ouranos VPN, git will complain about not being able
    to clone the `docs/notebooks` submodules. It's usually not a problem, unless you wanted to read or
    edit them. They are in fact stored on a private repository on Ouranos' internal Gitlab server because
    they make use of private and internal data.

Then you can create the environment and install the package:

.. code-block:: console

    $ cd xscen
    $ conda env create -f environment.yml
    $ pip install .


.. _Github repo: https://github.com/Ouranosinc/xscen
