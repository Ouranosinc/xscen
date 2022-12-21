============
Installation
============

From sources
------------

Because of some packages being absent from pypi (`xESMF`), we highly recommend installing
xscen in a conda environment. An environment file is available on the `Github repo`_.

It's easier to do all this by first cloning the repo locally:

.. code-block:: console

    $ git clone git@github.com:Ouranosinc/xscen.git

Then you can create the environment and install the package:

.. code-block:: console

    $ cd xscen
    $ conda env create -f environment.yml
    $ pip install .


.. _Github repo: https://github.com/Ouranosinc/xscen
