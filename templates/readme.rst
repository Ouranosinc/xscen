Workflow templates
==================

`This folder <https://github.com/Ouranosinc/xscen/tree/main/templates>`_ contains templates of xscen workflows to provide additional "real-world" examples besides the notebooks and API docs. Most of them are not usable as-is, but usually only the configuration (some yaml files) needs to be edited.

Most of the templates are heavily commented in their python code as well as their configuration files, but a small summary of what each does is included here.

.. warning::

	The link above brings you to the development version of the templates. You might want to access a version specific to your installed xscen. Click the dropdown menu in the upper left corner that says "main" and navigate to "tags" and the specific version of interest.


1 - Basic workflow with config
------------------------------
The archetypal xscen workflow that does every steps of a normal climate scenarisation project : extract, regrid, biasadjust, cleanup, rechunk, diagnostics, indicators, climatology, delta, ensembles. It is controlled from the ``config1.yml`` file. For each step, it will iterate over each member of the ensemble, thus creating many intermediate files before the final products.

2 - Compute indicators
----------------------
A basic, single-step workflow to compute a list (module) of xclim indicators. Also controlled from its ``config2.yml`` file, but its path needs to be passed to the script through the command line.
