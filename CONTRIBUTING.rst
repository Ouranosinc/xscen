.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/Ouranosinc/xscen/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

xscen could always use more documentation, whether as part of the official xscen docs, in docstrings, or even on the web in blog posts, articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/Ouranosinc/xscen/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up ``xscen`` for local development.

#. Clone the repo locally::

    $ git clone git@github.com:Ouranosinc/xscen.git

#. Install your local copy into a development environment. Using ``mamba``, you can create a new development environment with::

    $ mamba env create -f environment-dev.yml
    $ conda activate xscen
    $ python -m pip install --editable ".[dev]"

#. As xscen was installed in editable mode, we also need to compile the translation catalogs manually:

    $ make translate

#. To ensure a consistent style, please install the pre-commit hooks to your repo::

    $ pre-commit install

   Special style and formatting checks will be run when you commit your changes. You
   can always run the hooks on their own with:

    $ pre-commit run -a

#. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

#. When you're done making changes, check that your changes pass ``black``, ``blackdoc``, ``flake8``, ``isort``, ``ruff``, and the tests, including testing other Python versions with tox::

     $ black --check xscen tests
     $ isort --check xscen tests
     $ ruff xscen tests
     $ flake8 xscen tests
     $ blackdoc --check xscen docs
     $ python -m pytest
     $ tox

   To get ``black``, ``blackdoc``, ``flake8``, ``isort``, ``ruff``, and tox, just pip install them into your virtualenv.

   Alternatively, you can run the tests using `make`::

    $ make lint
    $ make test

   Running `make lint` and `make test` demands that your runtime/dev environment have all necessary development dependencies installed.

   .. warning::

        Due to some dependencies only being available via Anaconda/conda-forge or built from source, `tox`-based testing will only work if `ESMF`_ is available in your system path. This also requires that the `ESMF_VERSION` environment variable (matching the version of ESMF installed) be accessible within your shell as well (e.g.: `$ export ESMF_VERSION=8.5.0`).

#. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

#. If you are editing the docs, compile and open them with::

    $ make docs
    # or to simply generate the html
    $ cd docs/
    $ make html

   .. note::

       When building the documentation, the default behaviour is to evaluate notebooks ('nbsphinx_execute = "always"'), rather than simply parse the content ('nbsphinx_execute = "never"'). Due to their complexity, this can sometimes be a very computationally demanding task and should only be performed when necessary (i.e.: when the notebooks have been modified).

       In order to speed up documentation builds, setting a value for the environment variable "SKIP_NOTEBOOKS" (e.g. "$ export SKIP_NOTEBOOKS=1") will prevent the notebooks from being evaluated on all subsequent "$ tox -e docs" or "$ make docs" invocations.

#. Submit a pull request through the GitHub website.

.. _translating-xscen:

Translating xscen
~~~~~~~~~~~~~~~~~

If your additions to ``xscen` play with plain text attributes like "long_name" or "description", you should also provide
French translations for those fields. To manage translations, xscen uses python's ``gettext`` with the help of ``babel``.

To update an attribute while enabling translation, use :py:func:`utils.add_attr` instead of a normal set-item. For example:

    .. code-block:: python

        ds.attrs["description"] = "The English description"

becomes:

    .. code-block:: python

        from xscen.utils import add_attr


        def _(s):
            return s


        add_attr(ds, "description", _("English description of {a}"), a="var")

See also :py:func:`update_attr` for the special case where an attribute is updated using its previous version.

Once the code is implemented and translatable strings are marked as such, we need to extract them and catalog them in the French translation map. From the root directory of xscen, run::

    $ make findfrench

Then go edit ``xscen/xscen/data/fr/LC_MESSAGES/xscen.po`` with the correct French translations. Finally, running::

    $ make translate

This will compile the edited catalogs, allowing python to detect and use them.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

#. The pull request should include tests.

#. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in ``README.rst``.

#. The pull request should not break the templates.

#. The pull request should work for Python 3.8, 3.9, 3.10, and 3.11. Check that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

$ pytest tests.test_xscen

Versioning/Tagging
------------------

A reminder for the maintainers on how to deploy. This section is only relevant for maintainers when they are producing a new point release for the package.

#. Create a new branch from `main` (e.g. `release-0.2.0`).
#. Update the `CHANGES.rst` file to change the `Unreleased` section to the current date.
#. Create a pull request from your branch to `main`.
#. Once the pull request is merged, create a new release on GitHub. On the main branch, run:

    .. code-block:: shell

        $ bump-my-version bump minor # In most cases, we will be releasing a minor version
        $ git push
        $ git push --tags

    This will trigger the CI to build the package and upload it to TestPyPI. In order to upload to PyPI, this can be done by publishing a new version on GitHub. This will then trigger the workflow to build and upload the package to PyPI.

#. Once the release is published, it will go into a `staging` mode on Github Actions. Once the tests pass, admins can approve the release (an e-mail will be sent) and it will be published on PyPI.

.. note::

    The ``bump-version.yml`` GitHub workflow will automatically bump the patch version when pull requests are pushed to the ``main`` branch on GitHub. It is not necessary to manually bump the version in your branch when merging (non-release) pull requests.

.. warning::

    It is important to be aware that any changes to files found within the ``xscen`` folder (with the exception of ``xscen/__init__.py``) will trigger the ``bump-version.yml`` workflow. Be careful not to commit changes to files in this folder when preparing a new release.

Packaging
---------

When a new version has been minted (features have been successfully integrated test coverage and stability is adequate), maintainers should update the ``pip``-installable package (wheel and source release) on PyPI as well as the binary on conda-forge.

The simple approach
~~~~~~~~~~~~~~~~~~~

The simplest approach to packaging for general support (pip wheels) requires the following packages installed:
 * build
 * setuptools
 * twine
 * wheel

From the command line on your Linux distribution, simply run the following from the clone's main dev branch::

    # To build the packages (sources and wheel)
    $ python -m build --sdist --wheel

    # To upload to PyPI
    $ twine upload dist/*


.. _`ESMF`: http://earthsystemmodeling.org/download/
