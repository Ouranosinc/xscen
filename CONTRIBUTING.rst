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
* Remember that this is a volunteer-driven project, and that contributions are welcome. :)

Get Started!
------------

.. note::

    If you are new to using `GitHub <https://github.com/>`_ and ``git``, please read `this guide <https://guides.github.com/activities/hello-world/>`_ first.

.. warning::

    Anaconda Python users: Due to the complexity of some packages, the default dependency solver can take a long time to resolve the environment. Consider running the following commands in order to speed up the process:

    .. code-block:: console

        conda install -n base conda-libmamba-solver
        conda config --set solver libmamba

    For more information, please see the following link: https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community

    Alternatively, you can use the `mamba <https://mamba.readthedocs.io/en/latest/index.html>`_ package manager, which is a drop-in replacement for ``conda``. If you are already using `mamba`, replace the following commands with ``mamba`` instead of ``conda``.

Ready to contribute? Here's how to set up ``xscen`` for local development.

#. First, clone the ``xscen`` repo locally.

    * If you are not a ``xscen`` collaborator, first fork the ``xscen`` repo on GitHub, then clone your fork locally.

        .. code-block:: console

            git clone git@github.com:your_name_here/xscen.git

    * If you are a ``xscen`` collaborator, clone the ``xscen`` repo directly.

        .. code-block:: console

            git clone git@github.com:Ouranosinc/xscen.git

#. Install your local copy into a development environment. You can create a new Anaconda development environment with:

    .. code-block:: console

        conda env create -f environment-dev.yml
        conda activate xscen-dev
        make dev

    If you are on Windows, replace the ``make dev`` command with the following:

    .. code-block:: console

        python -m pip install -e .[dev]
        pre-commit install

    This installs ``xscen`` in an "editable" state, meaning that changes to the code are immediately seen by the environment. To ensure a consistent coding style, `make dev` also installs the ``pre-commit`` hooks to your local clone.

    On commit, ``pre-commit`` will check that ``flake8``, and ``ruff`` checks are passing, perform automatic fixes if possible, and warn of violations that require intervention. If your commit fails the checks initially, simply fix the errors, re-add the files, and re-commit.

    You can also run the hooks manually with:

    .. code-block:: console

        pre-commit run -a

    If you want to skip the ``pre-commit`` hooks temporarily, you can pass the `--no-verify` flag to `$ git commit`.

#. As xscen was installed in editable mode, we also need to compile the translation catalogs manually:

    .. code-block:: console

        make translate

#. Create a branch for local development:

    .. code-block:: console

        git checkout -b name-of-your-bugfix-or-feature

    Now you can make your changes locally.

#. When you're done making changes, we **strongly** suggest running the tests in your environment or with the help of ``tox``:

    .. code-block:: console

        make lint
        python -m pytest
        # Or, to run multiple build tests
        python -m tox

    Alternatively, you can run the tests using `make`:

    .. code-block:: console

        make lint
        make test

    Running `make lint` and `make test` demands that your runtime/dev environment have all necessary development dependencies installed.

   .. warning::

        Due to some dependencies only being available via Anaconda/conda-forge or built from source, `tox`-based testing will only work if `ESMF <http://earthsystemmodeling.org/download/>`_ is available in your system path. This also requires that the `ESMF_VERSION` environment variable (matching the version of ESMF installed) be accessible within your shell as well (e.g.: `$ export ESMF_VERSION=8.5.0`).

#. Commit your changes and push your branch to GitHub:

    .. code-block:: console

        git add .
        git commit -m "Your detailed description of your changes."
        git push origin name-of-your-bugfix-or-feature

    If ``pre-commit`` hooks fail, try fixing the issues, re-staging the files to be committed, and re-committing your changes (or, if need be, you can skip them with `git commit --no-verify`).


#. Submit a `Pull Request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_ through the GitHub website.

#. If changes to your branch are made on GitHub, you can update your local branch with:

    .. code-block:: console

        git checkout name-of-your-bugfix-or-feature
        git fetch
        git pull origin name-of-your-bugfix-or-feature

    If you have merge conflicts, you might need to replace `git pull` with `git merge` and resolve the conflicts manually.
    Resolving conflicts from the command line can be tricky. If you are not comfortable with this, you can ignore the last command and instead use a GUI like PyCharm or Visual Studio Code to merge the remote changes and resolve the conflicts.

#. Before merging, your Pull Request will need to be based on the `main` branch of the  `xscen`` repository. If your branch is not up-to-date with the `main` branch, you can perform similar steps as above to update your branch:

    .. code-block:: console

        git checkout name-of-your-bugfix-or-feature
        git fetch
        git pull origin main

    See the previous step for more information on resolving conflicts.

#. When pushing your changes to your branch on GitHub, the documentation will automatically be tested to reflect the changes in your Pull Request. This build process can take several minutes at times. If you are actively making changes that affect the documentation and wish to save time, you can compile and test your changes beforehand locally with:

    .. code-block:: console

        # To generate the html and open it in your browser
        make docs
        # To only generate the html
        make autodoc
        make -C docs html
        # To simply test that the docs pass build checks
        python -m tox -e docs

   .. note::

       When building the documentation, the default behaviour is to evaluate notebooks ('nbsphinx_execute = "always"'), rather than simply parse the content ('nbsphinx_execute = "never"'). Due to their complexity, this can sometimes be a very computationally demanding task and should only be performed when necessary (i.e.: when the notebooks have been modified).

       In order to speed up documentation builds, setting a value for the environment variable "SKIP_NOTEBOOKS" (e.g. "$ export SKIP_NOTEBOOKS=1") will prevent the notebooks from being evaluated on all subsequent "$ tox -e docs" or "$ make docs" invocations.

#. Once your Pull Request has been accepted and merged to the `main` branch, several automated workflows will be triggered:

   - The ``bump-version.yml`` workflow will automatically bump the patch version when pull requests are pushed to the `main` branch on GitHub. **It is not recommended to manually bump the version in your branch when merging (non-release) pull requests (this will cause the version to be bumped twice).**
   - `ReadTheDocs` will automatically build the documentation and publish it to the `latest` branch of `xscen` documentation website.
   - If your branch is not a fork (ie: you are a maintainer), your branch will be automatically deleted.

You will have contributed to ``xscen``!

.. _translating-xscen:

Translating xscen
~~~~~~~~~~~~~~~~~

If your additions to ``xscen`` play with plain text attributes like "long_name" or "description", you should also provide
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

Then go edit ``xscen/xscen/data/fr/LC_MESSAGES/xscen.po`` with the correct French translations. Finally, run::

    $ make translate

This will compile the edited catalogs, allowing python to detect and use them.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

#. The pull request should include tests and should aim to provide `code coverage <https://en.wikipedia.org/wiki/Code_coverage>`_ for all new lines of code. You can use the ``--cov-report html --cov xscen`` flags during the call to ``pytest`` to generate an HTML report and analyse the current test coverage.

#. All functions should be documented with `docstrings` following the `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ format.

#. If the pull request adds functionality, either update the documentation or create a new notebook that demonstrates the feature. Library-defining features should also be listed in ``README.rst``.

#. The pull request should not break the templates.

#. The pull request should work for all currently supported Python versions. Check the `pyproject.toml` or `tox.ini` files for the list of supported versions. We aim to follow the support and drop schedule of Python versions as recommended by the NumPy NEP calendar: https://numpy.org/neps/nep-0029-deprecation_policy.html

Tips
----

To run a subset of tests:

.. code-block:: console

    python -m pytest tests/test_xscen.py

You can also directly call a specific test class or test function using:

.. code-block:: console

    python -m pytest tests/test_xscen.py::TestClassName::test_function_name

For more information on running tests, see the `pytest documentation <https://docs.pytest.org/en/latest/usage.html>`_.

To run specific code style checks:

.. code-block:: console

    python -m ruff check src/xscen tests
    python -m flake8 src/xscen tests
    python -m numpydoc lint src/xscen/**.py

To get ``ruff``, ``flake8`` (with the ``flake8-rst-docstrings`` plugin), and ``numpydoc``, simply install them with ``pip`` (or ``conda``) into your environment.

Versioning/Tagging
------------------

A reminder for the **maintainers** on how to deploy. This section is only relevant when producing a new point release for the package.

.. warning::

    It is important to be aware that any changes to files found within the ``xscen`` folder (with the exception of ``xscen/__init__.py``) will trigger the ``bump-version.yml`` workflow. Be careful not to commit changes to files in this folder when preparing a new release.

#. Create a new branch from `main` (e.g. `release-0.2.0`).
#. Update the `CHANGES.rst` file to change the `Unreleased` section to the current date.
#. Bump the version in your branch to the next version (e.g. `v0.1.0 -> v0.2.0`)::

    $ bump-my-version bump minor # In most cases, we will be releasing a minor version
    $ git push

#. Create a pull request from your branch to `main`.
#. Once the pull request is merged, create a new release on GitHub. On the main branch, run::

    $ git tag v0.2.0
    $ git push --tags

   This will trigger a GitHub workflow to build the package and upload it to TestPyPI. At the same time, the GitHub workflow will create a draft release on GitHub. Assuming that the workflow passes, the final release can then be published on GitHub by finalizing the draft release.

#. Once the release is published, the `publish-pypi.yml` workflow will go into an `awaiting approval` mode on Github Actions. Only authorized users may approve this workflow (notifications will be sent) to trigger the upload to PyPI.

.. warning::

    Uploads to PyPI can **never** be overwritten. If you make a mistake, you will need to bump the version and re-release the package. If the package uploaded to PyPI is broken, you should modify the GitHub release to mark the package as broken, as well as yank the package (mark the version  "broken") on PyPI.

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

Code of Conduct
---------------

Please note that this project is released with a `Contributor Code of Conduct <https://github.com/Ouranosinc/xscen/blob/main/CODE_OF_CONDUCT.md>`_.
By participating in this project you agree to abide by its terms.
