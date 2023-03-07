.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

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

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

xscen could always use more documentation, whether as part of the
official xscen docs, in docstrings, or even on the web in blog posts,
articles, and such.

Currently, the documentation includes notebooks and sample catalogs which use private
internal data from Ouranos. As such, they are not included in this public repo. Rather,
a git submodule allows people with access to the Ouranos VPN to recursively fetch them
from an internal Gitlab server.

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

Ready to contribute? Here's how to set up `xscen` for local development.

1. Clone the repo locally, including the notebooks::

    $ git clone --recurse-submodules git@github.com:Ouranosinc/xscen.git

If this command is not executed from within the Ouranos VPN, it will show error messages about
the `docs/notebooks` submodule not being fetched, but this doesn't affect the usage of xscen in general.

2. Install your local copy into an isolated environment. We usually use `conda` for this::

    $ cd xscen/
    $ conda env create -f environment-dev.yml
    $ pip install -e .

3. To ensure a consistent style, please install the pre-commit hooks to your repo::

    $ pre-commit install

   Special style and formatting checks will be run when you commit your changes. You
   can always run the hooks on their own with:

    $ pre-commit run -a

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8, black, and the
   tests, including testing other Python versions with tox::

    $ tox

   To get flake8, black, and tox, just pip install them into your virtualenv.

.. warning::

   Due to some dependencies only being available via Anaconda/conda-forge, `tox` will only work if both `tox` and `tox-conda`
   are installed in a conda-based environment. Running `pytest` demands that your runtime/dev environment have all necessary
   dependencies installed.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. If you are editing the docs (but not the notebooks, nor the sample catalogs),
   compile and open them with::

    $ make docs
    # or to simply generate the html
    $ cd docs/
    $ make html

8. If you are editing the notebooks or the sample catalogs, building the docs is done
   the same way, but you have to commit your changes to the internal `xscen-notebooks`
   repository instead of the public one. This assumes you are in the Ouranos VPN and
   have a Gitlab account on the internal server.::

   $ cd docs/notebooks # Now git commands are made to the notebook repo.
   $ git checkout -b new-branch
   $ git commit -am "detailed description of your changes"
   $ git push origin new-branch
   $ cd ../..
   $ git commit -am 'updated the notebooks'  # This updates the commit to which the submodule points

9. Submit a pull request through the GitHub website, or the gitlab one for changes to the notebooks
   and sample catalogs. If your changes necessitate both a github PR and a gitlab MR, please make sure
   that the notebooks changes have been merged and the submodules updated before merging the main PR.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should not break the templates.
4. The pull request should work for Python 3.8, 3.9, and 3.10.

Tips
----

To run a subset of tests::

$ pytest tests.test_xscen

Versioning/Tagging
------------------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
The templates must also be tested manually before each release.
Then run::

$ bumpversion patch # possible: major / minor / patch
$ git push
$ git push --tags
