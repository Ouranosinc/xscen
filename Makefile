.PHONY:  clean clean-build clean-pyc clean-test coverage development dist docs help install lint release test
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"
LOCALES := docs/locales

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-docs clean-pyc clean-test ## remove all build, test, docs, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-docs: ## remove documentation artifacts
	rm -fr docs/notebooks/_data/
	rm -fr docs/notebooks/.ipynb_checkpoints/
	rm -f docs/apidoc/xscen*.rst
	rm -f docs/apidoc/modules.rst
	rm -f docs/locales/fr/LC_MESSAGES/*.mo
	$(MAKE) -C docs clean

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .coverage/
	rm -fr .pytest_cache
	rm -fr .tox/
	rm -fr htmlcov/

install-lint: ## install dependencies needed for linting
	python -m pip install --quiet --group lint

install-docs: ## install dependencies needed for building the docs
	python -m pip install --quiet --group docs

install-test: ## install dependencies needed for standard testing
	python -m pip install --quiet --group test

install-tox: ## install base dependencies needed for running tox
	python -m pip install --quiet --group tox

lint: install-lint ## check style
	python -m ruff check src/xscen tests
	python -m flake8 --config=.flake8 src/xscen tests
	# python -m numpydoc lint src/xscen/**.py # FIXME: disabled until the codebase is fully numpydoc compliant
	python -m vulture src/xscen tests
	codespell src/xscen tests docs
	python -m deptry src
	python -m yamllint --config-file=.yamllint.yaml src/xscen

test: install-test ## run tests quickly with the default Python
	python -m pytest

test-all: install-tox ## run tests on every Python version with tox
	python -m tox

initialize-translations: clean-docs ## initialize translations, ignoring autodoc-generated files
	${MAKE} -C docs gettext
	sphinx-intl update -p docs/_build/gettext -d docs/locales -l fr

autodoc: install-docs clean-docs ## create sphinx-apidoc files:
	sphinx-apidoc -o docs/apidoc --module-first src/xscen

linkcheck: autodoc ## run checks over all external links found throughout the documentation
	env SKIP_NOTEBOOKS=1 $(MAKE) -C docs linkcheck

build-docs: autodoc ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs html
	$(MAKE) -C docs html BUILDDIR="_build/html/en"
ifneq ("$(wildcard $(LOCALES))","")
	${MAKE} -C docs gettext
	$(MAKE) -C docs html BUILDDIR="_build/html/fr" SPHINXOPTS="-D language='fr'"
endif

docs: build-docs  ## open the built documentation in a web browser
ifndef READTHEDOCS
	$(BROWSER) docs/_build/html/index.html
	$(BROWSER) docs/_build/html/en/html/index.html
endif

servedocs: autodoc ## compile the docs while watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

dist: clean ## builds source and wheel package
	python -m flit build
	ls -l dist

release: dist ## package and upload a release
	python -m flit publish dist/*

install: clean ## install the package to the active Python's site-packages
	python -m pip install --no-user .

development: clean ## install the package to the active Python's site-packages
	python -m pip install --group dev
	python -m pip install --no-user --editable .[extras]
	prek install

findfrench:  ## Extract phrases and update the French translation catalog (this doesn't translate)
	pybabel extract -o src/xscen/data/messages.pot --omit-header --input-dirs=src/xscen/
	pybabel update -l fr -D xscen -i src/xscen/data/messages.pot -d src/xscen/data/ --omit-header --no-location

translate:   ## Compile the translation catalogs.
	pybabel compile -f -D xscen -d src/xscen/data/

MO_LAST_COMMIT = $(shell git log -n 1 --pretty=format:%H -- src/xscen/data/fr/LC_MESSAGES/xscen.mo)
PO_LAST_COMMIT = $(shell git log -n 1 --pretty=format:%H -- src/xscen/data/fr/LC_MESSAGES/xscen.po)
checkfrench:  ## Error if the catalog could be update or if the compilation is older than the catalog.
	rm -f .check_messages.pot
	pybabel extract -o .check_messages.pot --omit-header --input-dirs=src/xscen/ --no-location
	pybabel update -l fr -D xscen -i .check_messages.pot -d src/xscen/data/ --omit-header --check
	rm -f .check_messages.pot
	# Last commit that touched the PO file must be an ancestor of the last that touched the MO
	if git merge-base --is-ancestor $(PO_LAST_COMMIT) $(MO_LAST_COMMIT); then echo "ok"; else echo "Compilation is older than translations. Please compile with 'make translate'."; exit 1; fi
