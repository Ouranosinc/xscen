.PHONY: clean clean-build clean-pyc clean-test docs help install lint lint/flake8 lint/black
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

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-docs clean-pyc clean-test ## remove all build, test, docs, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-docs: ## remove documentation artifacts
	rm -fr docs/notebooks/_data/
	rm -fr docs/notebooks/.ipynb_checkpoints/
	rm -f docs/apidoc/xscen*.rst
	rm -f docs/apidoc/modules.rst
	$(MAKE) -C docs clean

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint/flake8: ## check style with flake8
	ruff xscen tests
	flake8 --config=.flake8 xscen tests

lint/black: ## check style with black
	black --check xscen tests
	blackdoc --check xscen docs
	isort --check xscen tests

lint: lint/black lint/flake8 ## check style

test: ## run tests quickly with the default Python
	python -m pytest

test-all: ## run tests on every Python version with tox
	tox

autodoc: clean-docs ## create sphinx-apidoc files
	sphinx-apidoc -o docs/apidoc --module-first xscen

linkcheck: autodoc ## run checks over all external links found throughout the documentation
	env SKIP_NOTEBOOKS=1 $(MAKE) -C docs linkcheck

docs: autodoc ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs html
ifndef READTHEDOCS
	$(BROWSER) docs/_build/html/index.html
endif

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

dist: clean ## builds source and wheel package
	python -m build --sdist
	python -m build --wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python -m pip install .

dev: clean ## install the package in editable mode with all development dependencies
	python -m pip install --editable ".[dev]"

findfrench:  ## Extract phrases and update the French translation catalog (this doesn't translate)
	python setup.py extract_messages
	python setup.py update_catalog -l fr

translate:   ## Compile the translation catalogs.
	python setup.py compile_catalog
