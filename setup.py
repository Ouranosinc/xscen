#!/usr/bin/env python

"""The setup script."""

import re

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

# Copied from xclim. Needed for replacing roles with hyperlinks in documentation on PyPI
hyperlink_replacements = {
    r":issue:`([0-9]+)`": r"`GH/\1 <https://github.com/Ouranosinc/xscen/issues/\1>`_",
    r":pull:`([0-9]+)`": r"`PR/\1 <https://github.com/Ouranosinc/xscen/pull/\1>`_",
    r":user:`([a-zA-Z0-9_.-]+)`": r"`@\1 <https://github.com/\1>`_",
}
for search, replacement in hyperlink_replacements.items():
    history = re.sub(search, replacement, history)

# This list are the minimum requirements for xscen
# this is only meant to make `pip check` work
# xscen dependencies can only be installed through conda-forge.
requirements = [
    "cartopy",
    "cftime",
    "cf_xarray>=0.7.6",
    "clisops>=0.9.5",
    "dask",
    "flox",
    "fsspec",
    "geopandas",
    "h5netcdf",
    "intake",
    "intake-esm>=2022.9.18",
    "matplotlib",
    "netCDF4",
    "numpy",
    "pandas!=1.5.3",
    "pyarrow",  # Used when opening catalogs.
    "pygeos",
    "pyyaml",
    "rechunker",
    "shapely",
    "xarray<2023.3.0",  # See https://github.com/Ouranosinc/xscen/pull/173
    "xclim>=0.37",
    "xesmf>=0.7",  # This is not available on pypi.
    "zarr",
]

setup(
    author="Gabriel Rondeau-Genesse",
    author_email="rondeau-genesse.gabriel@ouranos.ca",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    description="A climate change scenario-building analysis framework, built with xclim/xarray.",
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="xscen",
    name="xscen",
    packages=find_packages(include=["xscen", "xscen.*"]),
    project_urls={
        "About Ouranos": "https://www.ouranos.ca/en/",
        "Changelog": "https://xscen.readthedocs.io/en/stable/history.html",
        "Issue tracker": "https://github.com/Ouranosinc/xscen/issues",
    },
    test_suite="tests",
    tests_require=["pytest", "pytest-cov"],
    url="https://github.com/Ouranosinc/xscen",
    version="0.6.0",
    zip_safe=False,
)
