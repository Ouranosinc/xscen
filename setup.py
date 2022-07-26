#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()


# This list are the minimum requirements for xscen
# this is only meant to make `pip check` work
# xscen dependencies can only be installed through conda-forge.
requirements = [
    "intake_esm>=2021.8.17",
    "xesmf>=0.6.2",  # This is not available on pypi.
    "xarray",
    "xclim>=0.37",
    "dask",
    "pandas",
    "clisops",
    "rechunker",
    "pyyaml",
    "matplotlib",
    "cartopy",
    "h5py",
    "zarr",
    "netCDF4",
]

setup(
    author="Gabriel Rondeau-Genesse",
    author_email="rondeau-genesse.gabriel@ouranos.ca",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="A climate change scenario-building analysis framework, built with xclim/xarray.",
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="xscen",
    name="xscen",
    packages=find_packages(include=["xscen", "xscen.*"]),
    test_suite="tests",
    tests_require=["pytest"],
    url="https://github.com/Ouranosinc/xscen",
    version="0.2.3-beta",
    zip_safe=False,
)
