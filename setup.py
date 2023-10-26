#!/usr/bin/env python
"""The setup script."""
from setuptools import find_packages, setup
from setuptools.command.install import install


class InstallWithCompile(install):
    """Injection of the catalog compilation in the installation process."""

    # Taken from https://stackoverflow.com/a/41120180/9291575

    def run(self):
        """Install the package, but compile the i18n catalogs first."""
        from babel.messages.frontend import compile_catalog

        compiler = compile_catalog(self.distribution)
        option_dict = self.distribution.get_option_dict("compile_catalog")
        compiler.domain = [option_dict["domain"][1]]
        compiler.directory = option_dict["directory"][1]
        compiler.run()
        super().run()


with open("README.rst") as readme_file:
    readme = readme_file.read()

# Copied from xclim. Needed for replacing roles with hyperlinks in documentation on PyPI
# hyperlink_replacements = {
#     r":issue:`([0-9]+)`": r"`GH/\1 <https://github.com/Ouranosinc/xscen/issues/\1>`_",
#     r":pull:`([0-9]+)`": r"`PR/\1 <https://github.com/Ouranosinc/xscen/pull/\1>`_",
#     r":user:`([a-zA-Z0-9_.-]+)`": r"`@\1 <https://github.com/\1>`_",
# }
# for search, replacement in hyperlink_replacements.items():
#     history = re.sub(search, replacement, history)

# This list are the minimum requirements for xscen
# this is only meant to make `pip check` work
# xscen dependencies can only be installed through conda-forge.
requirements = [
    "cartopy",
    "cftime",
    "cf_xarray>=0.7.6",
    "clisops>=0.10",
    "dask",
    "flox",
    "fsspec<2023.10.0",
    "geopandas",
    "h5netcdf",
    "h5py",
    "intake-esm>=2023.07.07",
    "matplotlib",
    "netCDF4",
    "numpy",
    "pandas>=2.0",
    "parse",
    "pyarrow",  # Used when opening catalogs.
    "pyyaml",
    "rechunker",
    "shapely>=2.0",
    "sparse",
    "toolz",
    "xarray",
    "xclim>=0.43",
    "xesmf>=0.7.0",
    "zarr",
]

dev_requirements = ["pytest", "pytest-cov", "xdoctest"]

setup(
    author="Gabriel Rondeau-Genesse",
    author_email="rondeau-genesse.gabriel@ouranos.ca",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    cmdclass={"install": InstallWithCompile},
    description="A climate change scenario-building analysis framework, built with xclim/xarray.",
    install_requires=requirements,
    long_description=readme,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="xscen",
    message_extractors={"xscen": [("**.py", "python", None)]},
    name="xscen",
    packages=find_packages(include=["xscen", "xscen.*"]),
    project_urls={
        "About Ouranos": "https://www.ouranos.ca/en/",
        "Changelog": "https://xscen.readthedocs.io/en/stable/history.html",
        "Issue tracker": "https://github.com/Ouranosinc/xscen/issues",
    },
    setup_requires=["babel"],
    test_suite="tests",
    extras_require={"dev": dev_requirements},
    url="https://github.com/Ouranosinc/xscen",
    version="0.7.15-beta",
    zip_safe=False,
)
