#!/usr/bin/env python
"""Tests for `xscen` package."""

import pathlib
import pkgutil


class TestSmokeTest:
    def test_version(self):
        import xscen

        with pathlib.Path(__file__).parent.parent.joinpath(
            "setup.cfg"
        ).open() as reader:
            for line in reader.readlines():
                if line.startswith("current_version"):
                    assert xscen.__version__ == line.split()[-1]
                    break

    def test_package_metadata(self):
        """Test the package metadata."""

    project = pkgutil.get_loader("xscen").get_filename()

    metadata = pathlib.Path(project).resolve().parent.joinpath("__init__.py")

    with open(metadata) as f:
        contents = f.read()
        assert """Gabriel Rondeau-Genesse""" in contents
        assert '__email__ = "rondeau-genesse.gabriel@ouranos.ca"' in contents
        assert '__version__ = "0.7.1"' in contents
