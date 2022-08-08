#!/usr/bin/env python
"""Tests for `xscen` package."""
from pathlib import Path


class TestSmokeTest:
    def test_version(self):
        import xscen

        with Path(__file__).parent.parent.joinpath("setup.cfg").open() as reader:
            for line in reader.readlines():
                if line.startswith("current_version"):
                    assert xscen.__version__ == line.split()[-1]
                    break
