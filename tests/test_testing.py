from pathlib import Path

import pytest

import xscen as xs


class TestPublish:
    @pytest.mark.requires_netcdf
    @pytest.mark.parametrize("fmt", ["md", "rst"])
    def test_normal(self, fmt):
        out = xs.testing.publish_release_notes(
            fmt,
            changes=Path(__file__).parent.parent.joinpath("CHANGELOG.rst"),
            latest=False,
        )
        if fmt == "md":
            assert out.startswith("# Changelog\n\n")
            assert "[PR/413](https://github.com/Ouranosinc/xscen/pull/413)" in out
        elif fmt == "rst":
            assert out.startswith("=========\nChangelog\n=========\n\n")
            assert "`PR/413 <https://github.com/Ouranosinc/xscen/pull/\\>`_" in out

    def test_error(self):
        with pytest.raises(FileNotFoundError):
            xs.testing.publish_release_notes("md", changes="foo")
        with pytest.raises(NotImplementedError):
            xs.testing.publish_release_notes("foo", changes=Path(__file__).parent.parent.joinpath("CHANGELOG.rst"))

    @pytest.mark.requires_netcdf
    def test_file(self, tmpdir):
        xs.testing.publish_release_notes(
            "md",
            file=tmpdir / "foo.md",
            changes=Path(__file__).parent.parent.joinpath("CHANGELOG.rst"),
        )
        with Path(tmpdir).joinpath("foo.md").open(encoding="utf-8") as f:
            assert f.read().startswith("# Changelog\n\n")

    @pytest.mark.parametrize("latest", [True, False])
    @pytest.mark.requires_netcdf
    def test_latest(self, tmpdir, latest):
        out = xs.testing.publish_release_notes(
            "md",
            changes=Path(__file__).parent.parent.joinpath("CHANGELOG.rst"),
            latest=latest,
        )
        if latest:
            assert len(out.split("\n\n## v0.")) == 2
        else:
            assert len(out.split("\n\n## v0.")) > 2


def test_show_version(tmpdir):
    xs.testing.show_versions(file=tmpdir / "versions.txt")
    with Path(tmpdir).joinpath("versions.txt").open(encoding="utf-8") as f:
        out = f.read()
    assert "xscen" in out
    assert "xclim" in out
    assert "xesmf" in out
    assert "xarray" in out
    assert "numpy" in out
    assert "pandas" in out
    assert "dask" in out
    assert "cftime" in out
    assert "netcdf4" in out
