from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
from conftest import SAMPLES_DIR

from xscen import catutils as cu


# Sample files are in the doc folder
@pytest.mark.requires_netcdf
@pytest.mark.parametrize(
    "exts,lens,dirglob,N",
    (
        [{".nc"}, {7}, None, 10],
        [{".zarr"}, {7}, None, 0],
        [{".nc"}, {6}, None, 0],
        [{".nc", ".zarr"}, {6, 7, 8}, "*ssp126*", 2],
    ),
)
def test_find_assets(exts, lens, dirglob, N):
    finder = cu._find_assets(str(SAMPLES_DIR), exts=exts, lengths=lens, dirglob=dirglob)
    assert isinstance(finder, Generator)
    assert len(list(finder)) == N


def test_name_parser():
    d = cu._name_parser(
        "station-obs/GovCan/AHCCD/day/tasmin/tasmin_day_GovCan_AHCCD_1960-1999.nc",
        root="",
        patterns=[
            "{type}/{institution}/{source}/{frequency}/{variable}/{?var}_{?:_}_{DATES}.nc"
        ],
    )
    assert d["type"] == "station-obs"
    assert d["date_start"] == "1960"
    assert d["format"] == "nc"
    assert set(d.keys()) == {
        "type",
        "institution",
        "source",
        "frequency",
        "variable",
        "date_start",
        "date_end",
        "path",
        "format",
    }


def test_parse_dir_filechecks(tmp_path):
    file = tmp_path / "this_is_not_a_netcdf_file.nc"
    file.touch()
    file.chmod(0o111)  # remove read and write perms for all

    # No checks
    parsed = cu._parse_dir(tmp_path, patterns=["{?:_}.nc"], checks=None)
    assert len(parsed) == 1

    # Not readable
    parsed = cu._parse_dir(tmp_path, patterns=["{?:_}.nc"], checks=["readable"])
    assert len(parsed) == 0

    # Not writable
    file.chmod(0o444)  # read-only
    parsed = cu._parse_dir(tmp_path, patterns=["{?:_}.nc"], checks=["writable"])
    assert len(parsed) == 0

    # No checks
    file.chmod(0o777)
    parsed = cu._parse_dir(tmp_path, patterns=["{?:_}.nc"], checks=["ncvalid"])
    assert len(parsed) == 0

    # Just to be sure
    parsed = cu._parse_dir(
        tmp_path, patterns=["{?:_}.nc"], checks=["readable", "writable"]
    )
    assert len(parsed) == 1


@cu.register_parse_type("rev")
def _reverse_word(text):
    return "".join(reversed(text))


@pytest.mark.requires_netcdf
def test_parse_directory():
    df = cu.parse_directory(
        directories=[str(SAMPLES_DIR)],
        patterns=[
            "{activity}/{domain}/{institution}/{source}/{experiment}/{member:rev}/{frequency}/{?:_}.nc"
        ],
        homogenous_info={
            "mip_era": "CMIP6",
            "type": "simulation",
            "processing_level": "raw",
        },
        read_from_file=["variable", "date_start", "date_end", "version"],
        xr_open_kwargs={"engine": "h5netcdf"},
        cvs={
            "domain": {"example-region": "exreg"},
            "experiment": {"ssp126": {"driving_model": "driver"}},
            "attributes": {"version_id": "version"},
        },
        file_checks=["readable", "ncvalid"],
    )

    assert len(df) == 10
    assert (df["activity"] == "ScenarioMIP").all()
    assert (df["mip_era"] == "CMIP6").all()
    assert (df["domain"] == "exreg").all()  # CVS simple
    assert set(df["member"].unique()) == {"1f1p1i1r", "1f1p1i2r"}  # Custom parse type
    assert (
        df[df["frequency"] == "fx"]["variable"] == ("sftlf",)
    ).all()  # Read from file
    assert (
        df[df["experiment"] == "ssp126"]["driving_model"] == "driver"
    ).all()  # CVS complex
    assert (
        df[df["experiment"] != "ssp126"]["driving_model"].isnull().all()
    )  # CVS complex
    assert df.date_start.dtype == "<M8[ms]"
    assert df.date_end.dtype == "<M8[ms]"
    assert (
        df[df["frequency"] == "day"]["date_end"] == pd.Timestamp("2002-12-31")
    ).all()  # Read from file
    # Read from file + attrs cvs
    assert set(
        df[df["id"] == "CMIP6_ScenarioMIP_driver_NCC_NorESM2-MM_ssp126_1f1p1i1r_exreg"][
            "version"
        ]
    ) == {"v20191108", "v20200702"}


@pytest.mark.requires_netcdf
def test_parse_directory_readgroups():
    df = cu.parse_directory(
        directories=[str(SAMPLES_DIR)],
        patterns=[
            "{activity}/{domain}/{institution}/{source}/{experiment}/{member}/{frequency}/{?:_}.nc"
        ],
        read_from_file=[
            ["experiment", "frequency"],
            ["variable", "date_start", "date_end"],
        ],
        cvs={"variable": {"sftlf": None, "tas": "t2m"}},
    )
    assert len(df) == 10
    t2m = df.variable.apply(lambda v: "t2m" in v)
    assert (df[t2m]["date_end"] == pd.Timestamp("2002-12-31")).all()
    assert (df[~t2m].variable.apply(len) == 0).all()


@pytest.mark.requires_netcdf
def test_parse_directory_offcols():
    with pytest.raises(
        ValueError, match="Patterns include fields which are not recognized by xscen"
    ):
        cu.parse_directory(
            directories=[str(SAMPLES_DIR)],
            patterns=[
                "{activité}/{domain}/{institution}/{source}/{experiment}/{member}/{frequency}/{?:_}.nc"
            ],
        )
    df = cu.parse_directory(
        directories=[str(SAMPLES_DIR)],
        patterns=[
            "{activité}/{domain}/{institution}/{source}/{experiment}/{member}/{frequency}/{?:_}.nc"
        ],
        only_official_columns=False,
    )
    assert (df["activité"] == "ScenarioMIP").all()


@pytest.mark.requires_netcdf
def test_parse_directory_idcols():
    df = cu.parse_directory(
        directories=[str(SAMPLES_DIR)],
        patterns=[
            "{activity}/{domain}/{institution}/{source}/{experiment}/{member}/{frequency}/{?:_}.nc"
        ],
        only_official_columns=False,
        id_columns=["domain", "institution"],
    )
    assert (df["id"] == "example-region_NCC").all()


def test_parse_from_ds():
    # Real ds
    ds = xr.tutorial.open_dataset("air_temperature")
    info = cu.parse_from_ds(
        ds, names=["frequency", "source", "variable"], attrs_map={"platform": "source"}
    )
    assert info == {"frequency": "6hr", "source": "Model", "variable": ("air",)}


def test_parse_from_zarr(tmp_path):
    # Real ds
    ds = xr.tutorial.open_dataset("air_temperature")
    ds.to_zarr(tmp_path / "air.zarr")
    info = cu.parse_from_ds(
        tmp_path / "air.zarr",
        names=["frequency", "source", "variable"],
        attrs_map={"platform": "source"},
    )
    assert info == {"frequency": "6hr", "source": "Model", "variable": ("air",)}


def test_parse_from_nc(tmp_path):
    # Real ds
    ds = xr.tutorial.open_dataset("air_temperature")
    ds.to_netcdf(tmp_path / "air.nc")
    info = cu.parse_from_ds(
        tmp_path / "air.nc",
        names=["frequency", "source", "variable"],
        attrs_map={"platform": "source"},
    )
    assert info == {"frequency": "6hr", "source": "Model", "variable": ("air",)}


def test_build_path(samplecat):
    df = cu.build_path(samplecat, root="/test", mip_era="CMIP5")
    assert (
        "/test/simulation/raw/CMIP5/ScenarioMIP/example-region/NCC/NorESM2-MM/ssp585/r1i1p1f1/fx/sftlf/"  # pragma: allowlist secret
        "sftlf_fx_CMIP5_ScenarioMIP_example-region_NCC_NorESM2-MM_ssp585_r1i1p1f1_fx.nc"
    ) in df.new_path.values


def test_build_path_ds():
    ds = xr.tutorial.open_dataset("air_temperature")
    ds = ds.assign(time=xr.cftime_range("0001-01-01", freq="6H", periods=ds.time.size))
    ds.attrs.update(source="source", institution="institution")
    new_path = cu.build_path(
        ds,
        schemas={
            "folders": ["source", "institution", ["variable", "xrfreq"]],
            "filename": ["source", "institution", "variable", "frequency", "DATES"],
        },
    )
    assert new_path == Path(
        "source/institution/air_6H/source_institution_air_6hr_0001-0002"
    )


def test_build_path_multivar(samplecat):
    info = samplecat.df.iloc[0].copy()
    info["variable"] = ("tas", "tasmin")
    with pytest.raises(
        ValueError,
        match="Selected schema original-sims-raw is meant to be used with single-variable datasets.",
    ):
        cu.build_path(info)
