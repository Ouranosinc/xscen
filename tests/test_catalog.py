from pathlib import Path

import pandas as pd
import xarray as xr
from conftest import SAMPLES_DIR, notebooks

import xscen as xs
from xscen import catalog, extract


def test_subset_file_coverage():
    df = pd.DataFrame.from_records(
        [
            {"id": "A", "date_start": "1950-01-01", "date_end": "1959-12-31"},
            {"id": "A", "date_start": "1960-01-01", "date_end": "1969-12-31"},
        ]
    )
    df["date_start"] = pd.to_datetime(df.date_start)
    df["date_end"] = pd.to_datetime(df.date_end)

    # Ok
    pd.testing.assert_frame_equal(catalog.subset_file_coverage(df, [1951, 1970], coverage=0.8), df)

    # Insufficient coverage (no files)
    pd.testing.assert_frame_equal(
        catalog.subset_file_coverage(df, [1975, 1986], coverage=0.8),
        pd.DataFrame(columns=df.columns),
    )

    # Insufficient coverage (partial files)
    pd.testing.assert_frame_equal(
        catalog.subset_file_coverage(df, [1951, 1976], coverage=0.8),
        pd.DataFrame(columns=df.columns),
    )


def test_subset_file_coverage_2300():
    df = pd.DataFrame.from_records(
        [
            {"id": "A", "date_start": "2100-01-01", "date_end": "2199-12-31"},
            {"id": "A", "date_start": "2200-01-01", "date_end": "2270-12-31"},
            {"id": "A", "date_start": "2271-01-01", "date_end": "2300-12-31"},
        ]
    )

    pd.testing.assert_frame_equal(catalog.subset_file_coverage(df, [2250, 2305], coverage=0.8), df[1:3])


def test_xrfreq_fix():
    cat = catalog.DataCatalog(SAMPLES_DIR.parent / "pangeo-cmip6.json")
    assert set(cat.df.xrfreq) == {"3h", "D", "fx"}


class TestCopyFiles:
    def test_flat(self, samplecat, tmp_path):
        newcat = samplecat.copy_files(tmp_path, flat=True)
        assert len(list(tmp_path.glob("*.nc"))) == len(newcat.df)

    def test_inplace(self, samplecat, tmp_path):
        dsid, scat = extract.search_data_catalogs(
            data_catalogs=[samplecat],
            variables_and_freqs={"tas": "MS"},
            allow_resampling=True,
            other_search_criteria={
                "experiment": "ssp585",
                "source": "NorESM.*",
                "member": "r1i1p1f1",
            },
        ).popitem()
        scat.copy_files(tmp_path, inplace=True)
        assert len(list(tmp_path.glob("*.nc"))) == len(scat.df)

        _, ds = extract.extract_dataset(scat).popitem()
        frq = xr.infer_freq(ds.time)
        assert frq == "MS"

    def test_zipunzip(self, samplecat, tmp_path):
        dsid, scat = extract.search_data_catalogs(
            data_catalogs=[samplecat],
            variables_and_freqs={"tas": "D"},
            allow_resampling=True,
            other_search_criteria={
                "experiment": "ssp585",
                "source": "NorESM.*",
                "member": "r1i1p1f1",
            },
        ).popitem()
        _, ds = extract.extract_dataset(scat).popitem()
        ds.to_zarr(tmp_path / "temp.zarr")
        scat.esmcat.df.loc[0, "path"] = tmp_path / "temp.zarr"

        rz = tmp_path / "zipped"
        rz.mkdir()
        scat_z = scat.copy_files(rz, zipzarr=True)
        f = Path(scat_z.df.path.iloc[0])
        assert f.suffix == ".zip"
        assert f.parent.name == rz.name
        assert f.is_file()

        ru = tmp_path / "unzipped"
        ru.mkdir()
        scat_uz = scat.copy_files(ru, unzip=True)
        f = Path(scat_uz.df.path.iloc[0])
        assert f.suffix == ".zarr"
        assert f.parent.name == ru.name
        assert f.is_dir()


def test_from_df():
    df = xs.parse_directory(
        directories=[SAMPLES_DIR],
        patterns=["{activity}/{domain}/{institution}/{source}/{experiment}/{member}/{frequency}/{?:_}.nc"],
        homogenous_info={
            "mip_era": "CMIP6",
            "type": "simulation",
            "processing_level": "raw",
        },
        read_from_file=["variable", "date_start", "date_end"],
    )

    cat = xs.catalog.DataCatalog.from_df(df)

    assert (len(cat.df) == len(df)) and all(cat.df.columns == df.columns)


def test_from_df_with_files():
    cat = xs.catalog.DataCatalog.from_df(notebooks / "samples" / "pangeo-cmip6.csv", esmdata=notebooks / "samples" / "pangeo-cmip6.json")

    assert len(cat.df) == 47


def test_search_period():
    cat = catalog.DataCatalog(SAMPLES_DIR.parent / "pangeo-cmip6.json")

    scat = cat.search(periods=["2015", "2016"])

    assert len(scat.df) == 17

    assert scat.df.date_end.min() >= pd.Timestamp("2016-12-31 00:00:00")
    assert scat.df.date_start.max() <= pd.Timestamp("2015-01-01 00:00:00")


def test_search_nothing():
    cat = catalog.DataCatalog(SAMPLES_DIR.parent / "pangeo-cmip6.json")

    scat = cat.search()

    assert (scat.df == cat.df).all().all()


def test_exist_in_cat(samplecat):
    assert samplecat.exists_in_cat(variable="tas")
    assert not samplecat.exists_in_cat(variable="nonexistent_variable")


def test_to_dataset(samplecat):
    ds = samplecat.search(member="r1i1p1f1", variable="tas").to_dataset(concat_on="experiment")

    assert "experiment" in ds.dims


def test_to_dataset_ensemble(samplecat):
    ds = samplecat.search(member="r1i1p1f1", variable="tas").to_dataset(create_ensemble_on="experiment")

    assert "realization" in ds.dims
    assert "ssp126" in ds.realization.values


def test_project_catalog_create_and_update(tmpdir):
    root = str(tmpdir / "_data")
    pcat = xs.catalog.ProjectCatalog(f"{root}/test.json", create=True, project={"title": "Test Project"})

    assert Path(f"{root}/test.json").exists()

    lpcat = len(pcat.df)

    df = xs.parse_directory(
        directories=[SAMPLES_DIR],
        patterns=["{activity}/{domain}/{institution}/{source}/{experiment}/{member}/{frequency}/{?:_}.zarr.zip"],
        homogenous_info={
            "mip_era": "CMIP6",
            "type": "simulation",
            "processing_level": "raw",
        },
        read_from_file=["variable", "date_start", "date_end"],
    )

    pcat.update(df)

    assert len(pcat.df) == lpcat + len(df)

    path = SAMPLES_DIR / "ScenarioMIP/example-region/NCC/NorESM2-MM/ssp126/r1i1p1f1/day/ScenarioMIP_NCC_NorESM2-MM_ssp126_r1i1p1f1_gn_raw.nc"
    ds = xr.open_dataset(path)
    pcat.update_from_ds(ds, path, info_dict={"experiment": "ssp999"}, variable="tas")

    assert pcat.df.iloc[-1].experiment == "ssp999"
    assert "tas" in pcat.df.iloc[-1].variable
