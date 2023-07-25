import numpy as np
import pytest

import xscen as xs


class TestRechunkForSaving:
    @pytest.mark.parametrize(
        "dims, xy",
        [
            (["lon", "lat"], True),
            (["lon", "lat"], False),
            (["rlon", "rlat"], True),
            (["rlon", "rlat"], False),
        ],
    )
    def test_options(self, datablock_3d, dims, xy):
        ds = datablock_3d(
            np.random.random((30, 30, 50)),
            variable="tas",
            x=dims[0],
            x_start=-70 if dims[0] == "lon" else 0,
            y=dims[1],
            y_start=45 if dims[1] == "lat" else 0,
            as_dataset=True,
        )
        x = "X" if xy else dims[0]
        y = "Y" if xy else dims[1]
        new_chunks = {y: 10, x: 10, "time": 20}
        ds_ch = xs.io.rechunk_for_saving(ds, new_chunks)
        for dim, chunks in ds_ch.chunks.items():
            dim = (
                dim
                if (xy is False or dim == "time")
                else "X"
                if dim == dims[0]
                else "Y"
            )
            assert chunks[0] == new_chunks[dim]

    def test_variables(self, datablock_3d):
        ds = datablock_3d(
            np.random.random((30, 30, 50)),
            variable="tas",
            x="lon",
            x_start=-70,
            y="lat",
            y_start=45,
            as_dataset=True,
        )
        ds["pr"] = datablock_3d(
            np.random.random((30, 30, 50)),
            variable="pr",
            x="lon",
            x_start=-70,
            y="lat",
            y_start=45,
            as_dataset=False,
        )

        new_chunks = {
            "tas": {"time": 20, "lon": 10, "lat": 7},
            "pr": {"time": 10, "lon": 5, "lat": 3},
        }
        ds_ch = xs.io.rechunk_for_saving(ds, new_chunks)
        for v in ds_ch.data_vars:
            for dim, chunks in zip(list(ds.dims), ds_ch[v].chunks):
                assert chunks[0] == new_chunks[v][dim]
