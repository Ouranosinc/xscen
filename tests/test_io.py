import numpy as np
import pytest
from xclim.testing.helpers import test_timeseries as timeseries
import pandas as pd
import xarray as xr
import numpy as np
import xscen as xs


class TestGenericRechunkingDims:

    def get_test_dataset(self, data: np.ndarray = None):

        """Create a 3D test Dataset.

        Parameters
        ----------
        data : a 3D numpy array to populate the dataset, dimensionsa are interpreted [lat, lon, time]
                (TODO: do with xarray.DataArray)

        Returns
        -------
        ds : Dataset with cf information.
        """

        if isinstance(data, np.ndarray) and len(data.shape) == 3:
            tas = data
            lat = np.arange(0, data.shape[0]) * 90 / data.shape[0]
            lon = np.arange(0, data.shape[1]) * 180 / data.shape[1]
            time = pd.date_range("2010-10-18", periods=data.shape[2])
        else:
            tas = [[[14.7, 10.1, 17.0],
                    [3.4, 23.5, 15.1]],
                   [[8.5, 12.4, 25.8],
                    [14.5, 5.8, 15.3]]]
            lon = [-88.83, -88.32]
            lat = [44.25, 44.21]
            time = pd.date_range("2010-10-18", periods=3)
        ds = xr.Dataset(
            data_vars=dict(
                tas=(["lat", "lon", "time"], tas, {'standard_name': 'air_temperature'}),
            ),
            coords=dict(
                lon=(["lon"], lon, {'standard_name': 'Longitude', 'axis': 'X'}),
                lat=(["lat"], lat, {'standard_name': 'Latitude', 'axis': 'Y'}),
                time=(['time'], time, {'standard_name': 'time'})
            ),
            attrs=dict(description="Climate data dummy."),
        )
        return ds

    def get_test_dataset_rlat_rlon(self, data: np.ndarray = None):

        ds = self.get_test_dataset(data)
        # make dummy rlat / rlon
        ds = ds.assign_coords({'rlon': ('lon', ds.lon.values), 'rlat': ('lat', ds.lat.values)})
        ds = ds.swap_dims({'lat': 'rlat', 'lon': 'rlon'})
        ds.coords['rlon'] = ds.coords['rlon'].assign_attrs(ds['lon'].attrs)
        ds.coords['rlon'] = ds.coords['rlon'].assign_attrs({'standard_name': 'grid_longitude'})
        ds.coords['rlat'] = ds.coords['rlat'].assign_attrs(ds['lat'].attrs)
        ds.coords['rlat'] = ds.coords['rlat'].assign_attrs({'standard_name': 'grid_latitude'})
        ds = ds.drop_vars(['lat', 'lon'])

        return ds

    def test_rechunk_for_saving_lat_lon(self):

        ds = self.get_test_dataset(np.random.random((30, 30, 50)))
        new_chunks = {'lat': 10, 'lon': 10, 'time': 20}
        ds_ch = xs.io.rechunk_for_saving(ds, new_chunks)
        for dim, chunks in ds_ch.chunks.items():
            assert chunks[0] == new_chunks[dim]

    def test_rechunk_for_saving_XY_lat_lon(self):

        ds = self.get_test_dataset(np.random.random((30, 30, 50)))
        new_chunks = {'X': 10, 'Y': 10, 'time': 20}
        ds_ch = xs.io.rechunk_for_saving(ds, new_chunks)
        for dim, chunks in zip(['X', 'Y', 'time'], ds_ch.chunks.values()):
            assert chunks[0] == new_chunks[dim]

    def test_rechunk_for_saving_rlat_rlon(self):

        ds = self.get_test_dataset_rlat_rlon(np.random.random((30, 30, 50)))
        new_chunks = {'rlat': 10, 'rlon': 10, 'time': 20}
        ds_ch = xs.io.rechunk_for_saving(ds, new_chunks)
        for dim, chunks in ds_ch.chunks.items():
            assert chunks[0] == new_chunks[dim]

    def test_rechunk_for_saving_XY_rlat_lon(self):

        ds = self.get_test_dataset_rlat_rlon(np.random.random((30, 30, 50)))
        new_chunks = {'X': 10, 'Y': 10, 'time': 20}
        ds_ch = xs.io.rechunk_for_saving(ds, new_chunks)
        for dim, chunks in zip(['X', 'Y', 'time'], ds_ch.chunks.values()):
            assert chunks[0] == new_chunks[dim]


