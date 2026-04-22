"""Zipped Zarr support for Zarr 3."""

import os
from pathlib import Path

from xarray import Dataset
from xarray.backends.common import T_PathFileOrDataStore
from xarray.backends.zarr import ZarrBackendEntrypoint
from zarr.storage import ZipStore


class ZarrZipBackendEntrypoint(ZarrBackendEntrypoint):
    """Backend for ".zarr.zip" files based on xarray's builtin zarr backend."""

    def guess_can_open(self, filename_or_obj: T_PathFileOrDataStore) -> bool:  # noqa: D102
        if isinstance(filename_or_obj, str | os.PathLike):
            # allow a trailing slash to account for an autocomplete
            # adding it.
            exts = Path(str(filename_or_obj).rstrip("/")).suffixes
            return exts[-2:] == [".zarr", ".zip"]

        return False

    def open_dataset(self, filename_or_obj: T_PathFileOrDataStore, **kwargs) -> Dataset:  # noqa: D102

        return super().open_dataset(ZipStore(filename_or_obj), **kwargs)

    def open_groups_as_dict(self, filename_or_obj: T_PathFileOrDataStore, **kwargs) -> dict[str, Dataset]:  # noqa: D102
        return super().open_groups_as_dict(ZipStore(filename_or_obj), **kwargs)
