"""Zipped Zarr support for Zarr 3."""

# FIXME: Remove this when Zarr supports transparent opening of zips (probably through ZEP8 url pipeline)
import os
from pathlib import Path

from xarray.backends.common import T_PathFileOrDataStore
from xarray.backends.zarr import ZarrBackendEntrypoint
from zarr.storage import ZipStore, _common


# Monkeypatch Zarr 3
old_make_store = _common.make_store


async def make_store(
    store_like,
    *,
    mode=None,
    storage_options=None,
):
    """Drop-in replacement for zarr.storage._common.make_store that opens zip paths directly in ZipStores."""  # numpydoc ignore=PR01,RT01
    if (isinstance(store_like, Path) and store_like.suffix == ".zip") or (isinstance(store_like, str) and store_like.endswith(".zip")):
        return await ZipStore.open(path=store_like, mode=mode, read_only=(mode == "r"))

    # Zarr is very strict and refuses to work with non-none storage options if store_like is not a fsspec object
    if storage_options == {}:
        storage_options = None
    return await old_make_store(store_like, mode=mode, storage_options=storage_options)


_common.__dict__["make_store"] = make_store


# Xarray backend entry point to make the fix above load automatically and make xarray aware of its ability to open zipped zarrs
class ZarrZipBackendEntrypoint(ZarrBackendEntrypoint):
    """Backend for ".zarr.zip" files based on xarray's builtin zarr backend."""

    def guess_can_open(self, filename_or_obj: T_PathFileOrDataStore) -> bool:
        """
        Inform xarray this entrypoint can open zipped zarrs.

        Parameters
        ----------
        filename_or_obj : Path or Data Store
            Filename or data store.

        Returns
        -------
        bool
            True if the input is a filename and as extension .zarr.zip, False otherwise.
        """
        if isinstance(filename_or_obj, str | os.PathLike):
            # allow a trailing slash to account for an autocomplete
            # adding it.
            exts = Path(str(filename_or_obj).rstrip("/")).suffixes
            return exts[-2:] == [".zarr", ".zip"]

        return False
