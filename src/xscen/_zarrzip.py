"""Zipped Zarr support for Zarr 3."""

from pathlib import Path

from zarr.storage import ZipStore, _common


old_make_store = _common.make_store


async def make_store(
    store_like,
    *,
    mode=None,
    storage_options=None,
):
    """Drop-in replacement for zarr.storage._common.make_store that opens zip paths directly in ZipStores."""  # numpydoc ignore=PR01,RT01
    if isinstance(store_like, Path) and store_like.suffix == ".zip":
        print("using shortcut")
        return await ZipStore.open(path=store_like, mode=mode, read_only=(mode == "r"))
    return await old_make_store(store_like, mode=mode, storage_options=storage_options)


_common.__dict__["make_store"] = make_store
