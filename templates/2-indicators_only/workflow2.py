"""Climate indices calculator with a minimal cli interface."""

import atexit
import logging
from argparse import ArgumentParser

import dask
from dask.distributed import Client

from xscen.catalog import ProjectCatalog
from xscen.config import CONFIG, load_config
from xscen.extract import extract_dataset, search_data_catalogs
from xscen.indicators import compute_indicators, load_xclim_module
from xscen.io import save_to_zarr
from xscen.scripting import send_mail_on_exit
from xscen.utils import get_cat_attrs


logger = logging.getLogger("workflow")


if __name__ == "__main__":
    # The config file is passed through the command line,
    # allowing to reuse this script for multiple datasets by having versions of the config, instead of copies of the script.
    parser = ArgumentParser(description="Compute a series of xclim indicators with xscen.")
    parser.add_argument("-c", "--conf", action="append")
    args = parser.parse_args()

    load_config(*args.conf, verbose=True)

    dask.config.set({k: v for k, v in CONFIG["dask"].items() if not k.startswith("client")})
    client = Client(**CONFIG["dask"]["client"])

    atexit.register(send_mail_on_exit)

    logger.info("Reading catalog and indicators.")
    pcat = ProjectCatalog(CONFIG["main"]["catalog"], create=True)
    mod = load_xclim_module(CONFIG["indicators"]["module"])

    # All arguments passed in the config
    cat = search_data_catalogs()

    for dsid, scat in cat.items():
        ds = extract_dataset(scat)["D"]
        to_compute = []  # A list of (name, indicator) tuples of those not already computed
        for name, ind in mod.iter_indicators():
            # Get the frequency and variable names to check if they are already computed
            outfreq = ind.injected_parameters["freq"].replace("YS", "AS-JAN")
            outnames = [cfatt["var_name"] for cfatt in ind.cf_attrs]
            if not pcat.exists_in_cat(
                id=dsid,
                variable=outnames,
                xrfreq=outfreq,
                processing_level="indicators",
            ):
                to_compute.append((name, ind))

        if not to_compute:
            msg = f"Everything computed for {dsid}."
            logger.info(msg)
            continue

        outd = compute_indicators(ds, indicators=to_compute, to_level="indicators")

        for outds in outd.values():
            outpath = CONFIG["main"]["outfilename"].format(**get_cat_attrs(outds))
            save_to_zarr(outds, outpath)
            pcat.update_from_ds(outds, path=outpath)
