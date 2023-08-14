"""Template of a typical workflow."""
import atexit
import logging
import os

import xarray as xr
from dask import config as dskconf
from dask.distributed import Client

import xscen as xs
from xscen.config import CONFIG

# Load configuration
# paths1.yml is used to add private information to your workflow, such as file paths, without running the risk of them being pushed to a public Github repo.
# For this to work as intended, 'paths1.yml' should be included in your .gitignore.
# config1.yml (or any number of those) can then contain the rest of the configuration.
# All configuration files are merged together into a single CONFIG instance when you call xs.load_config
xs.load_config(
    "paths1.yml", "config1.yml", verbose=(__name__ == "__main__"), reset=True
)

# get logger
if "logging" in CONFIG:
    logger = logging.getLogger("xscen")

# The workflow is made to be able to restart where it left off, in case of a crash or if you needed to stop it for some reason.
# To achieve this, it checks if the results of a task are already in the project catalog before doing it.
# When a task is completed, the produced data files are added to the project catalog.
# The workflow does NOT automatically remove intermediate files. You might run out of space.
if __name__ == "__main__":
    # Set dask  configuration
    daskkws = CONFIG["dask"].get("client", {})
    dskconf.set(**{k: v for k, v in CONFIG["dask"].items() if k != "client"})

    # Copy config to the top of the log file
    if "logging" in CONFIG and "file" in CONFIG["logging"]["handlers"]:
        f1 = open(CONFIG["logging"]["handlers"]["file"]["filename"], "a+")
        f2 = open("config1.yml")
        f1.write(f2.read())
        f1.close()
        f2.close()

    # Set email config
    if "scripting" in CONFIG:
        atexit.register(xs.send_mail_on_exit, subject=CONFIG["scripting"]["subject"])

    # Either load or initialize the Project Catalog
    pcat = xs.ProjectCatalog(
        CONFIG["paths"]["project_catalog"],
        project=CONFIG["project"],
        create=True,
        overwrite=False,
    )

    # Set some useful recurrent variables
    if CONFIG.get("to_dataset_dict", False):
        tdd = CONFIG["to_dataset_dict"]

    # --- EXTRACT---
    # Check if that step is in list of tasks before doing it
    if "extract" in CONFIG["tasks"]:
        # Iterate on types to extract (reconstruction, simulation) and get the dictionary for this type of data from the config
        for source_type, type_dict in CONFIG["extract"].items():
            # Filter the catalog to get only the datasets that match the arguments in the config.
            # Arguments are not passed automatically, because the config is different for each type of data.
            # Therefore, we must manually send the 'type_dict' entry to the search_data_catalogs function.
            cat = xs.search_data_catalogs(**type_dict["search_data_catalogs"])

            # Iterate over the datasets that matched the search
            # 'ds_id' is the ID of the dataset, 'dc' is the sub-catalog for this dataset
            for ds_id, dc in cat.items():
                # These are some relevant attributes that are used to check if the task was done already and to write the output path.
                cur = {
                    "id": ds_id,
                    "xrfreq": "D",
                    "processing_level": "extracted",
                }
                # We use the attributes to check if the dataset is already extracted and exists in the ProjectCatalog.
                if not pcat.exists_in_cat(**cur):
                    # Set up the dask client with fine-tuned parameters for this task, and measure time
                    with (
                        Client(**type_dict["dask"], **daskkws),
                        xs.measure_time(name=f"extract {cur}", logger=logger),
                    ):
                        # Extract and create a dictionary of datasets from the sub-catalog, with the requested domain, period, and frequency.
                        ds_dict = xs.extract_dataset(
                            catalog=dc,
                            # Once again, we manually pass the arguments from the config to the function.
                            **type_dict["extract_dataset"],
                        )

                        # Iterate over the different frequencies
                        for key_freq, ds in ds_dict.items():
                            # For future steps in the workflow, we can save a lot of time by stacking the spatial coordinates into a single dimension
                            # and dropping the NaNs. This is especially useful for big datasets.
                            if type_dict.get("stack_drop_nans", False):
                                ds = xs.utils.stack_drop_nans(
                                    ds,
                                    ds[list(ds.data_vars)[0]]
                                    .isel(time=0, drop=True)
                                    .notnull(),
                                )
                            # Prepare the filename for the zarr file, using the format specified in paths1.yml
                            path = CONFIG["paths"]["task"].format(**cur)
                            # Save to zarr
                            # Once again, we manually pass the arguments from the config to the function, as they will differ for each task.
                            xs.save_to_zarr(ds=ds, filename=path, **type_dict["save"])
                            pcat.update_from_ds(ds=ds, path=path)

    # The next steps follow the same pattern as the previous one, so we will not comment them in detail.
    # Typically, you'll always have to:
    # 1) search the ProjectCatalog for the relevant datasets
    # 2) iterate over the datasets
    # 3) check if the processed dataset is already in the ProjectCatalog
    # 4) set up the dask client
    # 5) call the function that does the processing
    #    - with arguments automatically passed from the config if consistent for all datasets
    #    - or with arguments manually passed if they differ
    # 6) save the processed dataset to zarr
    # 7) update the ProjectCatalog

    # --- REGRID ---
    if "regrid" in CONFIG["tasks"]:
        # Search the ProjectCatalog for the results of the previous step, then iterate over each dataset.
        # We usually don't have to rely on search_data_catalogs anymore after the initial extraction, because the content of the ProjectCatalog is smaller and more manageable.
        # In most cases, we can just use the search function with the 'type' and 'processing_level' attributes.
        input_dict = pcat.search(**CONFIG["regrid"]["inputs"]).to_dataset_dict(**tdd)
        for key_input, ds_input in input_dict.items():
            cur = {
                "id": ds_input.attrs["cat:id"],
                "xrfreq": ds_input.attrs["cat:xrfreq"],
                "processing_level": "regridded",
            }
            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["regrid"]["dask"], **daskkws),
                    xs.measure_time(name=f"{cur}", logger=logger),
                ):
                    # Get the output grid
                    ds_grid = pcat.search(**CONFIG["regrid"]["output"]).to_dataset(
                        **tdd
                    )

                    # Perform the regridding
                    # Most arguments are passed automatically from the config, but we still need to manually pass the input and the grid.
                    ds_regrid = xs.regrid_dataset(
                        ds=ds_input,
                        ds_grid=ds_grid,
                    )

                # Save to zarr
                path = f"{CONFIG['paths']['task']}".format(**cur)
                xs.save_to_zarr(ds=ds_regrid, filename=path, **CONFIG["regrid"]["save"])
                pcat.update_from_ds(ds=ds_regrid, path=path, info_dict=cur)

    # --- BIAS ADJUST ---
    if "biasadjust" in CONFIG["tasks"]:
        # Bias adjustment differs for each variable, so we need to iterate over them.
        for var, ba_dict in CONFIG["biasadjust"].items():
            # Search the ProjectCatalog for the results of the previous step, then iterate over each dataset.
            dict_sim = pcat.search(**ba_dict["sim_inputs"]).to_dataset_dict(**tdd)
            for id_sim, ds_sim in dict_sim.items():
                cur = {
                    "id": ds_sim.attrs["cat:id"],
                    "xrfreq": ds_sim.attrs["cat:xrfreq"],
                    "processing_level": "biasadjusted",
                    "variable": var,
                }
                if not pcat.exists_in_cat(**cur):
                    with (
                        Client(**ba_dict["dask"], **daskkws),
                        xs.measure_time(name=f" {cur}", logger=logger),
                    ):
                        # Find the reference
                        ds_ref = pcat.search(**ba_dict["ref_input"]).to_dataset(**tdd)
                        # Training step
                        ds_tr = xs.train(
                            dref=ds_ref,
                            dhist=ds_sim,
                            var=[var],
                            **ba_dict["xscen_train"],
                        )

                        # You might need to save ds_tr, if your data is too big, to avoid overloading dask.

                        # Adjusting step
                        ds_scen = xs.adjust(
                            dsim=ds_sim,
                            dtrain=ds_tr,
                            to_level="biasadjusted",
                            **ba_dict["xscen_adjust"],
                        )

                        # Save to zarr
                        path = f"{CONFIG['paths']['ba']}".format(**cur)
                        xs.save_to_zarr(ds=ds_scen, filename=path, **ba_dict["save"])
                        pcat.update_from_ds(ds=ds_scen, path=path)

    # --- CLEAN UP ---
    # This step potentially does a lot of different things.
    # In this example, we will:
    # - Unstack the spatial coordinates and chunk them.
    # - Change units to degC and mm/day.
    # - Convert the data to a standard calendar and fill the missing values.
    if "cleanup" in CONFIG["tasks"]:
        # In the previous step, we bias adjusted tasmax, dtr, and pr.
        # Here, we can use search_data_catalogs with ['tasmin', 'tasmax', 'pr'] to instruct intake-esm to compute 'tasmin' from 'tasmax' and 'dtr'.
        # You can find more details on DerivedVariables here: https://xscen.readthedocs.io/en/latest/notebooks/1_catalog.html#Derived-variables
        cu_cats = xs.search_data_catalogs(**CONFIG["cleanup"]["search_data_catalogs"])
        for cu_id, cu_cat in cu_cats.items():
            cur = {
                "id": cu_id,
                "xrfreq": "D",
                "processing_level": "cleaned",
            }
            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["cleanup"]["dask"], **daskkws),
                    xs.measure_time(name=f"{cur}", logger=logger),
                ):
                    # Alongside search_data_catalogs, extract_dataset will compute 'tasmin' and put the requested variables back together in one dataset.
                    freq_dict = xs.extract_dataset(catalog=cu_cat)

                    # Iterate over the frequencies (usually just 'D')
                    for key_freq, ds in freq_dict.items():
                        # Clean up the dataset
                        ds_clean = xs.clean_up(
                            ds=ds,
                            **CONFIG["cleanup"]["xscen_clean_up"],
                        )

                        # Save to zarr.
                        path = f"{CONFIG['paths']['task']}".format(**cur)
                        xs.save_to_zarr(ds_clean, path, **CONFIG["cleanup"]["save"])
                        pcat.update_from_ds(ds=ds_clean, path=path)

    # --- RECHUNK and store final daily data ---
    if "rechunk" in CONFIG["tasks"]:
        # Search the ProjectCatalog for the results of the previous step, then iterate over each dataset.
        dict_input = pcat.search(**CONFIG["rechunk"]["inputs"]).to_dataset_dict(**tdd)
        for key_input, ds_input in dict_input.items():
            cur = {
                "id": ds_input.attrs["cat:id"],
                "xrfreq": ds_input.attrs["cat:xrfreq"],
                "domain": ds_input.attrs["cat:domain"],
                "processing_level": "final",
            }
            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["rechunk"]["dask"], **daskkws),
                    xs.measure_time(name=f"rechunk {cur}", logger=logger),
                ):
                    # Final path for the data
                    path_out = f"{CONFIG['paths']['task']}".format(**cur)

                    # Rechunk and move the data to its final location
                    xs.io.rechunk(
                        path_in=ds_input.attrs["cat:path"],
                        path_out=path_out,
                        **CONFIG["rechunk"]["xscen_rechunk"],
                    )

                    # xscen.rechunk saves the data to zarr, so we need to open it again to update the ProjectCatalog.
                    ds = xr.open_zarr(path_out)
                    pcat.update_from_ds(
                        ds=ds,
                        path=str(path_out),
                        info_dict={"processing_level": "final"},
                    )

    # --- DIAGNOSTICS ---
    if "diagnostics" in CONFIG["tasks"]:
        # The properties and measures that we want to compute are different for each type of data (ref, sim, scen), so we need to iterate over them.
        for kind, kind_dict in CONFIG["diagnostics"]["kind"].items():
            # Search for the right datasets and iterate over them
            dict_input = pcat.search(**kind_dict["inputs"]).to_dataset_dict(**tdd)
            for key_input, ds_input in dict_input.items():
                cur = {
                    "id": ds_input.attrs["cat:id"],
                    "processing_level": kind_dict["properties_and_measures"].get(
                        "to_level_prop", "diag-properties"
                    ),
                    "xrfreq": "fx",
                }

                if not pcat.exists_in_cat(**cur):
                    with (
                        Client(**CONFIG["diagnostics"]["dask"], **daskkws),
                        xs.measure_time(name=f"{cur}", logger=logger),
                    ):
                        # Find the reference required for the measures
                        dref_for_measure = None
                        if "dref_for_measure" in kind_dict:
                            dref_for_measure = pcat.search(
                                **kind_dict["dref_for_measure"],
                            ).to_dataset(**tdd)

                        # Compute the properties and measures
                        prop, meas = xs.properties_and_measures(
                            ds=ds_input,
                            dref_for_measure=dref_for_measure,
                            **kind_dict["properties_and_measures"],
                        )

                        # Save to zarr
                        for out in [meas, prop]:
                            cur["processing_level"] = out.attrs["cat:processing_level"]
                            # Don't save if it's empty
                            if len(out.data_vars) > 0:
                                path_diag = f"{CONFIG['paths']['task']}".format(**cur)
                                xs.save_to_zarr(out, path_diag, **kind_dict["save"])
                                pcat.update_from_ds(ds=out, path=path_diag)

        # Create a summary of diagnostics
        # Search for the measures and iterate over them
        meas_dict = pcat.search(processing_level="diag-measures-sim").to_dataset_dict(
            **tdd
        )
        for id_meas, ds_meas_sim in meas_dict.items():
            cur = {
                "id": ds_meas_sim.attrs["cat:id"],
                "processing_level": "diag-improved",
                "xrfreq": "fx",
            }
            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["diagnostics"]["dask"], **daskkws),
                    xs.measure_time(name=f"summary diag {cur['id']}", logger=logger),
                ):
                    # Find the bias adjusted scenario associated with the simulation
                    meas_datasets = {}
                    meas_datasets["sim"] = ds_meas_sim
                    meas_datasets["scen"] = pcat.search(
                        processing_level="diag-measures-scen",
                        id=cur["id"],
                    ).to_dataset(**tdd)

                    # Compute heatmap
                    hm = xs.diagnostics.measures_heatmap(meas_datasets)

                    # Compute improvement map
                    ip = xs.diagnostics.measures_improvement(meas_datasets)

                    # Save to zarr
                    for ds in [hm, ip]:
                        cur["processing_level"] = ds.attrs["cat:processing_level"]
                        path_diag = f"{CONFIG['paths']['task']}".format(**cur)
                        xs.save_to_zarr(ds=ds, filename=path_diag, mode="o")
                        pcat.update_from_ds(ds=ds, path=path_diag)

    # --- INDICATORS ---
    if "indicators" in CONFIG["tasks"]:
        # Search for the right datasets and iterate over them
        dict_input = pcat.search(**CONFIG["indicators"]["inputs"]).to_dataset_dict(
            **tdd
        )
        for key_input, ds_input in dict_input.items():
            with (
                Client(**CONFIG["indicators"]["dask"], **daskkws),
                xs.measure_time(name=f"indicators {key_input}", logger=logger),
            ):
                # Compute indicators
                dict_indicator = xs.compute_indicators(
                    ds=ds_input,
                    indicators=CONFIG["indicators"]["path_yml"],
                )

                # Iterate over the different frequencies
                for key_freq, ds_ind in dict_indicator.items():
                    cur = {
                        "id": ds_input.attrs["cat:id"],
                        "xrfreq": key_freq,
                        "processing_level": "indicators",
                    }
                    if not pcat.exists_in_cat(**cur):
                        # Save to zarr
                        path_ind = f"{CONFIG['paths']['task']}".format(**cur)
                        xs.save_to_zarr(
                            ds_ind, path_ind, **CONFIG["indicators"]["save"]
                        )
                        pcat.update_from_ds(ds=ds_ind, path=path_ind)

    # --- CLIMATOLOGICAL MEAN ---
    if "climatology" in CONFIG["tasks"]:
        # Search for the right datasets and iterate over them
        ind_dict = pcat.search(**CONFIG["aggregate"]["input"]["clim"]).to_dataset_dict(
            **tdd
        )
        for key_input, ds_input in ind_dict.items():
            cur = {
                "id": ds_input.attrs["cat:id"],
                "xrfreq": ds_input.attrs["cat:xrfreq"],
                "processing_level": "climatology",
            }
            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["aggregate"]["dask"], **daskkws),
                    xs.measure_time(name=f"{cur}", logger=logger),
                ):
                    # Compute climatological mean
                    ds_mean = xs.climatological_mean(ds=ds_input)

                    # Save to zarr
                    path = f"{CONFIG['paths']['task']}".format(**cur)
                    xs.save_to_zarr(ds_mean, path, **CONFIG["aggregate"]["save"])
                    pcat.update_from_ds(ds=ds_mean, path=path)

    # --- DELTAS ---
    if "delta" in CONFIG["tasks"]:
        # Search for the right datasets and iterate over them
        ind_dict = pcat.search(**CONFIG["aggregate"]["input"]["delta"]).to_dataset_dict(
            **tdd
        )
        for key_input, ds_input in ind_dict.items():
            cur = {
                "id": ds_input.attrs["cat:id"],
                "xrfreq": ds_input.attrs["cat:xrfreq"],
                "processing_level": "delta",
            }
            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["aggregate"]["dask"], **daskkws),
                    xs.measure_time(name=f"{cur}", logger=logger),
                ):
                    # Compute deltas
                    ds_delta = xs.aggregate.compute_deltas(ds=ds_input)

                    # Save to zarr
                    path = f"{CONFIG['paths']['task']}".format(**cur)
                    xs.save_to_zarr(ds_delta, path, **CONFIG["aggregate"]["save"])
                    pcat.update_from_ds(ds=ds_delta, path=path)

    # --- ENSEMBLES ---
    if "ensembles" in CONFIG["tasks"]:
        # We want to create one ensemble for each processing level, experiment, and xrfreq.
        for processing_level in CONFIG["ensembles"]["processing_levels"]:
            ind_df = pcat.search(processing_level=processing_level).df
            # Iterate through available xrfreq, experiment and variables
            for experiment in ind_df.experiment.unique():
                for xrfreq in ind_df.xrfreq.unique():
                    # Gather all the datasets that match the arguments into a dictionary
                    ind_dict = pcat.search(
                        processing_level=processing_level,
                        experiment=experiment,
                        xrfreq=xrfreq,
                    ).to_dataset_dict(**tdd)

                    cur = {
                        "processing_level": f"ensemble-{processing_level}",
                        "experiment": experiment,
                        "xrfreq": xrfreq,
                    }
                    if not pcat.exists_in_cat(**cur):
                        with (
                            Client(**CONFIG["ensembles"]["dask"], **daskkws),
                            xs.measure_time(name=f"{cur}", logger=logger),
                        ):
                            ens_stats = xs.ensemble_stats(
                                datasets=ind_dict,
                                to_level=f"ensemble-{processing_level}",
                            )

                            # Add the new ID
                            cur["id"] = ens_stats.attrs["cat:id"]

                            # Save to zarr
                            path = f"{CONFIG['paths']['task']}".format(**cur)
                            xs.save_to_zarr(
                                ens_stats, path, **CONFIG["ensembles"]["save"]
                            )
                            pcat.update_from_ds(ds=ens_stats, path=path)

    xs.send_mail(
        subject="Template 1 - Success",
        msg="Congratulations! All tasks of the workflow are done!",
    )
