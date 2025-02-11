===========
Basic Usage
===========

``xsdba`` performs a training on a source dataset `hist` to reproduce a target `ref`. "ref" is short for reference, and "hist" for historical: In climate services, it is common to refer to the training data as historical, a period of time where observations are available. The training is then applied to a dataset `sim` to obtain an adjusted dataset. This may simply be the training data `hist` which is not adjusted in the first training part. An example with a basic Quantile Mapping is given below.

.. code-block:: python

    import xsdba

    # Example: Using a Quantile Mapping method
    # Instantiate a TrainAdjust class with the training, with the source `hist` mapped to a target `ref`
    ADJ = xsdba.EmpiricalQuantileMapping.train(ref=ref, hist=hist)
    # Perform adjust for data outside the training period, `sim`
    adj = ADJ.adjust(sim=sim)
..


Units handling
--------------

``xsdba`` implements some unit handling through ``pint``. High level adjustment objects will (usually) parse units from the passed xarray objects, check that the different inputs have matching units and perform conversions when units are compatible but don't match. ``xsdba`` imports `cf-xarray's unit registry <https://cf-xarray.readthedocs.io/en/latest/units.html>`_ by default and, as such, expects units matching the `CF conventions <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#units>`_. This is very similar to ``xclim``, except for two features:

    - ``xclim`` is able to detect more complex conversions based on the CF "standard name". ``xsdba`` will not implement this, the user is expected to provide coherent inputs to the adjustment objects.
    - ``xclim`` has a convenience "hydro" context that allows for conversion between rates and fluxes, thicknesses and amounts of liquid water quantities by using an implicit the water density (1000 kg/m³).

The context of that last point can still be used if ``xclim`` is imported beside ``xsdba`` :

.. code-block:: python

    import xsdba
    import xclim  # importing xclim makes xsdba use xclim's unit registry

    pr_ref  # reference precipitation data in kg m-2 s-1
    pr_hist  # historical precipitation data in mm/d
    # In normal xsdba, the two quantities are not compatible.
    # But with xclim's hydro context, an implicit density allows for conversion

    with xclim.core.units.units.context("hydro"):
        QDM = xsdba.QuantileDeltaMapping.train(ref=ref, hist=hist)

..


Under the hood, ``xsdba`` picks whatever unit registry has been declared the "application registry" (`see pint's doc <https://pint.readthedocs.io/en/stable/api/base.html#pint.get_application_registry>`_). However, it expects some features as declared in ``cf-xarray``, so a compatible registry (as xclim's) must be used.
