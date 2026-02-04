API
===

Catalog
-------

.. automodule:: xscen.catalog
   :members:
   :noindex:

.. automodule:: xscen.catutils
   :members:
   :noindex:

Extraction
----------

.. automodule:: xscen.extract
   :members:
   :noindex:

Regridding
----------

.. automodule:: xscen.regrid
   :members:
   :noindex:

Bias Adjustment
---------------

.. automodule:: xscen.biasadjust
   :members: adjust train
   :noindex:

Indicators
----------

.. automodule:: xscen.indicators
   :members:
   :noindex:

Ensembles
---------

.. automodule:: xscen.ensembles
   :members:
   :noindex:

Aggregation
-----------

.. automodule:: xscen.aggregate
   :members:
   :noindex:

Diagnostics and Quality Checks
------------------------------

.. automodule:: xscen.diagnostics
   :members:
   :noindex:

Input / Output
--------------

.. automodule:: xscen.io
   :members:
   :noindex:

Spatial tools
-------------

.. automodule:: xscen.spatial
   :members:
   :noindex:

.. py:data:: Region

   A region specification, a dictionary with the following valid entries:

   - ``name`` : region name for the `domain` column
   - ``method`` : the method used for subsetting the region from a dataset:

     - ``"shape"`` : The region is defined by a polygon in entry "shape".
     - ``"bbox"`` : The region is defined by bounds in entries "lat_bnds" and "lon_bnds".
     - ``"sel"`` : The region is defined by bounds in entries with the same name as the spatial coordinates.
     - ``"gridpoint"`` : The region is a list of points defined in entries "lon" and "lat"

   - ``tile_buffer`` : For methods "bbox" and "shape", the actual subset is enlarged by this many grid cell sizes (approximated). This differs from clisops' ``buffer`` argument in :py:func:`~clisops.core.subset.subset_shape`..
   - ``shape`` : For method "shape" only. A shapely Polygon, geopandas GeoDataFrame or GeoSeries, or a path to a file that GeoPandas can open. When multiplie shapes are present, the subsetting is done with the unary union of the geometries. When the CRS information is missing, EPSG 4326 is assumed.
   - ``lat_bnds`` and ``lon_bnds`` : For method "bbox" only. Tuples of the minimum and maximum latitude or longitude values.
   - ``lon`` and ``lat`` : For method "gridpoint", 1D arrays of the longitude and latitude coordinates of the points to extract.
   - ``<coordinate name>`` : For method "sel", any other entry is understood as a tuple of the bounds to extract for that spatial dimension.


Controlled Vocabulary and Mappings
----------------------------------

.. automodule:: xscen.utils.CV
   :members:
   :noindex:

Configuration Utilities
-----------------------

.. automodule:: xscen.config
   :members:
   :noindex:

Script Utilities
----------------

.. automodule:: xscen.scripting
   :members:
   :noindex:

Packaging Utilities
-------------------

.. automodule:: xscen.utils
   :members:
   :noindex:
