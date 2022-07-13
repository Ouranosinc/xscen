=======
History
=======

v0.2.1
------------------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`).

Announcements
^^^^^^^^^^^^^
* N/A

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* `parse_directory`: Fixes to `xr_open_kwargs` and support for wildcards (*) in the directories (:pull:`19`).

Breaking changes
^^^^^^^^^^^^^^^^
* N/A

Internal changes
^^^^^^^^^^^^^^^^
* N/A


v0.2.0 (first official release)
------------------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Pascal Bourgault (:user:`aulemahal`), Trevor James Smith (:user:`Zeitsperre`), Juliette Lavoie (:user:`juliettelavoie`).

Announcements
^^^^^^^^^^^^^
* This is the first official release for xscen!

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Supports workflows with YAML configuration files for better transparency, reproducibility, and long-term backups.
* Intake_esm-based catalog to find and manage climate data.
* Climate dataset extraction, subsetting, and temporal aggregation.
* Calculate missing variables through Intake-esm's DerivedVariableRegistry.
* Regridding with xESMF.
* Bias adjustment with xclim.

Breaking changes
^^^^^^^^^^^^^^^^
* N/A

Internal changes
^^^^^^^^^^^^^^^^
* N/A
