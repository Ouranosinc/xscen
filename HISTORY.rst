=======
History
=======

v0.3.0 (unreleased)
------------------------------
Contributors to this version: Gabriel Rondeau-Genesse (:user:`RondeauG`), Juliette Lavoie (:user:`juliettelavoie`) .

Announcements
^^^^^^^^^^^^^
* N/A

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New function `clean_up` (`:issue:22`, `:pull:24`)
* `parse_directory`: Support for wildcards (*) in the directories (`:pull:19`).
* New function `xscen.ensemble.ensemble stats` (':issue:3`, `:pull:28`)

Breaking changes
^^^^^^^^^^^^^^^^
* N/A

Internal changes
^^^^^^^^^^^^^^^^
* `parse_directory`: Fixes to `xr_open_kwargs` (:pull:`19`).
* Fix for indicators removing the 'time' dimension (:pull:`23`).


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
