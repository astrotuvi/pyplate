PyPlate configuration files
===========================

File structure
--------------

PyPlate reads its configuration from a `TOML <https://toml.io>`_ file::

    from pyplate.config import load_config
    cfg = load_config('/path/to/pyplate.toml')

The file is divided into sections (TOML tables) with key–value pairs::

    [Section]
    text = "value"
    number = 3
    switch = true
    names = ["a.csv", "b.csv"]

Every key is checked: a key that its section does not accept, or a value of
the wrong type, stops with an error that names the section and the key.
Section names with spaces are quoted, for example ``["Keyword values"]``.

Old configuration files in INI format are converted with::

    python -m pyplate.config.ini2toml old.conf [more.conf ...] new.toml

Several INI files are merged in order. The converter renames the keys (see
`Converting old configuration files`_), lists the settings of removed
features that it drops, and stops on any other key it does not know.

Example
-------

Configuration file::

    [Archive]
    archive_id = 7
    archive_name = "Lippert-Astrograph"

    [Files]
    csv_dir = "/path/to/CSV/dir"
    plate_csv = ["my_plates.csv"]
    scan_csv = ["my_scans.csv"]

    [CSV."my_plates.csv"]
    csv_list_delimiter = ";"
    plate_num = 1
    date_orig_start = 2
    time_orig_start = 3
    exptime = 4
    observer = 5
    tz_orig = 6

    [CSV."my_scans.csv"]
    filename_scan = 1
    plate_num = 2
    scan_date = 3
    scan_author = 4

    ["Keyword values"]
    observatory = "Hamburger Sternwarte"
    site_longitude = 10.242
    site_latitude = 53.482

Plate metadata file (my_plates.csv)::

    plate1,1956-03-11,21:11:30,600,Observer Name,Europe/Berlin
    plate2,1956-03-11,21:30:00,1800,Observer Name,UT
    plate3,1956-03-12,23:05:00,1200,Observer Name,ST

Scan metadata file (my_scans.csv)::

    plate_0001.fits,plate1,Scan Author,2018-10-12
    plate_0002.fits,plate2,Scan Author,2018-10-12
    plate_0003.fits,plate3,Scan Author,2018-10-13

Metadata
--------

The metadata keys are the column names of the database tables (see
``pyplate/database/applause_dr5.yaml``).

``[Archive]``
    ``archive_id``, ``archive_name``, ``institute``. ``archive_id`` is
    copied into every record; it is needed for writing to the database.

``[Logbook.<name>]``
    One table per logbook, with the columns of the ``logbook`` table
    (``logbook_num``, ``logbook_title``, ``logbook_type``,
    ``logbook_notes``). Without such tables, the logbooks come from the
    logbook CSV file.

``[Files]``
    The input files: ``csv_dir`` and the lists ``plate_csv``, ``scan_csv``,
    ``preview_csv``, ``logbook_csv``, ``logpage_csv``; ``wfpdb_dir`` and the
    WFPDB files ``wfpdb_maindata``, ``wfpdb_quality``, ``wfpdb_notes``,
    ``wfpdb_observer``; the image directories ``fits_dir``, ``tiff_dir``,
    ``preview_dir``, ``logpage_dir``, ``cover_dir``; the catalogue
    directories ``index_dir`` and ``gaia_dir``. Output directories:
    ``work_dir``, ``write_log_dir``, ``write_header_dir``,
    ``write_fits_dir``, ``write_wedge_dir``, ``write_wcs_dir``,
    ``write_source_dir``, ``write_phot_dir``, ``write_db_source_dir``,
    ``write_db_source_calib_dir``, ``write_db_source_xmatch_dir``,
    ``write_db_solution_healpix_dir``. ``create_checksum`` adds checksums
    to the written FITS files.

``[CSV."<file name>"]``
    How a CSV file named in ``[Files]`` is read. Each key gives the
    position (from 1) of a column in the CSV rows. The keys are the columns
    of the records the file fills:

    - plate CSV: ``plate`` and ``exposure`` columns; ``plate_num``
      identifies the plate (it may also hold the WFPDB ID). A column of both
      tables goes to the plate; the ``exposure`` sub-table
      (``[CSV."<file name>".exposure]``) maps columns to the exposures
      explicitly. Per-exposure values are separated by
      ``csv_list_delimiter``, and ``|`` separates the times of
      sub-exposures. Further keys: ``fn_cover`` and ``fn_log`` (cover and
      logbook page images), ``fn_pre`` (preview image), ``fn_scan`` (scans),
      and ``ut_start_orig``, ``ut_end_orig``, ``st_start_orig``,
      ``st_end_orig`` (start and end times given as UT or sidereal time);
    - scan CSV: ``scan`` columns, identified by ``filename_scan``, and
      ``plate_num`` or ``wfpdb_id`` of the plate;
    - preview CSV: ``preview`` columns, identified by ``filename``, and
      ``plate_num`` or ``wfpdb_id``;
    - logbook CSV: ``logbook`` columns, identified by ``logbook_num``;
    - logpage CSV: ``logpage`` columns, identified by ``filename``, and
      ``logbook_num``.

    Settings: ``csv_delimiter`` (default ``,``), ``csv_quotechar``
    (default ``"``), ``csv_list_delimiter`` (default ``,``),
    ``csv_timezone`` (time zone of the recorded times: ``UT``, ``ST`` or a
    name such as ``Europe/Berlin``; a ``tz_orig`` column wins),
    ``csv_encoding`` (default ``utf-8``).

``[WFPDB]``
    ``maindata_timezone`` (time zone of the times in the maindata file),
    ``notes_timezone`` (of the times in the notes file),
    ``exptime_minutes`` (exposure times in minutes), ``encoding`` (default
    ``utf-8``).

``[Image]``
    ``logpage_exif_timezone``, ``preview_exif_timezone``,
    ``scan_exif_timezone`` (time zones of EXIF dates), ``wedge_height`` and
    ``cut_wedge`` (TIFF conversion).

``[Metadata]``
    ``evening_date`` (the recorded dates are dates of the evening, so times
    before noon belong to the next day), ``german_transliteration`` (in FITS
    headers, write ä as ae and so on).

``["Keyword values"]``
    Values of ``plate``, ``exposure`` and ``scan`` columns that apply to all
    plates of the archive. They override the input files.

``[Lookup.<column>]``
    Replacements of input values of a column, for example::

        [Lookup.observer]
        "j. mets" = "Jaan Mets"

    The input values are matched without regard to case.

The order of precedence is: WFPDB maindata, quality, observer and notes
files, plate CSV files, scan CSV file, ``["Keyword values"]``, lookups.

FITS header
-----------

``[Header]``
    ``detector``, ``licence`` and ``acknowledgements``: written to every
    header of the archive.

``["Integer values"]``, ``["Float values"]``, ``["String values"]``
    FITS keywords with fixed values.

``["FITS keyword comments"]``
    Comments of FITS keywords, instead of the default ones.

``["FITS keyword order"]``
    ``keywords``: the list of keywords in the order they are written, with
    ``sep:<title>`` for section separators.

Processing
----------

``[Pipeline]``
    Steps and outputs of ``PlatePipeline``: ``read_wfpdb``, ``read_csv``,
    ``read_fits``, ``output_header_file``, ``output_header_fits``,
    ``invert_image``, ``extract_sources``, ``classify_artifacts``,
    ``solve_plate``, ``output_solution_db``, ``output_wcs_file``,
    ``crossmatch_gaia``, ``calibrate_photometry``, ``output_calibration_db``,
    ``output_sources_db``, ``output_sources_csv``; ``processes``,
    ``process_max_tasks``, ``wait_start`` for parallel runs.

``[Solve]``
    Source extraction: ``threshold_sigma``, ``use_filter``, ``filter_path``,
    ``use_psf``, ``psf_threshold_sigma``, ``psf_model_sigma``,
    ``min_model_sources``, ``max_model_sources``, ``circular_film``.
    Astrometry: ``plate_epoch``, ``sip``, ``skip_bright``, ``distort``,
    ``allow_force``, ``repeat_find``. Crossmatch: ``crossmatch_radius``,
    ``crossmatch_nsigma``, ``crossmatch_nlogarea``, ``crossmatch_maxradius``.

``[Catalog]``
    ``catalog`` (``Gaia``), ``protocol`` (``TAP`` or ``SQL``),
    ``table_name``; with ``SQL``, the database in ``[Catalog.connection]``.

``[Programs]``
    ``sextractor_path``, ``psfex_path``, ``scamp_path``,
    ``solve_field_path``, ``wcs_to_tan_path``.

Database
--------

``[Database]``
    ``enable_db_log`` (write the process log to the database),
    ``write_sources_csv`` and ``write_solution_healpix_csv`` (write these
    tables to CSV files instead), ``dr_num``, ``process_num_digits``,
    ``source_num_digits`` (digits of the source IDs).

``[Database.connection]``
    The results are written to this database. ``rdbms`` is ``pgsql``,
    ``mysql`` or ``sqlite``; further keys ``host``, ``port``, ``socket``,
    ``user``, ``password``, ``database``; ``file`` for SQLite; ``schema``
    (schema or database name of the tables) and ``yaml`` (schema file,
    default ``<schema>.yaml``, ``applause_dr5.yaml`` without a schema).

Converting old configuration files
----------------------------------

``ini2toml`` renames the old keys in the CSV sections and in
``[Keyword values]``:

=================  ===================
Old key            New key
=================  ===================
``plate_id``       ``plate_num``
``date_orig``      ``date_orig_start``
``tms_orig``       ``time_orig_start``
``tme_orig``       ``time_orig_end``
``coord_flag``     ``flag_coord``
``time_flag``      ``flag_time``
``jd_avg_orig``    ``jd_mid_orig``
``filename``       ``filename_scan`` (scan CSV)
``datescan``       ``scan_date``
``pix_size1/2``    ``pixel_size1/2``
``fn_wedge``       ``filename_wedge``
=================  ===================

and the other metadata keys to their column names. The section named by
``[Database] output_db`` becomes ``[Database.connection]``, and the section
named by ``[Catalog] db`` becomes ``[Catalog.connection]``. ``detector``,
``licence`` and ``fits_acknowledgements`` move from ``[Keyword values]`` to
``[Header]``. Settings of removed features are dropped: the Tycho-2, UCAC4
and APASS settings, ``use_gaia_fits``, the sub-field recursion settings,
database IDs in CSV files (``scan_id``, ``logbook_id``, ...), ``method``
(derived from ``method_code``), and keys that the old code never read
(``skycond``, ``build_index_path``, ``xy2sky_path``).
