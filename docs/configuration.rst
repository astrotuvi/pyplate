PyPlate configuration files
===========================

File structure
--------------

PyPlate supports configuration files with structure similar to Microsoft
Windows INI files. The configuration files are read with the Python 3 
built-in ``configparser`` module.

Configuration files are divided into sections. Section names are given
in square brackets. Inside each section, there are keyword-value pairs::

    [Section]
    keyword1 = value1
    keyword2 = value2

For metadata files in CSV format, PyPlate supports filenames in section
names. This way, one can specify the structure of each CSV file.

Example (simple)
----------------

::

    [Files]
    csv_dir = /path/to/CSV/dir
    plate_csv = my_plates.csv
    scan_csv = my_scans.csv

    [my_plates.csv]
    plate_id = 1
    date_orig = 2
    tms_orig = 3
    exptime = 4
    observer = 5
    tz_orig = 6

    [my_scans.csv]
    filename = 1
    plate_id = 2
    datescan = 3
    scan_author = 4

Example (complex)
-----------------

In this example, database credentials and several directory names have been 
removed.

::

    [Archive]
    archive_id = 101
    archive_name = Lippert-Astrograph

    [Pipeline]
    read_wfpdb = False
    read_csv = True
    read_fits = False
    output_header_file = True
    output_header_fits = True
    invert_image = True
    extract_sources = True
    solve_plate = True
    output_solution_db = True
    output_wcs_file = True
    get_reference_catalogs = True
    solve_recursive = True
    calibrate_photometry = True
    improve_photometry = True
    output_calibration_db = True
    output_sources_db = True
    output_sources_csv = True
    processes = 25
    process_max_tasks = 10
    wait_start = 10

    [Solve]
    use_filter = True
    filter_path = /opt/sextractor/share/sextractor/gauss_3.0_5x5.conv
    threshold_sigma = 4
    use_psf = True
    psf_threshold_sigma = 15
    psf_model_sigma = 15
    plate_epoch = 1930
    crossmatch_nsigma = 10
    crossmatch_nlogarea = 2
    crossmatch_maxradius = 20
    astref_catalog = UCAC4
    photref_catalog = APASS

    [Database]
    use_tycho2_fits = True
    use_ucac4_db = True
    ucac4_db_host = 
    ucac4_db_user = 
    ucac4_db_name = 
    ucac4_db_passwd = 
    ucac4_db_table = UCAC4_X_APASSDR9
    use_apass_db = True
    apass_db_host = 
    apass_db_user = 
    apass_db_name = 
    apass_db_passwd = 
    apass_db_table = UCAC4_X_APASSDR9
    output_db_host = 
    output_db_user = 
    output_db_name = 
    output_db_passwd = 
    enable_db_log = True
    write_sources_csv = True

    [UCAC4_X_APASSDR9]
    ucac4_id = UCAC4
    ucac4_ra = RAJ2000
    ucac4_dec = DEJ2000
    ucac4_raerr = e_RAJ2000
    ucac4_decerr = e_DEJ2000
    ucac4_pmra = pmRA
    ucac4_pmdec = pmDE
    ucac4_mag = amag
    ucac4_magerr = e_amag
    ucac4_bmag = Bmag
    ucac4_bmagerr = e_Bmag
    ucac4_vmag = Vmag
    ucac4_vmagerr = e_Vmag
    apass_ra = RAdeg
    apass_dec = DEdeg
    apass_bmag = B
    apass_bmagerr = e_B
    apass_vmag = V
    apass_vmagerr = e_V
    healpix = HEALPix256

    [Programs]
    sextractor_path = /opt/sextractor/bin/sex
    scamp_path = /opt/scamp/bin/scamp
    psfex_path = /opt/psfex/bin/psfex
    solve_field_path = /opt/astrometry/bin/solve-field
    build_index_path = /opt/astrometry/bin/build-astrometry-index
    wcs_to_tan_path = /opt/astrometry/bin/wcs-to-tan
    xy2sky_path = /opt/wcstools/bin/xy2sky

    [Files]
    wfpdb_dir =
    wfpdb_maindata =
    wfpdb_notes =
    wfpdb_observer =
    wfpdb_quality =
    csv_dir = /path/to/CSV/dir
    plate_csv = LA_plates.csv
    scan_csv = LA_scans.csv
    logbook_csv = LA_logbooks.csv
    logpage_csv = LA_logpages.csv
    preview_csv = LA_previews.csv
    header_dir =
    tiff_dir =
    fits_dir = 
    preview_dir = 
    logpage_dir = 
    cover_dir = 
    tycho2_dir = 
    work_dir = 
    write_fits_dir = 
    write_wedge_dir = 
    write_log_dir = 
    write_header_dir = 
    write_wcs_dir = 
    write_source_dir = 
    write_db_source_dir = 
    write_db_source_calib_dir = 
    write_phot_dir =
    create_checksum = True

    [LA_plates.csv]
    csv_delimiter = ;
    csv_quotechar = "
    csv_list_delimiter = |
    plate_num = 1
    wfpdb_id = 2
    ra_orig = 5
    dec_orig = 7
    object_name = 9
    date_orig = 11
    plate_format = 13
    skycond = 14
    numexp = 15
    prism = 16
    exptime = 18
    emulsion = 19
    filter = 21
    observer = 22
    notes = 23
    fn_pre = 25
    fn_cover = 26
    fn_scan = 33
    fn_log = 29
    sky_transparency = 34
    sky_calmness = 35
    sky_sharpness = 36
    ota_name = 37
    ota_diameter = 38
    ota_aperture = 39
    ota_foclen = 40
    ota_scale = 41
    prism = 42
    prism_angle = 43
    method_code = 44
    method = 45
    plate_size1 = 46
    plate_size2 = 47

    [LA_scans.csv]
    csv_delimiter = ,
    csv_quotechar = "
    filename = 1
    plate_id = 2
    plate_num = 2
    scan_id = 3
    datescan = 4
    scan_author = 5

    [LA_previews.csv]
    csv_delimiter = ,
    filename = 1
    plate_id = 2
    preview_type = 3

    [LA_logbooks.csv]
    csv_delimiter = ,
    logbook_num = 1
    logbook_id = 2
    logbook_type = 3
    logbook_title = 4

    [LA_logpages.csv]
    csv_delimiter = ,
    filename = 1
    logpage_type = 2
    logbook_num = 3
    logpage_id = 4

    [Keyword values]
    observatory = Hamburger Sternwarte
    site_name = Hamburg-Bergedorf, Germany
    site_longitude = 10.242
    site_latitude = 53.482
    site_elevation = 41
    telescope = Lippert-Astrograph
    scanner = Epson Expression 10000XL
    scan_res1 = 2400
    scan_res2 = 2400
    pix_size1 = 10.5833
    pix_size2 = 10.5833
    origin = Hamburger Sternwarte (Universitaet Hamburg)
    detector = photographic plate
    licence = https://creativecommons.org/publicdomain/zero/1.0/


