
DROP INDEX IF EXISTS plate_archive_ind;
DROP INDEX IF EXISTS plate_wfpdb_ind;
DROP INDEX IF EXISTS plate_method_ind;

DROP INDEX IF EXISTS exposure_plate_ind;
DROP INDEX IF EXISTS exposure_archive_ind;
DROP INDEX IF EXISTS exposure_ra_icrs_ind;
DROP INDEX IF EXISTS exposure_dec_icrs_ind;

DROP INDEX IF EXISTS exposure_sub_exposure_ind;
DROP INDEX IF EXISTS exposure_sub_plate_ind;

DROP INDEX IF EXISTS scan_plate_ind;
DROP INDEX IF EXISTS scan_archive_ind;

DROP INDEX IF EXISTS preview_archive_ind;

DROP INDEX IF EXISTS logbook_archive_ind;

DROP INDEX IF EXISTS logpage_archive_ind;

DROP INDEX IF EXISTS plate_logpage_plate_ind;
DROP INDEX IF EXISTS plate_logpage_logpage_ind;

DROP INDEX IF EXISTS source_process_ind;
DROP INDEX IF EXISTS source_scan_ind;
DROP INDEX IF EXISTS source_exposure_ind;
DROP INDEX IF EXISTS source_plate_ind;
DROP INDEX IF EXISTS source_archive_ind;
DROP INDEX IF EXISTS source_annularbin_ind;

DROP INDEX IF EXISTS source_calib_source_ind;
DROP INDEX IF EXISTS source_calib_process_ind;
DROP INDEX IF EXISTS source_calib_scan_ind;
DROP INDEX IF EXISTS source_calib_plate_ind;
DROP INDEX IF EXISTS source_calib_archive_ind;
DROP INDEX IF EXISTS source_calib_annularbin_ind;
DROP INDEX IF EXISTS source_calib_ra_icrs_ind;
DROP INDEX IF EXISTS source_calib_dec_icrs_ind;
DROP INDEX IF EXISTS source_calib_x_ind;
DROP INDEX IF EXISTS source_calib_y_ind;
DROP INDEX IF EXISTS source_calib_z_ind;
DROP INDEX IF EXISTS source_calib_healpix256_ind;
DROP INDEX IF EXISTS source_calib_nndist_ind;
DROP INDEX IF EXISTS source_calib_natmag_ind;
DROP INDEX IF EXISTS source_calib_bmag_ind;
DROP INDEX IF EXISTS source_calib_vmag_ind;
DROP INDEX IF EXISTS source_calib_gaiaedr3_ind;

DROP INDEX IF EXISTS source_xmatch_source_ind;
DROP INDEX IF EXISTS source_xmatch_process_ind;
DROP INDEX IF EXISTS source_xmatch_scan_ind;
DROP INDEX IF EXISTS source_xmatch_plate_ind;
DROP INDEX IF EXISTS source_xmatch_archive_ind;
DROP INDEX IF EXISTS source_xmatch_gaiaedr3_id_ind;

DROP INDEX IF EXISTS solution_set_process_ind;
DROP INDEX IF EXISTS solution_set_scan_ind;
DROP INDEX IF EXISTS solution_set_plate_ind;
DROP INDEX IF EXISTS solution_set_archive_ind;

DROP INDEX IF EXISTS solution_solutionset_ind;
DROP INDEX IF EXISTS solution_process_ind;
DROP INDEX IF EXISTS solution_scan_ind;
DROP INDEX IF EXISTS solution_plate_ind;
DROP INDEX IF EXISTS solution_archive_ind;
DROP INDEX IF EXISTS solution_ra_icrs_ind;
DROP INDEX IF EXISTS solution_dec_icrs_ind;

DROP INDEX IF EXISTS solution_healpix_solution_ind;
DROP INDEX IF EXISTS solution_healpix_solutionset_ind;
DROP INDEX IF EXISTS solution_healpix_process_ind;
DROP INDEX IF EXISTS solution_healpix_scan_ind;
DROP INDEX IF EXISTS solution_healpix_plate_ind;
DROP INDEX IF EXISTS solution_healpix_archive_ind;
DROP INDEX IF EXISTS solution_healpix_healpix_ind;

DROP INDEX IF EXISTS scanner_pattern_process_ind;
DROP INDEX IF EXISTS scanner_pattern_scan_ind;
DROP INDEX IF EXISTS scanner_pattern_plate_ind;
DROP INDEX IF EXISTS scanner_pattern_archive_ind;

DROP INDEX IF EXISTS phot_calib_process_ind;
DROP INDEX IF EXISTS phot_calib_scan_ind;
DROP INDEX IF EXISTS phot_calib_plate_ind;
DROP INDEX IF EXISTS phot_calib_archive_ind;

DROP INDEX IF EXISTS phot_calib_curve_calib_ind;
DROP INDEX IF EXISTS phot_calib_curve_process_ind;
DROP INDEX IF EXISTS phot_calib_curve_scan_ind;
DROP INDEX IF EXISTS phot_calib_curve_plate_ind;
DROP INDEX IF EXISTS phot_calib_curve_archive_ind;

DROP INDEX IF EXISTS process_scan_ind;
DROP INDEX IF EXISTS process_plate_ind;
DROP INDEX IF EXISTS process_archive_ind;
DROP INDEX IF EXISTS process_filename_ind;

DROP INDEX IF EXISTS process_log_process_ind;
DROP INDEX IF EXISTS process_log_scan_ind;
DROP INDEX IF EXISTS process_log_plate_ind;
DROP INDEX IF EXISTS process_log_archive_ind;
DROP INDEX IF EXISTS process_log_event_ind;

