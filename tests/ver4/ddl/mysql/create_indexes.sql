

ALTER TABLE applause_dr4.archive ADD CONSTRAINT archive_pky PRIMARY KEY (archive_id);

ALTER TABLE applause_dr4.plate ADD CONSTRAINT plate_pky PRIMARY KEY (plate_id); ALTER TABLE applause_dr4.plate MODIFY plate_id INT UNSIGNED NOT NULL AUTO_INCREMENT;
CREATE INDEX plate_archive_ind ON applause_dr4.plate (archive_id);
CREATE INDEX plate_wfpdb_ind ON applause_dr4.plate (wfpdb_id);
CREATE INDEX plate_method_ind ON applause_dr4.plate (method_code);

ALTER TABLE applause_dr4.exposure ADD CONSTRAINT exposure_pky PRIMARY KEY (exposure_id);ALTER TABLE applause_dr4.exposure MODIFY exposure_id INT UNSIGNED NOT NULL AUTO_INCREMENT;
CREATE INDEX exposure_plate_ind ON applause_dr4.exposure (plate_id);
CREATE INDEX exposure_archive_ind ON applause_dr4.exposure (archive_id);
CREATE INDEX exposure_ra_icrs_ind ON applause_dr4.exposure (ra_icrs);
CREATE INDEX exposure_dec_icrs_ind ON applause_dr4.exposure (dec_icrs);

ALTER TABLE applause_dr4.exposure_sub ADD CONSTRAINT subexposure_pky PRIMARY KEY (subexposure_id);ALTER TABLE applause_dr4.exposure_sub MODIFY subexposure_id INT UNSIGNED NOT NULL AUTO_INCREMENT;
CREATE INDEX exposure_sub_exposure_ind ON applause_dr4.exposure_sub (exposure_id);
CREATE INDEX exposure_sub_plate_ind ON applause_dr4.exposure_sub (plate_id);

ALTER TABLE applause_dr4.scan ADD CONSTRAINT scan_pky PRIMARY KEY (scan_id);ALTER TABLE applause_dr4.scan MODIFY scan_id INT UNSIGNED NOT NULL AUTO_INCREMENT;
CREATE INDEX scan_plate_ind ON applause_dr4.scan (plate_id);
CREATE INDEX scan_archive_ind ON applause_dr4.scan (archive_id);

ALTER TABLE applause_dr4.preview ADD CONSTRAINT preview_pky PRIMARY KEY (preview_id);ALTER TABLE applause_dr4.preview MODIFY preview_id INT UNSIGNED NOT NULL AUTO_INCREMENT;
CREATE INDEX preview_archive_ind ON applause_dr4.preview (archive_id);

ALTER TABLE applause_dr4.logbook ADD CONSTRAINT logbook_pky PRIMARY KEY (logbook_id);ALTER TABLE applause_dr4.logbook MODIFY logbook_id INT UNSIGNED NOT NULL AUTO_INCREMENT;
CREATE INDEX logbook_archive_ind ON applause_dr4.logbook (archive_id);

ALTER TABLE applause_dr4.logpage ADD CONSTRAINT logpage_pky PRIMARY KEY (logpage_id);ALTER TABLE applause_dr4.logpage MODIFY logpage_id INT UNSIGNED NOT NULL AUTO_INCREMENT;
CREATE INDEX logpage_archive_ind ON applause_dr4.logpage (archive_id);

CREATE INDEX plate_logpage_plate_ind ON applause_dr4.plate_logpage (plate_id);
CREATE INDEX plate_logpage_logpage_ind ON applause_dr4.plate_logpage (logpage_id);

ALTER TABLE applause_dr4.source ADD CONSTRAINT source_pky PRIMARY KEY (source_id);
CREATE INDEX source_process_ind ON applause_dr4.source (process_id);
CREATE INDEX source_scan_ind ON applause_dr4.source (scan_id);
CREATE INDEX source_exposure_ind ON applause_dr4.source (exposure_id);
CREATE INDEX source_plate_ind ON applause_dr4.source (plate_id);
CREATE INDEX source_archive_ind ON applause_dr4.source (archive_id);
CREATE INDEX source_annularbin_ind ON applause_dr4.source (annular_bin);

ALTER TABLE applause_dr4.source_calib ADD CONSTRAINT source_calib_pky PRIMARY KEY (source_id);
CREATE INDEX source_calib_source_ind ON applause_dr4.source_calib (source_id);
CREATE INDEX source_calib_process_ind ON applause_dr4.source_calib (process_id);
CREATE INDEX source_calib_scan_ind ON applause_dr4.source_calib (scan_id);
CREATE INDEX source_calib_plate_ind ON applause_dr4.source_calib (plate_id);
CREATE INDEX source_calib_archive_ind ON applause_dr4.source_calib (archive_id);
CREATE INDEX source_calib_annularbin_ind ON applause_dr4.source_calib (annular_bin);
CREATE INDEX source_calib_ra_icrs_ind ON applause_dr4.source_calib (ra_icrs);
CREATE INDEX source_calib_dec_icrs_ind ON applause_dr4.source_calib (dec_icrs);
CREATE INDEX source_calib_x_ind ON applause_dr4.source_calib (x_sphere);
CREATE INDEX source_calib_y_ind ON applause_dr4.source_calib (y_sphere);
CREATE INDEX source_calib_z_ind ON applause_dr4.source_calib (z_sphere);
CREATE INDEX source_calib_healpix256_ind ON applause_dr4.source_calib (healpix256);
CREATE INDEX source_calib_nndist_ind ON applause_dr4.source_calib (nn_dist);
CREATE INDEX source_calib_natmag_ind ON applause_dr4.source_calib (natmag);
CREATE INDEX source_calib_bmag_ind ON applause_dr4.source_calib (bpmag);
CREATE INDEX source_calib_vmag_ind ON applause_dr4.source_calib (rpmag);
CREATE INDEX source_calib_gaiaedr3_ind ON applause_dr4.source_calib (gaiaedr3_id);

ALTER TABLE applause_dr4.source_xmatch ADD CONSTRAINT source_xmatch_pky PRIMARY KEY (source_id, gaiaedr3_id);
CREATE INDEX source_xmatch_source_ind ON applause_dr4.source_xmatch (source_id);
CREATE INDEX source_xmatch_process_ind ON applause_dr4.source_xmatch (process_id);
CREATE INDEX source_xmatch_scan_ind ON applause_dr4.source_xmatch (scan_id);
CREATE INDEX source_xmatch_plate_ind ON applause_dr4.source_xmatch (plate_id);
CREATE INDEX source_xmatch_archive_ind ON applause_dr4.source_xmatch (archive_id);
CREATE INDEX source_xmatch_gaiaedr3_id_ind ON applause_dr4.source_xmatch (gaiaedr3_id);

ALTER TABLE applause_dr4.solution_set ADD CONSTRAINT solutionset_pky PRIMARY KEY (solutionset_id);ALTER TABLE applause_dr4.solution_set MODIFY solutionset_id INT UNSIGNED NOT NULL AUTO_INCREMENT;
CREATE INDEX solution_set_process_ind ON applause_dr4.solution_set (process_id);
CREATE INDEX solution_set_scan_ind ON applause_dr4.solution_set (scan_id);
CREATE INDEX solution_set_plate_ind ON applause_dr4.solution_set (plate_id);
CREATE INDEX solution_set_archive_ind ON applause_dr4.solution_set (archive_id);

ALTER TABLE applause_dr4.solution ADD CONSTRAINT solution_pky PRIMARY KEY (solution_id);ALTER TABLE applause_dr4.solution MODIFY solution_id INT UNSIGNED NOT NULL AUTO_INCREMENT;
CREATE INDEX solution_solutionset_ind ON applause_dr4.solution (solutionset_id);
CREATE INDEX solution_process_ind ON applause_dr4.solution (process_id);
CREATE INDEX solution_scan_ind ON applause_dr4.solution (scan_id);
CREATE INDEX solution_plate_ind ON applause_dr4.solution (plate_id);
CREATE INDEX solution_archive_ind ON applause_dr4.solution (archive_id);
CREATE INDEX solution_ra_icrs_ind ON applause_dr4.solution (ra_icrs);
CREATE INDEX solution_dec_icrs_ind ON applause_dr4.solution (dec_icrs);

CREATE INDEX solution_healpix_solution_ind ON applause_dr4.solution_healpix (solution_id);
CREATE INDEX solution_healpix_solutionset_ind ON applause_dr4.solution_healpix (solutionset_id);
CREATE INDEX solution_healpix_process_ind ON applause_dr4.solution_healpix (process_id);
CREATE INDEX solution_healpix_scan_ind ON applause_dr4.solution_healpix (scan_id);
CREATE INDEX solution_healpix_plate_ind ON applause_dr4.solution_healpix (plate_id);
CREATE INDEX solution_healpix_archive_ind ON applause_dr4.solution_healpix (archive_id);
CREATE INDEX solution_healpix_healpix_ind ON applause_dr4.solution_healpix (healpix1024);

ALTER TABLE applause_dr4.scanner_pattern ADD CONSTRAINT pattern_pky PRIMARY KEY (pattern_id);ALTER TABLE applause_dr4.scanner_pattern MODIFY pattern_id INT UNSIGNED NOT NULL AUTO_INCREMENT;
CREATE INDEX scanner_pattern_process_ind ON applause_dr4.scanner_pattern (process_id);
CREATE INDEX scanner_pattern_scan_ind ON applause_dr4.scanner_pattern (scan_id);
CREATE INDEX scanner_pattern_plate_ind ON applause_dr4.scanner_pattern (plate_id);
CREATE INDEX scanner_pattern_archive_ind ON applause_dr4.scanner_pattern (archive_id);

ALTER TABLE applause_dr4.phot_calib ADD CONSTRAINT calib_pky PRIMARY KEY (calib_id);ALTER TABLE applause_dr4.phot_calib MODIFY calib_id INT UNSIGNED NOT NULL AUTO_INCREMENT;
CREATE INDEX phot_calib_process_ind ON applause_dr4.phot_calib (process_id);
CREATE INDEX phot_calib_scan_ind ON applause_dr4.phot_calib (scan_id);
CREATE INDEX phot_calib_plate_ind ON applause_dr4.phot_calib (plate_id);
CREATE INDEX phot_calib_archive_ind ON applause_dr4.phot_calib (archive_id);

ALTER TABLE applause_dr4.phot_calib_curve ADD CONSTRAINT calibcurve_pky PRIMARY KEY (calibcurve_id);ALTER TABLE applause_dr4.phot_calib_curve MODIFY calibcurve_id INT UNSIGNED NOT NULL AUTO_INCREMENT;
CREATE INDEX phot_calib_curve_calib_ind ON applause_dr4.phot_calib_curve (calib_id);
CREATE INDEX phot_calib_curve_process_ind ON applause_dr4.phot_calib_curve (process_id);
CREATE INDEX phot_calib_curve_scan_ind ON applause_dr4.phot_calib_curve (scan_id);
CREATE INDEX phot_calib_curve_plate_ind ON applause_dr4.phot_calib_curve (plate_id);
CREATE INDEX phot_calib_curve_archive_ind ON applause_dr4.phot_calib_curve (archive_id);

ALTER TABLE applause_dr4.process ADD CONSTRAINT process_pky PRIMARY KEY (process_id);ALTER TABLE applause_dr4.process MODIFY process_id INT UNSIGNED NOT NULL AUTO_INCREMENT;
CREATE INDEX process_scan_ind ON applause_dr4.process (scan_id);
CREATE INDEX process_plate_ind ON applause_dr4.process (plate_id);
CREATE INDEX process_archive_ind ON applause_dr4.process (archive_id);
CREATE INDEX process_filename_ind ON applause_dr4.process (filename);

ALTER TABLE applause_dr4.process_log ADD CONSTRAINT processlog_pky PRIMARY KEY (processlog_id);ALTER TABLE applause_dr4.process_log MODIFY processlog_id INT UNSIGNED NOT NULL AUTO_INCREMENT;
CREATE INDEX process_log_process_ind ON applause_dr4.process_log (process_id);
CREATE INDEX process_log_scan_ind ON applause_dr4.process_log (scan_id);
CREATE INDEX process_log_plate_ind ON applause_dr4.process_log (plate_id);
CREATE INDEX process_log_archive_ind ON applause_dr4.process_log (archive_id);
CREATE INDEX process_log_event_ind ON applause_dr4.process_log (event);