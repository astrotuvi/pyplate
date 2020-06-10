import numpy as np
import time
import socket
import os
import csv
import astropy.units as u
from collections import OrderedDict
from astropy.time import Time
from astropy.io import fits
from .db_pgsql import DB_pgsql
from .db_yaml import fetch_ordered_tables
from ..conf import read_conf
from .._version import __version__

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

try:
    import MySQLdb
except ImportError:
    pass


# Special class for handling None (NULL) values when writing data to
# CSV files for later ingestion into the database
# http://stackoverflow.com/questions/11379300/csv-reader-behavior-with-none-and-empty-string
class csvWriter(object):
    def __init__(self, csvfile, *args, **kwargs):
        self.writer = csv.writer(csvfile, *args, **kwargs)

    def writerow(self, row):
        self.writer.writerow(['\\N' if val is None else val for val in row])

    def writerows(self, rows):
        map(self.writerow, rows)


def _get_schema_old():
    return _schema 

_schema = OrderedDict()

_schema['archive'] = OrderedDict([
    ('archive_id',       ('INT UNSIGNED NOT NULL PRIMARY KEY', None)),
    ('archive_name',     ('VARCHAR(80)', None)),
    ('institute',        ('VARCHAR(80)', None)),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ])

_schema['plate'] = OrderedDict([
    ('plate_id', ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', None)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', 'archive_id')),
    ('plate_num',        ('CHAR(10)', 'plate_num')),
    ('plate_num_orig',   ('CHAR(10)', 'plate_num_orig')),
    ('wfpdb_id',         ('CHAR(15)', 'wfpdb_id')),
    ('series',           ('VARCHAR(80)', 'series')),
    ('plate_format',     ('CHAR(7)', 'plate_format')),
    ('plate_size1',      ('FLOAT', 'plate_size1')),
    ('plate_size2',      ('FLOAT', 'plate_size2')),
    ('emulsion',         ('VARCHAR(80)', 'emulsion')),
    ('filter',           ('VARCHAR(80)', 'filter')),
    ('development',      ('VARCHAR(80)', 'development')),
    ('plate_quality',    ('VARCHAR(80)', 'plate_quality')),
    ('plate_notes',      ('VARCHAR(255)', 'plate_notes')),
    ('date_orig',        ('DATE', 'date_orig')),
    ('numexp',           ('TINYINT UNSIGNED', 'numexp')),
    ('observatory',      ('VARCHAR(80)', 'observatory')),
    ('site_name',        ('VARCHAR(80)', 'site_name')),
    ('site_longitude',   ('DOUBLE', 'site_longitude')),
    ('site_latitude',    ('DOUBLE', 'site_latitude')),
    ('site_elevation',   ('FLOAT', 'site_elevation')),
    ('telescope',        ('VARCHAR(80)', 'telescope')),
    ('ota_name',         ('VARCHAR(80)', 'ota_name')),
    ('ota_diameter',     ('FLOAT', 'ota_diameter')),
    ('ota_aperture',     ('FLOAT', 'ota_aperture')),
    ('ota_foclen',       ('FLOAT', 'ota_foclen')),
    ('ota_scale',        ('FLOAT', 'ota_scale')),
    ('instrument',       ('VARCHAR(80)', 'instrument')),
    ('method_code',      ('TINYINT UNSIGNED', 'method_code')),
    ('prism',            ('VARCHAR(80)', 'prism')),
    ('prism_angle',      ('VARCHAR(10)', 'prism_angle')),
    ('dispersion',       ('FLOAT', 'dispersion')),
    ('grating',          ('VARCHAR(80)', 'grating')),
    ('air_temperature',  ('FLOAT', 'air_temperature')),
    ('sky_calmness',     ('VARCHAR(10)', 'sky_calmness')),
    ('sky_sharpness',    ('VARCHAR(10)', 'sky_sharpness')),
    ('sky_transparency', ('VARCHAR(10)', 'sky_transparency')),
    ('sky_conditions',   ('VARCHAR(80)', 'sky_conditions')),
    ('observer',         ('VARCHAR(80)', 'observer')),
    ('observer_notes',   ('VARCHAR(255)', 'observer_notes')),
    ('notes',            ('VARCHAR(255)', 'notes')),
    ('bibcode',          ('VARCHAR(80)', 'bibcode')),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX archive_ind', ('(archive_id)', None)),
    ('INDEX wfpdb_ind',   ('(wfpdb_id)', None)),
    ('INDEX method_ind',  ('(method_code)', None))
    ])

_schema['exposure'] = OrderedDict([
    ('exposure_id',      ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', None)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', 'db_plate_id')),
    ('archive_id',       ('INT UNSIGNED NOT NULL', 'archive_id')),
    ('exposure_num',     ('TINYINT UNSIGNED NOT NULL', None)),
    ('object_name',      ('VARCHAR(80)', 'object_name')),
    ('object_type_code', ('CHAR(2)', 'object_type_code')),
    ('ra_orig',          ('CHAR(11)', 'ra_orig')),
    ('dec_orig',         ('CHAR(11)', 'dec_orig')),
    ('flag_coord',       ('CHAR(1)', 'coord_flag')),
    ('raj2000',          ('DOUBLE', 'ra_deg')),
    ('dej2000',          ('DOUBLE', 'dec_deg')),
    ('raj2000_hms',      ('CHAR(11)', 'ra')),
    ('dej2000_dms',      ('CHAR(11)', 'dec')),
    ('date_orig_start',  ('VARCHAR(80)', 'date_orig')),
    ('date_orig_end',    ('VARCHAR(80)', 'date_orig_end')),
    ('time_orig_start',  ('VARCHAR(255)', 'tms_orig')),
    ('time_orig_end',    ('VARCHAR(255)', 'tme_orig')),
    ('flag_time',        ('CHAR(1)', 'time_flag')),
    ('ut_start',         ('DATETIME', 'date_obs')),
    ('ut_mid',           ('DATETIME', 'date_avg')),
    ('ut_weighted',      ('DATETIME', 'date_weighted')),
    ('ut_end',           ('DATETIME', 'date_end')),
    ('year_start',       ('DOUBLE', 'year')),
    ('year_mid',         ('DOUBLE', 'year_avg')),
    ('year_weighted',    ('DOUBLE', 'year_weighted')),
    ('year_end',         ('DOUBLE', 'year_end')),
    ('jd_start',         ('DOUBLE', 'jd')),
    ('jd_mid',           ('DOUBLE', 'jd_avg')),
    ('jd_weighted',      ('DOUBLE', 'jd_weighted')),
    ('jd_end',           ('DOUBLE', 'jd_end')),
    ('hjd_mid',          ('DOUBLE', 'hjd_avg')),
    ('hjd_weighted',     ('DOUBLE', 'hjd_weighted')),
    ('exptime',          ('FLOAT', 'exptime')),
    ('num_sub',          ('TINYINT UNSIGNED', 'numsub')),
    ('method_code',      ('TINYINT UNSIGNED', None)),
    ('focus',            ('FLOAT', 'focus')),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX plate_ind',   ('(plate_id)', None)),
    ('INDEX archive_ind', ('(archive_id)', None)),
    ('INDEX raj2000_ind', ('(raj2000)', None)),
    ('INDEX dej2000_ind', ('(dej2000)', None))
    ])

_schema['exposure_sub'] = OrderedDict([
    ('subexposure_id',   ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', None)),
    ('exposure_id',      ('INT UNSIGNED NOT NULL', None)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', None)),
    ('exposure_num',     ('TINYINT UNSIGNED NOT NULL', None)),
    ('subexposure_num',  ('TINYINT UNSIGNED NOT NULL', None)),
    ('date_orig_start',  ('VARCHAR(10)', 'date_orig')),
    ('time_orig_start',  ('VARCHAR(40)', 'tms_orig')),
    ('time_orig_end',    ('VARCHAR(40)', 'tme_orig')),
    ('ut_start',         ('DATETIME', 'date_obs')),
    ('ut_mid',           ('DATETIME', 'date_avg')),
    ('ut_end',           ('DATETIME', 'date_end')),
    ('jd_start',         ('DOUBLE', 'jd')),
    ('jd_mid',           ('DOUBLE', 'jd_avg')),
    ('jd_end',           ('DOUBLE', 'jd_end')),
    ('exptime',          ('FLOAT', 'exptime')),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX exposure_ind', ('(exposure_id)', None)),
    ('INDEX plate_ind',    ('(plate_id)', None))
    ])

_schema['scan'] = OrderedDict([
    ('scan_id',          ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', 
                          None)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', None)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', 'archive_id')),
    ('filename_scan',    ('VARCHAR(80)', 'filename')),
    ('filename_wedge',   ('VARCHAR(80)', 'fn_wedge')),
    ('naxis1',           ('SMALLINT UNSIGNED', 'fits_naxis1')),
    ('naxis2',           ('SMALLINT UNSIGNED', 'fits_naxis2')),
    ('minval',           ('INT', 'fits_minval')),
    ('maxval',           ('INT', 'fits_maxval')),
    ('scanner',          ('VARCHAR(80)', 'scanner')),
    ('scan_res1',        ('SMALLINT UNSIGNED', 'scan_res1')),
    ('scan_res2',        ('SMALLINT UNSIGNED', 'scan_res2')),
    ('pixel_size1',      ('FLOAT', 'pix_size1')),
    ('pixel_size2',      ('FLOAT', 'pix_size2')),
    ('scan_software',    ('VARCHAR(80)', 'scan_software')),
    ('scan_gamma',       ('FLOAT', 'scan_gamma')),
    ('scan_focus',       ('VARCHAR(80)', 'scan_focus')),
    ('wedge',            ('VARCHAR(80)', 'wedge')),
    ('scan_date',        ('DATETIME', 'datescan')),
    ('scan_author',      ('VARCHAR(80)', 'scan_author')),
    ('scan_notes',       ('VARCHAR(255)', 'scan_notes')),
    ('origin',           ('VARCHAR(80)', 'origin')),
    ('file_datetime',    ('DATETIME', 'fits_datetime')),
    ('file_size',        ('INT UNSIGNED', 'fits_size')),
    ('fits_checksum',    ('CHAR(16)', 'fits_checksum')),
    ('fits_datasum',     ('CHAR(10)', 'fits_datasum')),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX plate_ind',   ('(plate_id)', None)),
    ('INDEX archive_ind', ('(archive_id)', None))
    ])

_schema['preview'] = OrderedDict([
    ('preview_id',       ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY',
                          False)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', False)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', True)),
    ('preview_type',     ('TINYINT', True)),
    ('filename',         ('VARCHAR(80)', True)),
    ('file_format',      ('VARCHAR(80)', True)),
    ('image_width',      ('SMALLINT UNSIGNED', True)),
    ('image_height',     ('SMALLINT UNSIGNED', True)),
    ('image_datetime',   ('DATETIME', True)),
    ('file_datetime',    ('DATETIME', True)),
    ('file_size',        ('INT UNSIGNED', True)),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX archive_ind', ('(archive_id)', None))
    ])

_schema['logbook'] = OrderedDict([
    ('logbook_id',       ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', 
                          True)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', True)),
    ('logbook_num',      ('CHAR(10)', True)),
    ('logbook_title',    ('VARCHAR(80)', True)),
    ('logbook_type',     ('TINYINT', True)),
    ('logbook_notes',    ('VARCHAR(255)', True)),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX archive_ind', ('(archive_id)', None))
    ])

_schema['logpage'] = OrderedDict([
    ('logpage_id',       ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY',
                          True)),
    ('logbook_id',       ('INT UNSIGNED', True)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', True)),
    ('logpage_type',     ('TINYINT', True)),
    ('page_num',         ('SMALLINT', True)),
    ('logpage_order',    ('SMALLINT', True)),
    ('filename',         ('VARCHAR(80)', True)),
    ('file_format',      ('VARCHAR(80)', True)),
    ('image_width',      ('SMALLINT UNSIGNED', True)),
    ('image_height',     ('SMALLINT UNSIGNED', True)),
    ('image_datetime',   ('DATETIME', True)),
    ('file_datetime',    ('DATETIME', True)),
    ('file_size',        ('INT UNSIGNED', True)),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX archive_ind', ('(archive_id)', None))
    ])

_schema['plate_logpage'] = OrderedDict([
    ('plate_id',         ('INT UNSIGNED NOT NULL', None)),
    ('logpage_id',       ('INT UNSIGNED NOT NULL', None)),
    ('logpage_order',    ('TINYINT', None)),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX plate_ind',  ('(plate_id)', None)),
    ('INDEX logpage_ind',('(logpage_id)', None))
    ])

_schema['source'] = OrderedDict([
    ('source_id',        ('BIGINT UNSIGNED NOT NULL PRIMARY KEY', False)),
    ('process_id',       ('INT UNSIGNED NOT NULL', False)),
    ('scan_id',          ('INT UNSIGNED NOT NULL', False)),
    ('exposure_id',      ('INT UNSIGNED', False)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', False)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', False)),
    ('source_num',       ('INT UNSIGNED', True)),
    ('x_source',         ('DOUBLE', True)),
    ('y_source',         ('DOUBLE', True)),
    ('a_source',         ('FLOAT', True)),
    ('b_source',         ('FLOAT', True)),
    ('theta_source',     ('FLOAT', True)),
    ('erra_source',      ('FLOAT', True)),
    ('errb_source',      ('FLOAT', True)),
    ('errtheta_source',  ('FLOAT', True)),
    ('elongation',       ('FLOAT', True)),
    ('x_peak',           ('INT', True)),
    ('y_peak',           ('INT', True)),
    ('flag_usepsf',      ('TINYINT', True)),
    ('x_image',          ('DOUBLE', True)),
    ('y_image',          ('DOUBLE', True)),
    ('erra_image',       ('FLOAT', True)),
    ('errb_image',       ('FLOAT', True)),
    ('errtheta_image',   ('FLOAT', True)),
    ('x_psf',            ('DOUBLE', True)),
    ('y_psf',            ('DOUBLE', True)),
    ('erra_psf',         ('FLOAT', True)),
    ('errb_psf',         ('FLOAT', True)),
    ('errtheta_psf',     ('FLOAT', True)),
    ('mag_auto',         ('FLOAT', True)),
    ('magerr_auto',      ('FLOAT', True)),
    ('flux_auto',        ('FLOAT', True)),
    ('fluxerr_auto',     ('FLOAT', True)),
    ('mag_iso',          ('FLOAT', True)),
    ('magerr_iso',       ('FLOAT', True)),
    ('flux_iso',         ('FLOAT', True)),
    ('fluxerr_iso',      ('FLOAT', True)),
    ('flux_max',         ('FLOAT', True)),
    ('flux_radius',      ('FLOAT', True)),
    ('isoarea',          ('INT', True)),
    ('sqrt_isoarea',     ('FLOAT', True)),
    ('background',       ('FLOAT', True)),
    ('sextractor_flags', ('SMALLINT', True)),
    ('dist_center',      ('DOUBLE', True)),
    ('dist_edge',        ('DOUBLE', True)),
    ('annular_bin',      ('TINYINT', True)),
    ('flag_rim',         ('TINYINT', True)),
    ('flag_negradius',   ('TINYINT', True)),
    ('flag_clean',       ('TINYINT', True)),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX process_ind',    ('(process_id)', None)),
    ('INDEX scan_ind',       ('(scan_id)', None)),
    ('INDEX exposure_ind',   ('(exposure_id)', None)),
    ('INDEX plate_ind',      ('(plate_id)', None)),
    ('INDEX archive_ind',    ('(archive_id)', None)),
    ('INDEX annularbin_ind', ('(annular_bin)', None))
    ])

_schema['source_calib'] = OrderedDict([
    ('source_id',        ('BIGINT UNSIGNED NOT NULL PRIMARY KEY', False)),
    ('process_id',       ('INT UNSIGNED NOT NULL', False)),
    ('scan_id',          ('INT UNSIGNED NOT NULL', False)),
    ('exposure_id',      ('INT UNSIGNED', False)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', False)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', False)),
    ('annular_bin',      ('TINYINT', True)),
    ('dist_center',      ('DOUBLE', True)),
    ('dist_edge',        ('DOUBLE', True)),
    ('sextractor_flags', ('SMALLINT', True)),
    ('raj2000',          ('DOUBLE', True)),
    ('dej2000',          ('DOUBLE', True)),
    ('x_sphere',         ('DOUBLE', True)),
    ('y_sphere',         ('DOUBLE', True)),
    ('z_sphere',         ('DOUBLE', True)),
    ('healpix256',       ('INT', True)),
    ('raj2000_wcs',      ('DOUBLE', True)),
    ('dej2000_wcs',      ('DOUBLE', True)),
    ('raj2000_sub',      ('DOUBLE', True)),
    ('dej2000_sub',      ('DOUBLE', True)),
    ('raerr_sub',        ('FLOAT', True)),
    ('decerr_sub',       ('FLOAT', True)),
    ('astrom_sub_grid',  ('SMALLINT', True)),
    ('astrom_sub_id',    ('INT UNSIGNED', True)),
    ('nn_dist',          ('FLOAT', True)),
    ('zenith_angle',     ('FLOAT', True)),
    ('airmass',          ('FLOAT', True)),
    ('natmag',           ('FLOAT', True)),
    ('natmagerr',        ('FLOAT', True)),
    ('bmag',             ('FLOAT', True)),
    ('bmagerr',          ('FLOAT', True)),
    ('vmag',             ('FLOAT', True)),
    ('vmagerr',          ('FLOAT', True)),
    ('natmag_plate',     ('FLOAT', True)),
    ('natmagerr_plate',  ('FLOAT', True)),
    ('phot_plate_flags', ('TINYINT', True)),
    ('natmag_correction', ('FLOAT', True)),
    ('natmag_sub',       ('FLOAT', True)),
    ('natmagerr_sub',    ('FLOAT', True)),
    ('natmag_residual',  ('FLOAT', True)),
    ('phot_sub_grid',    ('SMALLINT', True)),
    ('phot_sub_id',      ('INT UNSIGNED', True)),
    ('phot_sub_flags',   ('TINYINT', True)),
    ('phot_calib_flags',  ('TINYINT', True)),
    ('color_term',       ('FLOAT', True)),
    ('color_bv',         ('FLOAT', True)),
    ('cat_natmag',       ('FLOAT', True)),
    ('match_radius',     ('FLOAT', True)),
    ('tycho2_id',        ('CHAR(12)', True)),
    ('tycho2_id_pad',    ('CHAR(12)', True)),
    ('tycho2_ra',        ('DOUBLE', True)),
    ('tycho2_dec',       ('DOUBLE', True)),
    ('tycho2_btmag',     ('FLOAT', True)),
    ('tycho2_vtmag',     ('FLOAT', True)),
    ('tycho2_btmagerr',  ('FLOAT', True)),
    ('tycho2_vtmagerr',  ('FLOAT', True)),
    ('tycho2_hip',       ('INT UNSIGNED', True)),
    ('tycho2_dist',      ('FLOAT', True)),
    ('tycho2_dist2',     ('FLOAT', True)),
    ('tycho2_nn_dist',   ('FLOAT', True)),
    ('ucac4_id',         ('CHAR(10)', True)),
    ('ucac4_ra',         ('DOUBLE', True)),
    ('ucac4_dec',        ('DOUBLE', True)),
    ('ucac4_bmag',       ('FLOAT', True)),
    ('ucac4_vmag',       ('FLOAT', True)),
    ('ucac4_bmagerr',    ('FLOAT', True)),
    ('ucac4_vmagerr',    ('FLOAT', True)),
    ('ucac4_dist',       ('FLOAT', True)),
    ('ucac4_dist2',      ('FLOAT', True)),
    ('ucac4_nn_dist',    ('FLOAT', True)),
    ('apass_ra',         ('DOUBLE', True)),
    ('apass_dec',        ('DOUBLE', True)),
    ('apass_bmag',       ('FLOAT', True)),
    ('apass_vmag',       ('FLOAT', True)),
    ('apass_bmagerr',    ('FLOAT', True)),
    ('apass_vmagerr',    ('FLOAT', True)),
    ('apass_dist',       ('FLOAT', True)),
    ('apass_dist2',      ('FLOAT', True)),
    ('apass_nn_dist',    ('FLOAT', True)),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX source_ind',   ('(source_id)', None)),
    ('INDEX process_ind',  ('(process_id)', None)),
    ('INDEX scan_ind',     ('(scan_id)', None)),
    ('INDEX exposure_ind', ('(exposure_id)', None)),
    ('INDEX plate_ind',    ('(plate_id)', None)),
    ('INDEX archive_ind',  ('(archive_id)', None)),
    ('INDEX annularbin_ind', ('(annular_bin)', None)),
    ('INDEX raj2000_ind',  ('(raj2000)', None)),
    ('INDEX dej2000_ind',  ('(dej2000)', None)),
    ('INDEX x_ind',        ('(x_sphere)', None)),
    ('INDEX y_ind',        ('(y_sphere)', None)),
    ('INDEX z_ind',        ('(z_sphere)', None)),
    ('INDEX healpix256_ind', ('(healpix256)', None)),
    ('INDEX nndist_ind',   ('(nn_dist)', None)),
    ('INDEX natmag_ind',   ('(natmag)', None)),
    ('INDEX bmag_ind',     ('(bmag)', None)),
    ('INDEX vmag_ind',     ('(vmag)', None)),
    ('INDEX tycho2_ind',   ('(tycho2_id)', None)),
    ('INDEX hip_ind',      ('(tycho2_hip)', None)),
    ('INDEX ucac4_ind',    ('(ucac4_id)', None))
    ])

_schema['solution'] = OrderedDict([
    ('solution_id',      ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', 
                          False)),
    ('process_id',       ('INT UNSIGNED NOT NULL', False)),
    ('scan_id',          ('INT UNSIGNED NOT NULL', False)),
    ('exposure_id',      ('INT UNSIGNED', False)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', False)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', False)),
    ('solution_num',     ('INT UNSIGNED', True)),
    ('raj2000',          ('DOUBLE', True)),
    ('dej2000',          ('DOUBLE', True)),
    ('raj2000_hms',      ('CHAR(11)', True)),
    ('dej2000_dms',      ('CHAR(11)', True)),
    ('fov1',             ('FLOAT', True)),
    ('fov2',             ('FLOAT', True)),
    ('pixel_scale',      ('FLOAT', True)),
    ('source_density',   ('FLOAT', True)),
    ('cd1_1',            ('DOUBLE', True)),
    ('cd1_2',            ('DOUBLE', True)),
    ('cd2_1',            ('DOUBLE', True)),
    ('cd2_2',            ('DOUBLE', True)),
    ('rotation_angle',   ('FLOAT', True)),
    ('plate_mirrored',   ('TINYINT(1)', True)),
    ('ncp_on_plate',     ('TINYINT(1)', True)),
    ('scp_on_plate',     ('TINYINT(1)', True)),
    ('stc_box',          ('VARCHAR(100)', True)),
    ('stc_polygon',      ('VARCHAR(200)', True)),
    ('header_wcs',       ('TEXT', True)),
    ('header_anet',      ('TEXT', True)),
    ('header_scamp',     ('TEXT', True)),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX process_ind',  ('(process_id)', None)),
    ('INDEX scan_ind',     ('(scan_id)', None)),
    ('INDEX exposure_ind', ('(exposure_id)', None)),
    ('INDEX plate_ind',    ('(plate_id)', None)),
    ('INDEX archive_ind',  ('(archive_id)', None)),
    ('INDEX raj2000_ind',  ('(raj2000)', None)),
    ('INDEX dej2000_ind',  ('(dej2000)', None))
    ])

_schema['astrom_sub'] = OrderedDict([
    ('sub_id',           ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', 
                          False)),
    ('process_id',       ('INT UNSIGNED NOT NULL', False)),
    ('scan_id',          ('INT UNSIGNED NOT NULL', False)),
    ('exposure_id',      ('INT UNSIGNED', False)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', False)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', False)),
    ('astrom_sub_grid',  ('SMALLINT', True)),
    ('astrom_sub_id',    ('INT UNSIGNED', True)),
    ('parent_sub_id',    ('INT UNSIGNED', True)),
    ('x_min',            ('FLOAT', True)),
    ('x_max',            ('FLOAT', True)),
    ('y_min',            ('FLOAT', True)),
    ('y_max',            ('FLOAT', True)),
    ('x_min_ext',        ('FLOAT', True)),
    ('x_max_ext',        ('FLOAT', True)),
    ('y_min_ext',        ('FLOAT', True)),
    ('y_max_ext',        ('FLOAT', True)),
    ('num_sub_stars',    ('INT UNSIGNED', True)),
    ('num_ref_stars',    ('INT UNSIGNED', True)),
    ('above_threshold',  ('TINYINT(1)', True)),
    ('num_selected_ref_stars', ('INT UNSIGNED', True)),
    ('scamp_crossid_radius', ('FLOAT', True)),
    ('num_scamp_stars',  ('INT UNSIGNED', True)),
    ('scamp_sigma_axis1', ('FLOAT', True)),
    ('scamp_sigma_axis2', ('FLOAT', True)),
    ('scamp_sigma_mean', ('FLOAT', True)),
    ('scamp_sigma_prev_axis1', ('FLOAT', True)),
    ('scamp_sigma_prev_axis2', ('FLOAT', True)),
    ('scamp_sigma_prev_mean', ('FLOAT', True)),
    ('scamp_sigma_diff', ('FLOAT', True)),
    ('apply_astrometry',  ('TINYINT(1)', True)),
    ('num_applied_stars', ('INT UNSIGNED', True)),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX process_ind',  ('(process_id)', None)),
    ('INDEX scan_ind',     ('(scan_id)', None)),
    ('INDEX exposure_ind', ('(exposure_id)', None)),
    ('INDEX plate_ind',    ('(plate_id)', None)),
    ('INDEX archive_ind',  ('(archive_id)', None)),
    ('INDEX astromsubid_ind', ('(astrom_sub_id)', None))
    ])
    
_schema['phot_calib'] = OrderedDict([
    ('calib_id',         ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', 
                          False)),
    ('process_id',       ('INT UNSIGNED NOT NULL', False)),
    ('scan_id',          ('INT UNSIGNED NOT NULL', False)),
    ('exposure_id',      ('INT UNSIGNED', False)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', False)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', False)),
    ('solution_num',     ('TINYINT', True)),
    ('color_term',       ('FLOAT', True)),
    ('color_term_err',   ('FLOAT', True)),
    ('num_candidate_stars', ('INT UNSIGNED', True)),
    ('num_calib_stars',  ('INT UNSIGNED', True)),
    ('num_bright_stars', ('INT UNSIGNED', True)),
    ('num_outliers',     ('INT UNSIGNED', True)),
    ('bright_limit',     ('FLOAT', True)),
    ('faint_limit',      ('FLOAT', True)),
    ('mag_range',        ('FLOAT', True)),
    ('rmse_min',         ('FLOAT', True)),
    ('rmse_median',      ('FLOAT', True)),
    ('rmse_max',         ('FLOAT', True)),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX process_ind',  ('(process_id)', None)),
    ('INDEX scan_ind',     ('(scan_id)', None)),
    ('INDEX exposure_ind', ('(exposure_id)', None)),
    ('INDEX plate_ind',    ('(plate_id)', None)),
    ('INDEX archive_ind',  ('(archive_id)', None)),
    ('INDEX annularbin_ind', ('(annular_bin)', None))
    ])
    
_schema['phot_sub'] = OrderedDict([
    ('sub_id',           ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', 
                          False)),
    ('process_id',       ('INT UNSIGNED NOT NULL', False)),
    ('scan_id',          ('INT UNSIGNED NOT NULL', False)),
    ('exposure_id',      ('INT UNSIGNED', False)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', False)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', False)),
    ('phot_sub_grid',    ('SMALLINT', True)),
    ('phot_sub_id',      ('INT UNSIGNED', True)),
    ('parent_sub_id',    ('INT UNSIGNED', True)),
    ('x_min',            ('FLOAT', True)),
    ('x_max',            ('FLOAT', True)),
    ('y_min',            ('FLOAT', True)),
    ('y_max',            ('FLOAT', True)),
    ('num_selected_stars', ('INT UNSIGNED', True)),
    ('above_threshold',  ('TINYINT(1)', True)),
    ('num_fit_stars',    ('INT UNSIGNED', True)),
    ('correction_min',   ('FLOAT', True)),
    ('correction_max',   ('FLOAT', True)),
    ('num_applied_stars', ('INT UNSIGNED', True)),
    ('rmse_min',         ('FLOAT', True)),
    ('rmse_median',      ('FLOAT', True)),
    ('rmse_max',         ('FLOAT', True)),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX process_ind',  ('(process_id)', None)),
    ('INDEX scan_ind',     ('(scan_id)', None)),
    ('INDEX exposure_ind', ('(exposure_id)', None)),
    ('INDEX plate_ind',    ('(plate_id)', None)),
    ('INDEX archive_ind',  ('(archive_id)', None)),
    ('INDEX photsubid_ind', ('(phot_sub_id)', None))
    ])
    
_schema['phot_color'] = OrderedDict([
    ('color_id',         ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', 
                          False)),
    ('process_id',       ('INT UNSIGNED NOT NULL', False)),
    ('scan_id',          ('INT UNSIGNED NOT NULL', False)),
    ('exposure_id',      ('INT UNSIGNED', False)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', False)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', False)),
    ('color_term',       ('FLOAT', True)),
    ('color_term_err',   ('FLOAT', True)),
    ('stdev_fit',        ('FLOAT', True)),
    ('stdev_min',        ('FLOAT', True)),
    ('cterm_min',        ('FLOAT', True)),
    ('cterm_max',        ('FLOAT', True)),
    ('iteration',        ('TINYINT UNSIGNED', True)),
    ('num_stars',        ('INT UNSIGNED', True)),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX process_ind',  ('(process_id)', None)),
    ('INDEX scan_ind',     ('(scan_id)', None)),
    ('INDEX exposure_ind', ('(exposure_id)', None)),
    ('INDEX plate_ind',    ('(plate_id)', None)),
    ('INDEX archive_ind',  ('(archive_id)', None))
    ])
    
_schema['phot_cterm'] = OrderedDict([
    ('cterm_id',         ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', 
                          False)),
    ('process_id',       ('INT UNSIGNED NOT NULL', False)),
    ('scan_id',          ('INT UNSIGNED NOT NULL', False)),
    ('exposure_id',      ('INT UNSIGNED', False)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', False)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', False)),
    ('iteration',        ('INT UNSIGNED', True)),
    ('cterm',            ('FLOAT', True)),
    ('stdev',            ('FLOAT', True)),
    ('num_stars',        ('INT UNSIGNED', True)),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX process_ind',  ('(process_id)', None)),
    ('INDEX scan_ind',     ('(scan_id)', None)),
    ('INDEX exposure_ind', ('(exposure_id)', None)),
    ('INDEX plate_ind',    ('(plate_id)', None)),
    ('INDEX archive_ind',  ('(archive_id)', None))
    ])
    
_schema['process'] = OrderedDict([
    ('process_id',       ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', 
                          None)),
    ('scan_id',          ('INT UNSIGNED', None)),
    ('plate_id',         ('INT UNSIGNED', None)),
    ('archive_id',       ('INT UNSIGNED', None)),
    ('filename',         ('VARCHAR(80)', None)),
    ('hostname',         ('VARCHAR(80)', None)),
    ('timestamp_start',  ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_end',    ('TIMESTAMP NULL', None)),
    ('duration',         ('INT UNSIGNED', None)),
    ('sky',              ('FLOAT', None)),
    ('sky_sigma',        ('FLOAT', None)),
    ('use_psf',          ('TINYINT(1)', None)),
    ('threshold',        ('FLOAT', None)),
    ('num_sources',      ('INT UNSIGNED', None)),
    ('num_psf_sources',  ('INT UNSIGNED', None)),
    ('solved',           ('TINYINT(1)', None)),
    ('astrom_sub_total', ('INT UNSIGNED', None)),
    ('astrom_sub_eff',   ('INT UNSIGNED', None)),
    ('num_ucac4',        ('INT UNSIGNED', None)),
    ('num_tycho2',       ('INT UNSIGNED', None)),
    ('num_apass',        ('INT UNSIGNED', None)),
    ('color_term',       ('FLOAT', None)),
    ('bright_limit',     ('FLOAT', None)),
    ('faint_limit',      ('FLOAT', None)),
    ('mag_range',        ('FLOAT', None)),
    ('num_calib',        ('INT UNSIGNED', None)),
    ('calibrated',       ('TINYINT(1)', None)),
    ('phot_sub_total',   ('INT UNSIGNED', None)),
    ('phot_sub_eff',     ('INT UNSIGNED', None)),
    ('completed',        ('TINYINT(1)', None)),
    ('pyplate_version',  ('VARCHAR(15)', None)),
    ('INDEX scan_ind',   ('(scan_id)', None)),
    ('INDEX plate_ind',  ('(plate_id)', None)),
    ('INDEX archive_ind', ('(archive_id)', None)),
    ('INDEX filename_ind', ('(filename)', None))
    ])

_schema['process_log'] = OrderedDict([
    ('processlog_id',    ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', 
                          None)),
    ('process_id',       ('INT UNSIGNED NOT NULL', None)),
    ('timestamp_log',    ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('scan_id',          ('INT UNSIGNED', None)),
    ('plate_id',         ('INT UNSIGNED', None)),
    ('archive_id',       ('INT UNSIGNED', None)),
    ('level',            ('TINYINT', None)),
    ('event',            ('SMALLINT', None)),
    ('message',          ('TEXT', None)),
    ('INDEX process_ind', ('(process_id)', None)),
    ('INDEX scan_ind',   ('(scan_id)', None)),
    ('INDEX plate_ind',  ('(plate_id)', None)),
    ('INDEX archive_ind', ('(archive_id)', None)),
    ('INDEX event_ind',  ('(event)', None))
    ])

def _get_columns_sql(table):
    """
    Print table column statements.

    """

    sql = None

    if table in _schema:
        #sql = 'CREATE TABLE {} (\n'.format(table)
        sql_list = ['    {:15s} {}'.format(k, v[0]) 
                    for k,v in _schema[table].items()]
        sql = ',\n'.join(sql_list)

    return sql

def print_tables(use_drop=False, engine='Aria'):
    """
    Print table creation SQL queries to standard output.

    """

    sql_drop = '\n'.join(['DROP TABLE IF EXISTS {};'.format(k) 
                          for k in _schema.keys()])

    sql_list = ['CREATE TABLE {} (\n{}\n) ENGINE={} '
                'CHARACTER SET=utf8 COLLATE=utf8_unicode_ci;\n'
                .format(k, _get_columns_sql(k), engine)
                for k in _schema.keys()]
    sql = '\n'.join(sql_list)

    if use_drop:
        sql = sql_drop + '\n\n' + sql

    print(sql)


class PlateDB:
    """
    Plate database class.

    """

    def __init__(self, **kwargs):
        """Initialise PlateDB class

        Parameters
        ----------
        rdbms : str
            Database management system ('pgsql', 'mysql')
        host : str
            Database host name
        port : int
            Port number for database connection
        user : str
            Database user name
        password : str
            Database password
        database : str
            Database name
        schema : str
            Database schema
        """

        self.rdbms = kwargs.pop('rdbms', 'pgsql')
        self.host = kwargs.pop('host', 'localhost')
        self.port = kwargs.pop('port', None)
        self.user = kwargs.pop('user', '')
        self.database = kwargs.pop('database', '')
        self.password = kwargs.pop('password', '')
        self.schema = kwargs.pop('schema', '')
        self.schema_dict = None
        self.trigger_dict = None
        self.db = None
        self.write_db_source_dir = ''
        self.write_db_source_calib_dir = ''

        # Read database schema
        self.read_schema()

    def assign_conf(self, conf):
        """
        Assign and parse configuration.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        self.conf = conf

        for attr in ['write_log_dir', 'write_db_source_dir', 
                     'write_db_source_calib_dir']:
            try:
                setattr(self, attr, conf.get('Files', attr))
            except configparser.Error:
                pass

        for attr in zip(['host', 'user', 'database', 'password', 'schema'],
                        ['output_db_host', 'output_db_user',
                         'output_db_name', 'output_db_passwd',
                         'output_db_schema']):
            try:
                setattr(self, attr[0], conf.get('Database', attr[1]))
            except configparser.Error:
                pass

        for attr in [('port', 'output_db_port')]:
            try:
                setattr(self, attr[0], conf.getint('Database', attr[1]))
            except configparser.Error:
                pass

        # Read database schema
        self.read_schema()

    def read_schema(self, schema=None):
        """Read schema from schema YAML file.

        Parameters
        ----------
        schema : str
            Database schema
        """

        if schema is None:
            schema = self.schema

        if schema in ['applause_dr4']:
            fn_yaml = '{}.yaml'.format(self.schema)
            path_yaml = os.path.join(os.path.dirname(__file__), '../config',
                                     fn_yaml)
            d1, d2 = fetch_ordered_tables(path_yaml, self.rdbms, True)
            self.schema_dict = d1
            self.trigger_dict = d2

    def table_name(self, table):
        """
        Combine schema and table names

        Paramaters
        ----------
        table : str
            Table name
        """

        if self.schema:
            return '{}.{}'.format(self.schema, table)
        else:
            return table

    def get_table_dict(self, table):
        """
        Combine schema and table names and get table structure from schema_dict

        Paramaters
        ----------
        table : str
            Table name
        """

        table_name = self.table_name(table)

        if self.schema_dict and table_name in self.schema_dict:
            return self.schema_dict[table_name].copy()
        else:
            return None

    def open_connection(self, rdbms='pgsql', host=None, port=None, user=None,
                        password=None, database=None, schema=None):
        """
        Open database connection.

        Parameters
        ----------
        rdbms : str
            Database management system ('pgsql', 'mysql')
        host : str
            Database host name
        port : int
            Port number for database connection
        user : str
            Database user name
        password : str
            Database password
        database : str
            Database name
        schema : str
            Database schema

        """

        if host is None:
            host = self.host

        if port is None:
            port = self.port

        if user is None:
            user = self.user

        if password is None:
            password = self.password

        if database is None:
            database = self.database

        if schema is not None:
            self.schema = schema

        if rdbms == 'pgsql':
            self.db = DB_pgsql()
            self.db.open_connection(host=host, port=port,
                                    user=user, password=password,
                                    database=database)
        elif rdbms == 'mysql':
            # Implement MySQL/MariaDB connection here
            pass

    def close_connection(self):
        """
        Close database connection.

        """

        if self.db is not None:
            self.db.close_connection()
        
    def write_plate(self, platemeta):
        """
        Write plate entry to the database.

        Parameters
        ----------
        platemeta : Plate
            Plate metadata instance

        Returns
        -------
        plate_id : int
            Plate ID number

        """

        # Create a dictionary of keywords that differ from database schema
        pmeta_dict = {}
        pmeta_dict['plate_id'] = 'db_plate_id'
        pmeta_dict['flag_coord'] = 'coord_flag'
        pmeta_dict['raj2000'] = 'ra_deg'
        pmeta_dict['dej2000'] = 'dec_deg'
        pmeta_dict['raj2000_hms'] = 'ra'
        pmeta_dict['dej2000_dms'] = 'dec'
        pmeta_dict['date_orig_start'] = 'date_orig'
        pmeta_dict['date_orig_end'] = 'date_orig_end'
        pmeta_dict['time_orig_start'] = 'tms_orig'
        pmeta_dict['time_orig_end'] = 'tme_orig'
        pmeta_dict['flag_time'] = 'time_flag'
        pmeta_dict['ut_start'] = 'date_obs'
        pmeta_dict['ut_mid'] = 'date_avg'
        pmeta_dict['ut_weighted'] = 'date_weighted'
        pmeta_dict['ut_end'] = 'date_end'
        pmeta_dict['year_start'] = 'year'
        pmeta_dict['year_mid'] = 'year_avg'
        pmeta_dict['jd_start'] = 'jd'
        pmeta_dict['jd_mid'] = 'jd_avg'
        pmeta_dict['hjd_mid'] = 'hjd_avg'
        pmeta_dict['num_sub'] = 'numsub'

        # The plate table
        col_list = []
        val_tuple = ()

        if (isinstance(platemeta['db_plate_id'], int) and 
            (platemeta['db_plate_id'] > 0)):
            col_list.append('plate_id')
            val_tuple = val_tuple + (platemeta['db_plate_id'],)

        plate_table = self.get_table_dict('plate')
        del plate_table['plate_id']

        for k,v in plate_table.items():
            # Take a keyword from pmeta_dict if it is there
            kw = pmeta_dict[k] if k in pmeta_dict else k

            if kw in platemeta:
                col_list.append(k)

                # Validate date type and insert NULL instead of invalid value
                if v == 'DATE':
                    try:
                        d = Time(platemeta.get_value(kw), scale='tai')

                        if d >= Time('1000-01-01', scale='tai'):
                            val_tuple += (platemeta.get_value(kw),)
                        else:
                            val_tuple += (None,)
                    except ValueError:
                        val_tuple += (None,)
                else:
                    val_tuple += (platemeta.get_value(kw),)

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))

        sql = ('INSERT INTO {} ({}) VALUES ({}) RETURNING plate_id'
               .format(self.table_name('plate'), col_str, val_str))
        plate_id = self.db.execute_query(sql, val_tuple)
        platemeta['db_plate_id'] = plate_id

        # The exposure table
        for exp in range(platemeta['numexp']):
            exp_num = exp + 1
            col_list = ['exposure_num']
            val_tuple = (exp_num,)

            exposure_table = self.get_table_dict('exposure')

            for k in exposure_table.keys():
                # Take a keyword from pmeta_dict if it is there
                kw = pmeta_dict[k] if k in pmeta_dict else k

                if kw in platemeta:
                    col_list.append(k)
                    val_tuple += (platemeta.get_value(kw, exp=exp),)

            col_str = ','.join(col_list)
            val_str = ','.join(['%s'] * len(col_list))

            sql = ('INSERT INTO {} ({}) VALUES ({}) RETURNING exposure_id'
                   .format(self.table_name('exposure'), col_str, val_str))
            exposure_id = self.db.execute_query(sql, val_tuple)

            # The exposure_sub table
            if len(platemeta['numsub']) > exp and platemeta['numsub'][exp] > 1:
                for subexp in np.arange(platemeta['numsub'][exp]):
                    subexp_num = subexp + 1
                    col_list = ['exposure_id', 'plate_id',
                                'exposure_num', 'subexposure_num']
                    val_tuple = (exposure_id, plate_id, exp_num,
                                 subexp_num)

                    expmeta = platemeta.exposures[exp]
                    exposure_sub_table = self.get_table_dict('exposure_sub')
                    del exposure_sub_table['plate_id']

                    for k in exposure_sub_table.keys():
                        # Take a keyword from pmeta_dict if it is there
                        kw = pmeta_dict[k] if k in pmeta_dict else k

                        if kw in expmeta:
                            col_list.append(k)
                            val_tuple += (expmeta.get_value(kw, exp=subexp),)

                    col_str = ','.join(col_list)
                    val_str = ','.join(['%s'] * len(col_list))

                    sql = ('INSERT INTO {} ({}) VALUES ({})'
                           .format(self.table_name('exposure_sub'), col_str,
                                   val_str))
                    self.db.execute_query(sql, val_tuple)

    def write_plate_logpage(self, platemeta):
        """
        Write plate-logpage relations to the database.

        Parameters
        ----------
        platemeta : Plate
            Plate metadata instance

        """

        fn_list = [platemeta['fn_cover']]
        fn_list.extend(platemeta['fn_log'])

        for order,filename in enumerate(fn_list):
            if filename:
                col_list = ['plate_id', 'logpage_id', 'logpage_order']

                if (isinstance(platemeta['db_plate_id'], int) and 
                    (platemeta['db_plate_id'] > 0)):
                    plate_id = platemeta['db_plate_id']
                else:
                    plate_id = self.get_plate_id(platemeta['plate_num'],
                                                 platemeta['archive_id'])

                    if plate_id is None:
                        plate_id = self.get_plate_id_wfpdb(platemeta['wfpdb_id'])

                logpage_id = self.get_logpage_id(filename, 
                                                 platemeta['archive_id'])

                if plate_id and logpage_id:
                    val_tuple = (plate_id, logpage_id, order)
                    col_str = ','.join(col_list)
                    val_str = ','.join(['%s'] * len(col_list))
                    sql = ('INSERT INTO {} ({}) VALUES ({})'
                           .format(self.table_name('plate_logpage'), col_str,
                                   val_str))
                    self.db.execute_query(sql, val_tuple)

    def write_scan(self, platemeta):
        """
        Write scan entry to the database.

        Parameters
        ----------
        platemeta : Plate
            Plate metadata instance

        Returns
        -------
        scan_id : int
            Scan ID number

        """

        # Create a dictionary of keywords that differ from database schema
        pmeta_dict = {}
        pmeta_dict['plate_id'] = 'db_plate_id'
        pmeta_dict['filename_scan'] = 'filename'
        pmeta_dict['filename_wedge'] = 'fn_wedge'
        pmeta_dict['naxis1'] = 'fits_naxis1'
        pmeta_dict['naxis2'] = 'fits_naxis2'
        pmeta_dict['minval'] = 'fits_minval'
        pmeta_dict['maxval'] = 'fits_maxval'
        pmeta_dict['pixel_size1'] = 'pix_size1'
        pmeta_dict['pixel_size2'] = 'pix_size2'
        pmeta_dict['scan_date'] = 'datescan'
        pmeta_dict['file_datetime'] = 'fits_datetime'
        pmeta_dict['file_size'] = 'fits_size'

        if (isinstance(platemeta['db_plate_id'], int) and 
            (platemeta['db_plate_id'] > 0)):
            plate_id = platemeta['db_plate_id']
        else:
            plate_id = self.get_plate_id(platemeta['plate_num'],
                                         platemeta['archive_id'])

            if plate_id is None:
                plate_id = self.get_plate_id_wfpdb(platemeta['wfpdb_id'])

        col_list = ['plate_id']
        val_tuple = (plate_id,)

        # Add scan_id only if it is given in platemeta
        if (isinstance(platemeta['scan_id'], int) and
            (platemeta['scan_id'] > 0)):
            col_list.append('scan_id')
            val_tuple += (platemeta['scan_id'],)

        # Get scan table columns from database schema
        scan_table = self.get_table_dict('scan')
        del scan_table['plate_id']
        del scan_table['scan_id']

        for k,v in scan_table.items():
            # Take a keyword from pmeta_dict if it is there
            kw = pmeta_dict[k] if k in pmeta_dict else k

            if kw in platemeta:
                col_list.append(k)

                val_tuple = val_tuple \
                        + (platemeta.get_value(kw), )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))

        sql = ('INSERT INTO {} ({}) VALUES ({}) RETURNING scan_id'
               .format(self.table_name('scan'), col_str, val_str))
        scan_id = self.db.execute_query(sql, val_tuple)

        return scan_id

    def update_scan(self, platemeta, filecols=False):
        """
        Update scan entry in the database.

        Parameters
        ----------
        platemeta : Plate
            Plate metadata instance
        filecols : bool
            If True, only specific file-related columns are updated

        """

        # Create a dictionary of keywords that differ from database schema
        pmeta_dict = {}
        pmeta_dict['plate_id'] = 'db_plate_id'
        pmeta_dict['filename_scan'] = 'filename'
        pmeta_dict['filename_wedge'] = 'fn_wedge'
        pmeta_dict['naxis1'] = 'fits_naxis1'
        pmeta_dict['naxis2'] = 'fits_naxis2'
        pmeta_dict['minval'] = 'fits_minval'
        pmeta_dict['maxval'] = 'fits_maxval'
        pmeta_dict['pixel_size1'] = 'pix_size1'
        pmeta_dict['pixel_size2'] = 'pix_size2'
        pmeta_dict['scan_date'] = 'datescan'
        pmeta_dict['file_datetime'] = 'fits_datetime'
        pmeta_dict['file_size'] = 'fits_size'

        if (isinstance(platemeta['scan_id'], int) and 
            (platemeta['scan_id'] > 0)):
            scan_id = platemeta['scan_id']
        else:
            scan_id,_ = self.get_scan_id(platemeta['filename'],
                                         platemeta['archive_id'])

            if scan_id is None:
                print('Cannot update scan metadata in the database '
                      '(filename={}, archive_id={})'
                      ''.format(platemeta['filename'],
                                platemeta['archive_id']))
                return

        col_list = []
        val_tuple = ()

        # Get scan table columns from database schema
        scan_table = self.get_table_dict('scan')
        del scan_table['scan_id']

        columns = [k for k in scan_table.keys()]

        # Update only specific columns
        if filecols:
            columns = ['file_datetime', 'file_size', 'fits_checksum', 
                       'fits_datasum']

        for c in columns:
            platemeta_key = pmeta_dict[c] if c in pmeta_dict else c

            if platemeta_key in platemeta:
                c_str = '{}=%s'.format(c)
                col_list.append(c_str)
                val_tuple += (platemeta.get_value(platemeta_key),)

        col_str = ','.join(col_list)
        val_tuple += (scan_id,)

        sql = ('UPDATE {} SET {} WHERE scan_id=%s'
               .format(self.table_name('scan'), col_str))
        self.db.execute_query(sql, val_tuple)

    def write_preview(self, previewmeta):
        """
        Write preview image entry to the database.

        Parameters
        ----------
        previewmeta : Preview
            Preview metadata instance

        Returns
        -------
        preview_id : int
            Preview ID number

        """

        if (isinstance(previewmeta['db_plate_id'], int) and 
            (previewmeta['db_plate_id'] > 0)):
            plate_id = previewmeta['db_plate_id']
        else:
            plate_id = self.get_plate_id(previewmeta['plate_num'],
                                         previewmeta['archive_id'])

            if plate_id is None:
                plate_id = self.get_plate_id_wfpdb(previewmeta['wfpdb_id'])

        col_list = ['plate_id']
        val_tuple = (plate_id,)

        # Add preview_id only if it is given in previewmeta
        if (isinstance(previewmeta['preview_id'], int) and 
            (previewmeta['preview_id'] > 0)):
            col_list.append('preview_id')
            val_tuple += (previewmeta['preview_id'],)

        # Get preview table columns from database schema
        preview_table = self.get_table_dict('preview')
        del preview_table['plate_id']
        del preview_table['preview_id']

        for k in preview_table.keys():
            if k in previewmeta:
                col_list.append(k)
                val_tuple = val_tuple + (previewmeta[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))

        sql = ('INSERT INTO {} ({}) VALUES ({}) RETURNING preview_id'
               .format(self.table_name('preview'), col_str, val_str))
        preview_id = self.db.execute_query(sql, val_tuple)
        return preview_id

    def write_logbook(self, logbookmeta):
        """
        Write a logbook to the database.

        """

        col_list = []
        val_tuple = ()

        # Get logbook table columns from database schema
        logbook_table = self.get_table_dict('logbook')
        del logbook_table['logbook_id']

        for k in logbook_table.keys():
            if k in logbookmeta:
                col_list.append(k)
                val_tuple = val_tuple + (logbookmeta[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO {} ({}) VALUES ({}) RETURNING logbook_id'
               .format(self.table_name('logbook'), col_str, val_str))
        logbook_id = self.db.execute_query(sql, val_tuple)
        return logbook_id

    def write_logpage(self, logpagemeta):
        """
        Write a single logpage to the database.

        """

        if logpagemeta['logbook_id'] is None:
            logbook_id = self.get_logbook_id(logpagemeta['logbook_num'],
                                             logpagemeta['archive_id'])
            logpagemeta['logbook_id'] = logbook_id
            
        col_list = []
        val_tuple = ()

        # Get logpage table columns from database schema
        logpage_table = self.get_table_dict('logpage')
        del logpage_table['logpage_id']

        for k in logpage_table.keys():
            if k in logpagemeta:
                col_list.append(k)
                val_tuple = val_tuple + (logpagemeta[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO {} ({}) VALUES ({}) RETURNING logpage_id'
               .format(self.table_name('logpage'), col_str, val_str))
        logpage_id = self.db.execute_query(sql, val_tuple)
        return logpage_id

    def write_solution(self, solution, process_id=None, scan_id=None, 
                       plate_id=None, archive_id=None):
        """
        Write plate solution to the database.

        """

        col_list = ['process_id', 'scan_id', 'plate_id', 'archive_id']
        val_tuple = (process_id, scan_id, plate_id, archive_id)

        # Get solution table columns from database schema
        solution_table = self.get_table_dict('solution')

        for k in solution_table.keys():
            if k in solution:
                col_list.append(k)

                if isinstance(solution[k], u.Quantity):
                    value = solution[k].value
                elif isinstance(solution[k], fits.Header):
                    value = solution[k].tostring(sep='\\n')
                elif isinstance(solution[k], bool):
                    value = int(solution[k])
                else:
                    value = solution[k]

                val_tuple = val_tuple + (value, )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO {} ({}) VALUES ({}) RETURNING solution_id'
               .format(self.table_name('solution'), col_str, val_str))
        solution_id = self.db.execute_query(sql, val_tuple)
        return solution_id

    def write_phot_cterm(self, phot_cterm, process_id=None, scan_id=None, 
                         plate_id=None, archive_id=None):
        """
        Write photometric color term data to the database.

        """

        col_list = ['process_id', 'scan_id', 'exposure_id',
                    'plate_id', 'archive_id']
        val_tuple = (process_id, scan_id, None, plate_id, archive_id)

        # Get phot_cterm table columns from database schema
        phot_cterm_table = self.get_table_dict('phot_cterm')

        for k in phot_cterm_table.keys():
            if k in phot_cterm:
                col_list.append(k)
                val_tuple = val_tuple + (phot_cterm[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO {} ({}) VALUES ({})'
               .format(self.table_name('phot_cterm'), col_str, val_str))
        self.db.execute_query(sql, val_tuple)

    def write_phot_color(self, phot_color, process_id=None, scan_id=None, 
                         plate_id=None, archive_id=None):
        """
        Write photometric color term result to the database.

        """

        col_list = ['process_id', 'scan_id', 'exposure_id',
                    'plate_id', 'archive_id']
        val_tuple = (process_id, scan_id, None, plate_id, archive_id)

        # Get phot_cterm table columns from database schema
        phot_color_table = self.get_table_dict('phot_color')

        for k in phot_color_table.keys():
            if k in phot_color:
                col_list.append(k)
                val_tuple = val_tuple + (phot_color[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO {} ({}) VALUES ({})'
               .format(self.table_name('phot_color'), col_str, val_str))
        self.db.execute_query(sql, val_tuple)

    def write_phot_calib(self, phot_calib, process_id=None, scan_id=None, 
                         plate_id=None, archive_id=None):
        """
        Write photometric calibration to the database.

        """

        col_list = ['process_id', 'scan_id', 'exposure_id',
                    'plate_id', 'archive_id']
        val_tuple = (process_id, scan_id, None, plate_id, archive_id)

        # Get phot_calib table columns from database schema
        phot_calib_table = self.get_table_dict('phot_calib')

        for k in phot_calib_table.keys():
            if k in phot_calib:
                col_list.append(k)
                val_tuple = val_tuple + (phot_calib[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO {} ({}) VALUES ({})'
               .format(self.table_name('phot_calib'), col_str, val_str))
        self.db.execute_query(sql, val_tuple)

    def write_sources(self, sources, process_id=None, scan_id=None, 
                      plate_id=None, archive_id=None, write_csv=None):
        """
        Write source data to the database.

        """

        # Open CSV files for writing
        if write_csv:
            fn_source_csv = '{:05d}_source.csv'.format(process_id)
            fn_source_csv = os.path.join(self.write_db_source_dir, 
                                         fn_source_csv)
            source_csv = open(fn_source_csv, 'wb')
            source_writer = csvWriter(source_csv, delimiter=',',
                                      quotechar='"', 
                                      quoting=csv.QUOTE_MINIMAL)
            fn_source_calib_csv = '{:05d}_source_calib.csv'.format(process_id)
            fn_source_calib_csv = os.path.join(self.write_db_source_calib_dir, 
                                               fn_source_calib_csv)
            source_calib_csv = open(fn_source_calib_csv, 'wb')
            source_calib_writer = csvWriter(source_calib_csv, delimiter=',',
                                            quotechar='"', 
                                            quoting=csv.QUOTE_MINIMAL)

        # Prepare query for the source table
        col_list = ['source_id', 'process_id', 'scan_id', 'plate_id',
                    'archive_id']

        # Get source table columns from database schema
        source_table = self.get_table_dict('source')

        for k in source_table.keys():
            if k in sources.columns:
                col_list.append(k)

        source_columns = col_list
        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql_source = ('INSERT INTO {} ({}) VALUES ({})'
                      .format(self.table_name('source'), col_str, val_str))

        # Prepare query for the source_calib table
        col_list = ['source_id', 'process_id', 'scan_id', 'plate_id',
                    'archive_id']

        # Get source table columns from database schema
        source_calib_table = self.get_table_dict('source_calib')

        for k in source_calib_table.keys():
            if k in sources.columns:
                col_list.append(k)

        source_calib_columns = col_list
        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql_source_calib = ('INSERT INTO {} ({}) VALUES ({})'
                            .format(self.table_name('source_calib'), col_str,
                                    val_str))

        # Write header rows to CSV files
        if write_csv:
            source_writer.writerow(source_columns)
            source_calib_writer.writerow(source_calib_columns)

        # Prepare data and execute queries
        source_data = []
        source_calib_data = []

        for i, source in enumerate(sources):
            # Insert 1000 rows simultaneously
            if not write_csv and i > 0 and i%1000 == 0:
                self.db.executemany_query(sql_source, source_data)
                source_data = []
                self.db.executemany_query(sql_source_calib, source_calib_data)
                source_calib_data = []

            # Prepare source data
            source_id = process_id * 10000000 + i + 1
            val_tuple = (source_id, process_id, scan_id, plate_id, archive_id)

            for k in source_columns:
                if k in sources.columns:
                    source_val = (source[k] if np.isfinite(source[k]) 
                                  else None)
                    val_tuple = val_tuple + (source_val, )

            if write_csv:
                source_writer.writerow(val_tuple)
            else:
                source_data.append(val_tuple)

            # Prepare source_calib data
            val_tuple = (source_id, process_id, scan_id, plate_id, archive_id)

            for k in source_calib_columns:
                if k in sources.columns:
                    try:
                        source_val = (source[k] if np.isfinite(source[k])
                                      else None)
                    except TypeError:
                        source_val = source[k]

                    if 'healpix' in k and source_val < 0:
                        source_val = None
                        
                    if 'ucac4_id' in k and source_val == '':
                        source_val = None
                        
                    if 'tycho2_id' in k and source_val == '':
                        source_val = None

                    if 'tycho2_id_pad' in k and source_val == '':
                        source_val = None

                    if 'tycho2_hip' in k and source_val < 0:
                        source_val = None
                        
                    val_tuple = val_tuple + (source_val, )

            if write_csv:
                source_calib_writer.writerow(val_tuple)
            else:
                source_calib_data.append(val_tuple)

        if write_csv:
            # Close CSV files
            source_csv.close()
            source_calib_csv.close()
        else:
            # Insert remaining rows
            self.db.executemany_query(sql_source, source_data)
            self.db.executemany_query(sql_source_calib, source_calib_data)

    def write_process_start(self, scan_id=None, plate_id=None, 
                            archive_id=None, filename=None, use_psf=None):
        """
        Write plate-solve process to the database.

        """

        col_list = ['scan_id', 'plate_id', 'archive_id', 'filename',
                    'hostname', 'use_psf', 'pyplate_version']

        if use_psf:
            use_psf = 1
        else:
            use_psf = 0

        #hostname = platform.node()
        hostname = socket.gethostname()
            
        val_tuple = (scan_id, plate_id, archive_id, filename, hostname,
                     use_psf, __version__)
        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO {} ({}) VALUES ({}) RETURNING process_id'
               .format(self.table_name('process'), col_str, val_str))
        process_id = self.db.execute_query(sql, val_tuple)
        return process_id

    def update_process(self, process_id, **kwargs):
        """
        Update plate-solve process in the database.

        """

        col_list = []
        val_tuple = ()

        for k in kwargs:
            # Check if the keyword matches a column name in the process table
            # and if the keyword is not None
            if k in _schema['process'] and kwargs[k] is not None:
                col_list.append('{}=%s'.format(k))
                val_tuple = val_tuple + (kwargs[k], )

        # If no valid keywords are given, then give up
        if not col_list:
            return

        col_str = ','.join(col_list)
        sql = ('UPDATE {} SET {} WHERE process_id=%s'
               .format(self.table_name('process'), col_str))
        val_tuple = val_tuple + (process_id, )
        self.db.execute_query(sql, val_tuple)

    def write_process_end(self, process_id, completed=None, duration=None):
        """
        Write plate-solve process end to the database.

        """

        sql = ('UPDATE {} SET timestamp_end=NOW(),duration=%s,completed=%s '
               'WHERE process_id=%s'.format(self.table_name('process')))
        val_tuple = (duration, completed, process_id)
        self.db.execute_query(sql, val_tuple)

    def write_processlog(self, level, message, event=None, process_id=None,
                         scan_id=None, plate_id=None, archive_id=None):
        """
        Write plate solve process log message to the database.

        """

        col_list = ['process_id', 'scan_id', 'plate_id', 'archive_id',
                    'level', 'event', 'message']
        val_tuple = (process_id, scan_id, plate_id, archive_id,
                     level, event, message)
        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO {} ({}) VALUES ({})'
               .format(self.table_name('process_log'), col_str, val_str))
        self.db.execute_query(sql, val_tuple)

    def get_plate_id(self, plate_num, archive_id):
        """
        Get plate ID from the database by archive ID and plate number.

        Parameters
        ----------
        plate_num : str
            Plate number
        archive_id : int
            Archive ID

        Returns
        -------
        plate_id : int
            Plate ID

        """

        sql = ('SELECT plate_id FROM {} WHERE archive_id=%s AND plate_num=%s'
               .format(self.table_name('plate')))
        plate_id = self.db.execute_query(sql, (archive_id, plate_num))
        return plate_id

    def get_plate_id_wfpdb(self, wfpdb_id):
        """
        Get plate ID from the database by WFPDB ID.

        Parameters
        ----------
        wfpdb_id : str
            WFPDB ID

        Returns
        -------
        plate_id : int
            Plate ID

        """

        sql = ('SELECT plate_id FROM {} WHERE wfpdb_id=%s'
               .format(self.table_name('plate')))
        plate_id = self.db.execute_query(sql, (wfpdb_id,))
        return plate_id

    def get_scan_id(self, filename, archive_id):
        """
        Get scan ID and plate ID from the database.

        Parameters
        ----------
        filename : str
            Filename of the scan
        archive_id : int
            Archive ID number

        Returns
        -------
        result : tuple
            A tuple containing scan ID and plate ID

        """

        sql = ('SELECT scan_id, plate_id FROM {} '
               'WHERE filename_scan=%s AND archive_id=%s'
               .format(self.table_name('scan')))
        result = self.db.execute_query(sql, (filename, archive_id))

        if result is None:
            result = (None, None)

        return result

    def get_logbook_id(self, logbook_num, archive_id):
        """
        Get logbook ID from the database by archive ID and logbook number.

        Parameters
        ----------
        logbook_num : str
            Logbook number (unique within the specified archive)
        archive_id : int
            Archive ID

        Returns
        -------
        logbook_id : int
            Logbook ID

        """

        sql = ('SELECT logbook_id FROM {} '
               'WHERE archive_id=%s AND logbook_num=%s'
               .format(self.table_name('logbook')))
        logbook_id = self.db.execute_query(sql, (archive_id, logbook_num))
        return logbook_id

    def get_logpage_id(self, filename, archive_id):
        """
        Get logpage ID from the database.

        Parameters
        ----------
        filename : str
            Filename of the logpage
        archive_id : int
            Archive ID number

        Returns
        -------
        result : int
            Logpage ID number

        """

        sql = ('SELECT logpage_id FROM {} '
               'WHERE filename=%s AND archive_id=%s'
               .format(self.table_name('logpage')))
        logpage_id = self.db.execute_query(sql, (filename, archive_id))
        return logpage_id

    def get_plate_epoch(self, plate_id):
        """
        Get plate epoch from the database.

        Parameters
        ----------
        plate_id : int
            Plate ID number

        Returns
        -------
        result : float
            Plate epoch as a float

        """

        sql = ('SELECT year_start FROM {} WHERE plate_id=%s '
               'AND year_start IS NOT NULL ORDER BY year_start'
               .format(self.table_name('exposure')))
        result = self.db.execute_query(sql, (plate_id,))
        return result
