import numpy as np
from collections import OrderedDict
from conf import read_conf

try:
    import MySQLdb
except ImportError:
    pass

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
    ('spectral_band',    ('VARCHAR(80)', 'spectral_band')),
    ('develop',          ('VARCHAR(80)', 'developing')),
    ('plate_quality',    ('VARCHAR(80)', 'plate_quality')),
    ('plate_notes',      ('VARCHAR(255)', 'plate_notes')),
    ('date_orig',        ('DATE', 'date_orig')),
    ('numexp',           ('TINYINT UNSIGNED', 'numexp')),
    ('observatory',      ('VARCHAR(80)', 'observatory')),
    ('sitename',         ('VARCHAR(80)', 'site_name')),
    ('longitude_str',    ('CHAR(12)', None)),
    ('latitude_str',     ('CHAR(12)', None)),
    ('longitude_deg',    ('DOUBLE', 'site_longitude')),
    ('latitude_deg',     ('DOUBLE', 'site_latitude')),
    ('elevation',        ('FLOAT', 'site_elevation')),
    ('telescope',        ('VARCHAR(80)', 'telescope')),
    ('tel_aper',         ('FLOAT', 'tel_aperture')),
    ('tel_foc',          ('FLOAT', 'tel_foclength')),
    ('tel_scale',        ('FLOAT', 'tel_scale')),
    ('plate_fov1',       ('FLOAT', None)),
    ('plate_fov2',       ('FLOAT', None)),
    ('instrument',       ('VARCHAR(80)', 'instrument')),
    ('method',           ('TINYINT UNSIGNED', 'method_code')),
    ('prism',            ('VARCHAR(80)', 'prism')),
    ('prism_angle',      ('VARCHAR(10)', 'prism_angle')),
    ('dispersion',       ('FLOAT', 'dispersion')),
    ('grating',          ('VARCHAR(80)', 'grating')),
    ('focus',            ('FLOAT', None)),
    ('air_temp',         ('FLOAT', 'temperature')),
    ('calmness',         ('CHAR(3)', 'calmness')),
    ('sharpness',        ('CHAR(3)', 'sharpness')),
    ('transparency',     ('CHAR(3)', 'transparency')),
    ('sky_conditions',   ('VARCHAR(80)', 'skycond')),
    ('observer',         ('VARCHAR(80)', 'observer')),
    ('obs_notes',        ('VARCHAR(255)', 'obsnotes')),
    ('notes',            ('VARCHAR(255)', 'notes')),
    ('bibcode',          ('VARCHAR(80)', 'bibcode')),
    ('filename_preview', ('VARCHAR(80)', 'fn_pre')),
    ('filename_cover',   ('VARCHAR(80)', 'fn_cover')),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX archive_ind', ('(archive_id)', None)),
    ('INDEX wfpdb_ind',   ('(wfpdb_id)', None)),
    ('INDEX method_ind',  ('(method)', None))
    ])

_schema['exposure'] = OrderedDict([
    ('exposure_id',      ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', None)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', 'plate_id')),
    ('archive_id',       ('INT UNSIGNED NOT NULL', 'archive_id')),
    ('exposure_num',     ('TINYINT UNSIGNED NOT NULL', None)),
    ('object_name',      ('VARCHAR(80)', 'object_name')),
    ('object_type',      ('CHAR(2)', 'object_type_code')),
    ('ra_orig',          ('CHAR(11)', 'ra_orig')),
    ('dec_orig',         ('CHAR(11)', 'dec_orig')),
    ('flag_coord',       ('CHAR(1)', 'coord_flag')),
    ('raj2000',          ('DOUBLE', 'ra_deg')),
    ('dej2000',          ('DOUBLE', 'dec_deg')),
    ('raj2000_hms',      ('CHAR(11)', 'ra')),
    ('dej2000_dms',      ('CHAR(11)', 'dec')),
    ('flag_wcs',         ('TINYINT UNSIGNED', None)),
    ('fov1',             ('FLOAT', None)),
    ('fov2',             ('FLOAT', None)),
    ('footprint',        ('VARCHAR(200)', None)),
    ('date_orig_start',  ('VARCHAR(10)', 'date_orig')),
    ('date_orig_end',    ('VARCHAR(10)', None)),
    ('time_orig_start',  ('VARCHAR(40)', 'tms_orig')),
    ('time_orig_end',    ('VARCHAR(40)', 'tme_orig')),
    ('flag_time',        ('CHAR(1)', 'time_flag')),
    ('ut_start',         ('DATETIME', 'date_obs')),
    ('ut_mid',           ('DATETIME', 'date_avg')),
    ('ut_weighted',      ('DATETIME', None)),
    ('ut_end',           ('DATETIME', 'date_end')),
    ('year_start',       ('DOUBLE', 'year')),
    ('year_mid',         ('DOUBLE', 'year_avg')),
    ('year_weighted',    ('DOUBLE', None)),
    ('year_end',         ('DOUBLE', 'year_end')),
    ('jd_start',         ('DOUBLE', 'jd')),
    ('jd_mid',           ('DOUBLE', 'jd_avg')),
    ('jd_weighted',      ('DOUBLE', None)),
    ('jd_end',           ('DOUBLE', 'jd_end')),
    ('hjd_mid',          ('DOUBLE', None)),
    ('hjd_weighted',     ('DOUBLE', None)),
    ('exptime',          ('FLOAT', 'exptime')),
    ('num_sub',          ('TINYINT UNSIGNED', None)),
    ('method',           ('TINYINT UNSIGNED', None)),
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
    ('exposure_id',      ('INT UNSIGNED NOT NULL', None)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', None)),
    ('exposure_num',     ('TINYINT UNSIGNED NOT NULL', None)),
    ('subexposure_num',  ('TINYINT UNSIGNED NOT NULL', None)),
    ('date_orig_start',  ('VARCHAR(10)', None)),
    ('time_orig_start',  ('VARCHAR(40)', None)),
    ('time_orig_end',    ('VARCHAR(40)', None)),
    ('ut_start',         ('DATETIME', None)),
    ('ut_mid',           ('DATETIME', None)),
    ('ut_end',           ('DATETIME', None)),
    ('jd_start',         ('DOUBLE', None)),
    ('jd_mid',           ('DOUBLE', None)),
    ('jd_end',           ('DOUBLE', None)),
    ('exptime',          ('FLOAT', None)),
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
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX plate_ind',   ('(plate_id)', None)),
    ('INDEX archive_ind', ('(archive_id)', None))
    ])

_schema['logbook'] = OrderedDict([
    ('logbook_id',       ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', 
                          False)),
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
                          False)),
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
    ('source_id',        ('INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY', 
                          False)),
    ('scan_id',          ('INT UNSIGNED NOT NULL', False)),
    ('exposure_id',      ('INT UNSIGNED', False)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', False)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', False)),
    ('source_num',       ('INT UNSIGNED', True)),
    ('x_source',         ('DOUBLE', True)),
    ('y_source',         ('DOUBLE', True)),
    ('erra_source',      ('FLOAT', True)),
    ('errb_source',      ('FLOAT', True)),
    ('errtheta_source',  ('FLOAT', True)),
    ('a_source',         ('FLOAT', True)),
    ('b_source',         ('FLOAT', True)),
    ('theta_source',     ('FLOAT', True)),
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
    ('INDEX plate_ind',      ('(plate_id)', None)),
    ('INDEX archive_ind',    ('(archive_id)', None)),
    ('INDEX exposure_ind',   ('(exposure_id)', None)),
    ('INDEX scan_ind',       ('(scan_id)', None)),
    ('INDEX annularbin_ind', ('(annular_bin)', None))
    ])

_schema['source_calib'] = OrderedDict([
    ('source_id',        ('INT UNSIGNED NOT NULL PRIMARY KEY', False)),
    ('scan_id',          ('INT UNSIGNED NOT NULL', False)),
    ('exposure_id',      ('INT UNSIGNED', False)),
    ('plate_id',         ('INT UNSIGNED NOT NULL', False)),
    ('archive_id',       ('INT UNSIGNED NOT NULL', False)),
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
    ('gridsize_sub',     ('SMALLINT', True)),
    ('bmag',             ('FLOAT', False)),
    ('err_bmag',         ('FLOAT', False)),
    ('vmag',             ('FLOAT', False)),
    ('err_vmag',         ('FLOAT', False)),
    ('tycho2_id',        ('CHAR(12)', False)),
    ('tycho2_btmag',     ('FLOAT', False)),
    ('tycho2_vtmag',     ('FLOAT', False)),
    ('ucac4_id',         ('CHAR(10)', True)),
    ('ucac4_bmag',       ('FLOAT', True)),
    ('ucac4_vmag',       ('FLOAT', True)),
    ('timestamp_insert', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', None)),
    ('timestamp_update', ('TIMESTAMP DEFAULT CURRENT_TIMESTAMP '
                          'ON UPDATE CURRENT_TIMESTAMP', None)),
    ('INDEX plate_ind',    ('(plate_id)', None)),
    ('INDEX archive_ind',  ('(archive_id)', None)),
    ('INDEX exposure_ind', ('(exposure_id)', None)),
    ('INDEX scan_ind',     ('(scan_id)', None)),
    ('INDEX raj2000_ind',  ('(raj2000)', None)),
    ('INDEX dej2000_ind',  ('(dej2000)', None)),
    ('INDEX x_ind',        ('(x_sphere)', None)),
    ('INDEX y_ind',        ('(y_sphere)', None)),
    ('INDEX z_ind',        ('(z_sphere)', None)),
    ('INDEX healpix256_ind', ('(healpix256)', None)),
    ('INDEX tycho2_ind',   ('(tycho2_id)', None)),
    ('INDEX ucac4_ind',    ('(ucac4_id)', None))
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

def print_tables(use_drop=False):
    """
    Print table creation SQL queries to standard output.

    """

    sql_drop = '\n'.join(['DROP TABLE IF EXISTS {};'.format(k) 
                          for k in _schema.keys()])

    sql_list = ['CREATE TABLE {} (\n{}\n) ENGINE=MyISAM '
                'CHARACTER SET=utf8 COLLATE=utf8_unicode_ci;\n'
                .format(k, _get_columns_sql(k))
                for k in _schema.keys()]
    sql = '\n'.join(sql_list)

    if use_drop:
        sql = sql_drop + '\n\n' + sql

    print sql


class PlateDB:
    """
    Plate database class.

    """

    def __init__(self):
        self.host = 'localhost'
        self.user = ''
        self.dbname = ''
        self.passwd = ''

        self.db = None
        self.cursor = None

    def assign_conf(self, conf):
        """
        Assign and parse configuration.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        self.conf = conf

        for attr in ['write_log_dir']:
            try:
                setattr(self, attr, conf.get('Files', attr))
            except ConfigParser.Error:
                pass

        for attr in zip(['host', 'user', 'dbname', 'passwd'],
                        ['output_db_host', 'output_db_user',
                         'output_db_name', 'output_db_passwd']):
            try:
                setattr(self, attr[0], conf.get('Database', attr[1]))
            except ConfigParser.Error:
                pass

    def open_connection(self, host=None, user=None, passwd=None, dbname=None):
        """
        Open MySQL database connection.

        Parameters
        ----------
        host : str
            MySQL database host name
        user : str
            MySQL database user name
        passwd : str
            MySQL database password
        dbname : str
            MySQL database name

        """

        if host is None:
            host = self.host

        if user is None:
            user = self.user

        if passwd is None:
            passwd = self.passwd

        if dbname is None:
            dbname = self.dbname

        self.db = MySQLdb.connect(host=host, user=user, passwd=passwd, 
                                  db=dbname)
        self.cursor = self.db.cursor()

    def close_connection(self):
        """
        Close MySQL database connection.

        """

        self.cursor.close()
        self.db.commit()
        self.db.close()
        
    def write_plate(self, platemeta):
        """
        Write plate entry to the database.

        Parameters
        ----------
        platemeta : PlateMeta
            Plate metadata instance

        Returns
        -------
        plate_id : int
            Plate ID number

        """

        # The plate table
        col_list = ['plate_id']
        val_tuple = (None,)

        for k,v in _schema['plate'].items():
            if v[1]:
                col_list.append(k)
                val_tuple = val_tuple \
                        + (platemeta.get_value(v[1]), )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))

        sql = ('INSERT INTO plate ({}) VALUES ({})'
               .format(col_str, val_str))
        self.cursor.execute(sql, val_tuple)
        plate_id = self.cursor.lastrowid
        platemeta['plate_id'] = plate_id

        # The exposure table
        for exp in np.arange(platemeta['numexp']):
            exp_num = exp + 1
            col_list = ['exposure_id', 'exposure_num']
            val_tuple = (None, exp_num)

            for k,v in _schema['exposure'].items():
                if v[1]:
                    col_list.append(k)
                    val_tuple = val_tuple \
                            + (platemeta.get_value(v[1], exp=exp), )

            col_str = ','.join(col_list)
            val_str = ','.join(['%s'] * len(col_list))

            sql = ('INSERT INTO exposure ({}) VALUES ({})'
                   .format(col_str, val_str))
            self.cursor.execute(sql, val_tuple)
            exposure_id = self.cursor.lastrowid

        return plate_id

    def write_plate_logpage(self, platemeta):
        """
        Write plate-logpage relations to the database.

        Parameters
        ----------
        platemeta : PlateMeta
            Plate metadata instance

        """

        fn_list = [platemeta['fn_cover']]
        fn_list.extend(platemeta['fn_log'])

        for order,filename in enumerate(fn_list):
            if filename:
                col_list = ['plate_id', 'logpage_id', 'logpage_order']
                plate_id = self.get_plate_id(platemeta['plate_num'],
                                             platemeta['archive_id'])
                logpage_id = self.get_logpage_id(filename, 
                                                 platemeta['archive_id'])

                if plate_id and logpage_id:
                    val_tuple = (plate_id, logpage_id, order)
                    col_str = ','.join(col_list)
                    val_str = ','.join(['%s'] * len(col_list))
                    sql = ('INSERT INTO plate_logpage ({}) VALUES ({})'
                           .format(col_str, val_str))
                    self.cursor.execute(sql, val_tuple)

    def write_scan(self, platemeta):
        """
        Write scan entry to the database.

        Parameters
        ----------
        platemeta : PlateMeta
            Plate metadata instance

        Returns
        -------
        scan_id : int
            Scan ID number

        """

        plate_id = self.get_plate_id(platemeta['plate_num'],
                                     platemeta['archive_id'])

        if plate_id is None:
            plate_id = self.get_plate_id_wfpdb(platemeta['wfpdb_id'])

        col_list = ['scan_id', 'plate_id']
        val_tuple = (None, plate_id)

        for k,v in _schema['scan'].items():
            if v[1]:
                col_list.append(k)
                val_tuple = val_tuple \
                        + (platemeta.get_value(v[1]), )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))

        sql = ('INSERT INTO scan ({}) VALUES ({})'
               .format(col_str, val_str))
        self.cursor.execute(sql, val_tuple)
        scan_id = self.cursor.lastrowid

        return scan_id

    def write_logbook(self, logbookmeta):
        """
        Write a logbook to the database.

        """

        col_list = ['logbook_id']
        val_tuple = (None, )

        for k,v in _schema['logbook'].items():
            if v[1]:
                col_list.append(k)
                val_tuple = val_tuple + (logbookmeta[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO logbook ({}) VALUES ({})'
               .format(col_str, val_str))
        self.cursor.execute(sql, val_tuple)
        logbook_id = self.cursor.lastrowid
        return logbook_id

    def write_logpage(self, logpagemeta):
        """
        Write a single logpage to the database.

        """

        if logpagemeta['logbook_id'] is None:
            logbook_id = self.get_logbook_id(logpagemeta['logbook_num'],
                                             logpagemeta['archive_id'])
            logpagemeta['logbook_id'] = logbook_id
            
        col_list = ['logpage_id']
        val_tuple = (None, )

        for k,v in _schema['logpage'].items():
            if v[1]:
                col_list.append(k)
                val_tuple = val_tuple + (logpagemeta[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO logpage ({}) VALUES ({})'
               .format(col_str, val_str))
        self.cursor.execute(sql, val_tuple)
        logpage_id = self.cursor.lastrowid
        return logpage_id

    def write_sources(self, sources, scan_id=None, plate_id=None, 
                      archive_id=None):
        """
        Write source list with calibrated RA and Dec to the database.

        """

        for i in np.arange(len(sources)):
            col_list = ['source_id', 'plate_id', 'archive_id', 'exposure_id', 
                        'scan_id']
            val_tuple = (None, plate_id, archive_id, None, scan_id)

            for k,v in _schema['source'].items():
                if v[1]:
                    col_list.append(k)
                    source_val = (sources[i][k] if np.isfinite(sources[i][k]) 
                                  else None)
                    val_tuple = val_tuple + (source_val, )

            col_str = ','.join(col_list)
            val_str = ','.join(['%s'] * len(col_list))
            sql = ('INSERT INTO source ({}) VALUES ({})'
                   .format(col_str, val_str))
            self.cursor.execute(sql, val_tuple)
            source_id = self.cursor.lastrowid

            col_list = ['source_id', 'plate_id', 'archive_id', 'exposure_id', 
                        'scan_id']
            val_tuple = (source_id, plate_id, archive_id, None, scan_id)

            for k,v in _schema['source_calib'].items():
                if v[1]:
                    col_list.append(k)

                    try:
                        source_val = (sources[i][k] 
                                      if np.isfinite(sources[i][k])
                                      else None)
                    except TypeError:
                        source_val = sources[i][k]

                    if 'healpix' in k and source_val < 0:
                        source_val = None
                        
                    if 'ucac4_id' in k and source_val == '':
                        source_val = None
                        
                    val_tuple = val_tuple + (source_val, )

            col_str = ','.join(col_list)
            val_str = ','.join(['%s'] * len(col_list))
            sql = ('INSERT INTO source_calib ({}) VALUES ({})'
                   .format(col_str, val_str))
            self.cursor.execute(sql, val_tuple)

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

        sql = ('SELECT plate_id FROM plate '
               'WHERE archive_id=%s AND plate_num=%s')
        numrows = self.cursor.execute(sql, (archive_id,plate_num))

        if numrows == 1:
            result = self.cursor.fetchone()
            plate_id = result[0]
        else:
            return None

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

        sql = ('SELECT plate_id FROM plate '
               'WHERE wfpdb_id=%s')
        numrows = self.cursor.execute(sql, (wfpdb_id,))

        if numrows == 1:
            result = self.cursor.fetchone()
            plate_id = result[0]
        else:
            return None

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

        sql = ('SELECT scan_id, plate_id FROM scan '
               'WHERE filename_scan=%s AND archive_id=%s')
        numrows = self.cursor.execute(sql, (filename,archive_id))

        if numrows == 1:
            result = self.cursor.fetchone()
        else:
            return (None, None)

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

        sql = ('SELECT logbook_id FROM logbook '
               'WHERE archive_id=%s AND logbook_num=%s')
        numrows = self.cursor.execute(sql, (archive_id,logbook_num))

        if numrows == 1:
            result = self.cursor.fetchone()
            logbook_id = result[0]
        else:
            return None

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

        sql = ('SELECT logpage_id FROM logpage '
               'WHERE filename=%s AND archive_id=%s')
        numrows = self.cursor.execute(sql, (filename, archive_id))

        if numrows == 1:
            result = self.cursor.fetchone()[0]
        else:
            return None

        return result


