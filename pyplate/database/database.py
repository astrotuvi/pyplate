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
        self.conf = None
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
            path_yaml = os.path.join(os.path.dirname(__file__), fn_yaml)
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
            self.db.assign_conf(self.conf)
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

    def create_schema(self):
        """Create database schema"""

        if self.db is not None:
            sql = self.db.get_schema_sql(mode='create_schema')
            self.db.execute_query(sql)

    def drop_schema(self):
        """Drop database schema"""

        if self.db is not None:
            sql = self.db.get_schema_sql(mode='drop_schema')
            self.db.execute_query(sql)

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

        # Get process table columns from database schema
        process_table = self.get_table_dict('process')

        for k in kwargs:
            # Check if the keyword matches a column name in the process table
            # and if the keyword is not None
            if k in process_table and kwargs[k] is not None:
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
