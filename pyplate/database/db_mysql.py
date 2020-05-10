import numpy as np
import time
import socket
import os
import csv
from collections import OrderedDict
from astropy.time import Time


from .config.local import ROOTDIR, SCHEMAFILE, RDBMS, FQTN
from .config.conf import read_conf
#from .config._version import __version__
from .db_yaml import fetch_ordered_tables, fetch_ordered_indexes

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

try:
    import pymysql
except ImportError:
    pass


# Special class for handling None (NULL) values when writing data to
# CSV files for later ingestion into the database
# http://stackoverflow.com/questions/11379300/csv-reader-behavior-with-none-and-empty-string
class csvWriter(object):
    def __init__(self, csvfile, *args, **kwrags):
        self.writer = csv.writer(csvfile, *args, **kwrags)

    def writerow(self, row):
        self.writer.writerow(['\\N' if val is None else val for val in row])

    def writerows(self, rows):
        map(self.writerow, rows)



yamlfile = SCHEMAFILE
_schema = OrderedDict()

_schema = fetch_ordered_tables(yamlfile,RDBMS,FQTN) 

def _export_scm():
    """
    return the OrderedDict
    """
    return _schema


class PlateDB:
    """
    Plate database class.

    """

    def __init__(self):
        self.host = 'localhost'
        self.port = '3306'
        self.user = ''
        self.password = ''
        self.database = ''

        self.db = None
        self.cursor = None

        self.write_db_source_dir = ''
        self.write_db_source_calib_dir = ''

### do we need this? 
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

        for attr in zip(['host', 'port', 'user', 'database', 'password'],
                        ['output_db_host', 'output_db_port', 'output_db_user',
                         'output_db_name', 'output_db_passwd']):
            try:
                setattr(self, attr[0], conf.get('Database', attr[1]))
            except configparser.Error:
                pass

    def open_connection(self, host=None, port=None, user=None, password=None, database=None):
        """
        Open mysql database connection.

        Parameters
        ----------

        host : str
            mysql database host name
        port : str
            mysql database port number
        user : str
            mysql database user name
        password : str
            mysql database password
        database : str
            mysql database name

        """

        if host is None:
            host = self.host

        if port is None:
            port = self.port

        if user is None:
            user = self.user

        if password is None:
            password = self.password

        # db (mysql) = schema (pg_sql)
        if database is None:
            database = self.database

        while True:
            try:
                self.db = pymysql.connect(host=host, port=port, user=user, password=password, 
                                          database=database)
                self.host = host
                self.port = port
                self.user = user
                self.password = password
                self.database = database

                break
            except pymysql.OperationalError as e:
                if (e.args[0] == 1040):
                    print('MySQL server reports too many connections, trying again')
                    print(e.args)
                    time.sleep(10)
                elif e.args[0] == 1045:
                    print('MySQL error {:d}: {}'.format(e.args[0], e.args[1]))
                    break
                else:
                    raise
        """
        Not yet really tested, whether these errors are those we want to check 
        """

        if self.db is not None:
            self.cursor = self.db.cursor()
            self.cursor.execute("SELECT version();")
            self.dbversion = self.cursor.fetchone()


    def execute_query(self, *args):
        """
        Execute SQL query and reopen connection if connection has been lost.

        """

        try:
            numrows = self.cursor.execute(*args)
        except AttributeError:
            numrows = None
        except pymysql.OperationalError as e:
            if e.args[0] == 2006:
                print('MySQL server has gone away, trying to reconnect')

                # Wait for 20 seconds, then open new connection and execute 
                # query again
                time.sleep(20)
                self.open_connection()
                numrows = self.cursor.execute(*args)
            else:
                raise

        return numrows

    def executemany_query(self, *args):
        """
        Execute SQL query with mutliple data rows and reopen connection if 
        connection has been lost.

        """

        try:
            numrows = self.cursor.executemany(*args)
        except AttributeError:
            numrows = None
        except pymysql.OperationalError as e:
            if e.args[0] == 2006:
                print('MySQL server has gone away, trying to reconnect')

                # Wait for 20 seconds, then open new connection and execute 
                # query again
                time.sleep(20)
                self.open_connection()
                numrows = self.cursor.executemany(*args)
            else:
                raise

        return numrows

    def close_connection(self):
        """
        Close MySQL database connection.

        """

        if self.db is not None:
            self.cursor.close()
            self.db.commit()
            self.db.close()
        
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

        # The plate table
        col_list = ['plate_id']

        if (isinstance(platemeta['db_plate_id'], int) and 
            (platemeta['db_plate_id'] > 0)):
            val_tuple = (platemeta['db_plate_id'],)
        else:
            val_tuple = (None,)

        for k,v in _schema['plate'].items():
            if v[1]:
                col_list.append(k)

                # Validate date type and insert NULL instead of invalid value
                if v[0] == 'DATE':
                    try:
                        d = Time(platemeta.get_value(v[1]), scale='tai')

                        if d >= Time('1000-01-01', scale='tai'):
                            val_tuple = (val_tuple 
                                         + (platemeta.get_value(v[1]),))
                        else:
                            val_tuple = val_tuple + (None,)
                    except ValueError:
                        val_tuple = val_tuple + (None,)
                else:
                    val_tuple = val_tuple + (platemeta.get_value(v[1]),)

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))

        sql = ('INSERT INTO plate ({}) VALUES ({})'
               .format(col_str, val_str))
        self.execute_query(sql, val_tuple)
        plate_id = self.cursor.lastrowid
        platemeta['db_plate_id'] = plate_id

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
            self.execute_query(sql, val_tuple)
            exposure_id = self.cursor.lastrowid

            # The exposure_sub table
            if len(platemeta['numsub']) > exp and platemeta['numsub'][exp] > 1:
                for subexp in np.arange(platemeta['numsub'][exp]):
                    subexp_num = subexp + 1
                    col_list = ['subexposure_id' ,'exposure_id', 'plate_id',
                                'exposure_num', 'subexposure_num']
                    val_tuple = (None, exposure_id, plate_id, exp_num, 
                                 subexp_num)

                    expmeta = platemeta.exposures[exp]

                    for k,v in _schema['exposure_sub'].items():
                        if v[1]:
                            col_list.append(k)
                            val_tuple = val_tuple \
                                    + (expmeta.get_value(v[1], exp=subexp), )

                    col_str = ','.join(col_list)
                    val_str = ','.join(['%s'] * len(col_list))

                    sql = ('INSERT INTO exposure_sub ({}) VALUES ({})'
                           .format(col_str, val_str))
                    self.execute_query(sql, val_tuple)

        return plate_id

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
                    sql = ('INSERT INTO plate_logpage ({}) VALUES ({})'
                           .format(col_str, val_str))
                    self.execute_query(sql, val_tuple)

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

        if (isinstance(platemeta['db_plate_id'], int) and 
            (platemeta['db_plate_id'] > 0)):
            plate_id = platemeta['db_plate_id']
        else:
            plate_id = self.get_plate_id(platemeta['plate_num'],
                                         platemeta['archive_id'])

            if plate_id is None:
                plate_id = self.get_plate_id_wfpdb(platemeta['wfpdb_id'])

        col_list = ['scan_id', 'plate_id']

        if (isinstance(platemeta['scan_id'], int) and 
            (platemeta['scan_id'] > 0)):
            val_tuple = (platemeta['scan_id'], plate_id)
        else:
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
        self.execute_query(sql, val_tuple)
        scan_id = self.cursor.lastrowid

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

        columns = [k for k,v in _schema['scan'].items() if v[1]]

        # Update only specific columns
        if filecols:
            columns = ['file_datetime', 'file_size', 'fits_checksum', 
                       'fits_datasum']

        for c in columns:
            c_str = '{}=%s'.format(c)
            col_list.append(c_str)
            platemeta_key = _schema['scan'][c][1]
            val_tuple = val_tuple + (platemeta.get_value(platemeta_key), )

        col_str = ','.join(col_list)
        val_tuple = val_tuple + (scan_id, )

        sql = 'UPDATE scan SET {} WHERE scan_id=%s'.format(col_str)
        self.execute_query(sql, val_tuple)

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

        col_list = ['preview_id', 'plate_id']

        if (isinstance(previewmeta['preview_id'], int) and 
            (previewmeta['preview_id'] > 0)):
            val_tuple = (previewmeta['preview_id'], plate_id)
        else:
            val_tuple = (None, plate_id)

        for k,v in _schema['preview'].items():
            if v[1]:
                col_list.append(k)
                val_tuple = val_tuple + (previewmeta[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))

        sql = ('INSERT INTO preview ({}) VALUES ({})'
               .format(col_str, val_str))
        self.execute_query(sql, val_tuple)
        preview_id = self.cursor.lastrowid

        return preview_id

    def write_logbook(self, logbookmeta):
        """
        Write a logbook to the database.

        """

        col_list = []
        val_tuple = ()

        for k,v in _schema['logbook'].items():
            if v[1]:
                col_list.append(k)
                val_tuple = val_tuple + (logbookmeta[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO logbook ({}) VALUES ({})'
               .format(col_str, val_str))
        self.execute_query(sql, val_tuple)
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
            
        col_list = []
        val_tuple = ()

        for k,v in _schema['logpage'].items():
            if v[1]:
                col_list.append(k)
                val_tuple = val_tuple + (logpagemeta[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO logpage ({}) VALUES ({})'
               .format(col_str, val_str))
        self.execute_query(sql, val_tuple)
        logpage_id = self.cursor.lastrowid
        return logpage_id

    def write_solution(self, solution, process_id=None, scan_id=None, 
                       plate_id=None, archive_id=None):
        """
        Write plate solution to the database.

        """

        col_list = ['solution_id', 'process_id', 'scan_id', 'exposure_id', 
                    'plate_id', 'archive_id']
        val_tuple = (None, process_id, scan_id, None, plate_id, archive_id)

        for k,v in _schema['solution'].items():
            if v[1]:
                col_list.append(k)
                val_tuple = val_tuple + (solution[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO solution ({}) VALUES ({})'
               .format(col_str, val_str))
        self.execute_query(sql, val_tuple)

    def write_astrom_sub(self, astrom_sub, process_id=None, scan_id=None, 
                       plate_id=None, archive_id=None):
        """
        Write astrometric sub-field calibration to the database.

        """

        col_list = ['sub_id', 'process_id', 'scan_id', 'exposure_id', 
                    'plate_id', 'archive_id']
        val_tuple = (None, process_id, scan_id, None, plate_id, archive_id)

        for k,v in _schema['astrom_sub'].items():
            if v[1]:
                col_list.append(k)
                val_tuple = val_tuple + (astrom_sub[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO astrom_sub ({}) VALUES ({})'
               .format(col_str, val_str))
        self.execute_query(sql, val_tuple)

    def write_phot_cterm(self, phot_cterm, process_id=None, scan_id=None, 
                         plate_id=None, archive_id=None):
        """
        Write photometric color term data to the database.

        """

        col_list = ['cterm_id', 'process_id', 'scan_id', 'exposure_id', 
                    'plate_id', 'archive_id']
        val_tuple = (None, process_id, scan_id, None, plate_id, archive_id)

        for k,v in _schema['phot_cterm'].items():
            if v[1]:
                col_list.append(k)
                val_tuple = val_tuple + (phot_cterm[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO phot_cterm ({}) VALUES ({})'
               .format(col_str, val_str))
        self.execute_query(sql, val_tuple)

    def write_phot_color(self, phot_color, process_id=None, scan_id=None, 
                         plate_id=None, archive_id=None):
        """
        Write photometric color term result to the database.

        """

        col_list = ['color_id', 'process_id', 'scan_id', 'exposure_id', 
                    'plate_id', 'archive_id']
        val_tuple = (None, process_id, scan_id, None, plate_id, archive_id)

        for k,v in _schema['phot_color'].items():
            if v[1]:
                col_list.append(k)
                val_tuple = val_tuple + (phot_color[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO phot_color ({}) VALUES ({})'
               .format(col_str, val_str))
        self.execute_query(sql, val_tuple)

    def write_phot_calib(self, phot_calib, process_id=None, scan_id=None, 
                         plate_id=None, archive_id=None):
        """
        Write photometric calibration to the database.

        """

        col_list = ['calib_id', 'process_id', 'scan_id', 'exposure_id', 
                    'plate_id', 'archive_id']
        val_tuple = (None, process_id, scan_id, None, plate_id, archive_id)

        for k,v in _schema['phot_calib'].items():
            if v[1]:
                col_list.append(k)
                val_tuple = val_tuple + (phot_calib[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO phot_calib ({}) VALUES ({})'
               .format(col_str, val_str))
        self.execute_query(sql, val_tuple)

    def write_phot_sub(self, phot_sub, process_id=None, scan_id=None, 
                       plate_id=None, archive_id=None):
        """
        Write photometric sub-field calibration to the database.

        """

        col_list = ['sub_id', 'process_id', 'scan_id', 'exposure_id', 
                    'plate_id', 'archive_id']
        val_tuple = (None, process_id, scan_id, None, plate_id, archive_id)

        for k,v in _schema['phot_sub'].items():
            if v[1]:
                col_list.append(k)
                val_tuple = val_tuple + (phot_sub[k], )

        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO phot_sub ({}) VALUES ({})'
               .format(col_str, val_str))
        self.execute_query(sql, val_tuple)

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
        col_list = ['source_id', 'process_id', 'scan_id', 'exposure_id', 
                    'plate_id', 'archive_id']
        for k,v in _schema['source'].items():
            if v[1]:
                col_list.append(k)

        source_columns = col_list
        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql_source = ('INSERT INTO source ({}) VALUES ({})'
                      .format(col_str, val_str))

        # Prepare query for the source_calib table
        col_list = ['source_id', 'process_id', 'scan_id', 'exposure_id', 
                    'plate_id', 'archive_id']

        for k,v in _schema['source_calib'].items():
            if v[1]:
                col_list.append(k)

        source_calib_columns = col_list
        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql_source_calib = ('INSERT INTO source_calib ({}) VALUES ({})'
                            .format(col_str, val_str))

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
                self.executemany_query(sql_source, source_data)
                source_data = []
                self.executemany_query(sql_source_calib, source_calib_data)
                source_calib_data = []

            # Prepare source data
            source_id = process_id * 10000000 + i + 1
            val_tuple = (source_id, process_id, scan_id, None, plate_id, 
                         archive_id)

            for k,v in _schema['source'].items():
                if v[1]:
                    source_val = (source[k] if np.isfinite(source[k]) 
                                  else None)
                    val_tuple = val_tuple + (source_val, )

            if write_csv:
                source_writer.writerow(val_tuple)
            else:
                source_data.append(val_tuple)

            # Prepare source_calib data
            val_tuple = (source_id, process_id, scan_id, None, plate_id, 
                         archive_id)

            for k,v in _schema['source_calib'].items():
                if v[1]:
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
            self.executemany_query(sql_source, source_data)
            self.executemany_query(sql_source_calib, source_calib_data)

    def write_process_start(self, scan_id=None, plate_id=None, 
                            archive_id=None, filename=None, use_psf=None):
        """
        Write plate-solve process to the database.

        """

        col_list = ['process_id', 'scan_id', 'plate_id', 'archive_id', 
                    'filename', 'hostname', 'timestamp_start', 'use_psf', 
                    'pyplate_version']

        if use_psf:
            use_psf = 1
        else:
            use_psf = 0

        #hostname = platform.node()
        hostname = socket.gethostname()
            
        val_tuple = (None, scan_id, plate_id, archive_id, filename, hostname, 
                     None, use_psf, __version__)
        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO process ({}) VALUES ({})'
               .format(col_str, val_str))
        numrows = self.execute_query(sql, val_tuple)

        if numrows is not None:
            process_id = self.cursor.lastrowid
        else:
            process_id = None

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
        sql = ('UPDATE process SET {} WHERE process_id=%s'.format(col_str))
        val_tuple = val_tuple + (process_id, )
        self.execute_query(sql, val_tuple)

    def write_process_end(self, process_id, completed=None, duration=None):
        """
        Write plate-solve process end to the database.

        """

        sql = ('UPDATE process '
               'SET timestamp_end=NOW(),duration=%s,completed=%s '
               'WHERE process_id=%s')
        val_tuple = (duration, completed, process_id)
        self.execute_query(sql, val_tuple)

    def write_processlog(self, level, message, event=None, process_id=None,
                         scan_id=None, plate_id=None, archive_id=None):
        """
        Write plate solve process log message to the database.

        """

        col_list = ['processlog_id', 'process_id', 'timestamp_log', 
                    'scan_id', 'plate_id', 'archive_id', 
                    'level', 'event', 'message']
        val_tuple = (None, process_id, None, scan_id, plate_id, archive_id, 
                     level, event, message)
        col_str = ','.join(col_list)
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('INSERT INTO process_log ({}) VALUES ({})'
               .format(col_str, val_str))
        self.execute_query(sql, val_tuple)

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
        numrows = self.execute_query(sql, (archive_id,plate_num))

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

        sql = 'SELECT plate_id FROM plate WHERE wfpdb_id=%s'
        numrows = self.execute_query(sql, (wfpdb_id,))

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
        numrows = self.execute_query(sql, (filename,archive_id))

        if numrows == 1:
            result = self.cursor.fetchone()
        else:
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

        sql = ('SELECT logbook_id FROM logbook '
               'WHERE archive_id=%s AND logbook_num=%s')
        numrows = self.execute_query(sql, (archive_id,logbook_num))

        if numrows == 1:
            result = self.cursor.fetchone()
            logbook_id = result[0]
        else:
            logbook_id = None

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
        numrows = self.execute_query(sql, (filename, archive_id))

        if numrows == 1:
            result = self.cursor.fetchone()[0]
        else:
            result = None

        return result

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

        sql = ('SELECT year_start FROM exposure WHERE plate_id=%s '
               'AND year_start IS NOT NULL ORDER BY year_start')
        numrows = self.execute_query(sql, (plate_id,))

        if numrows > 0:
            result = self.cursor.fetchone()[0]
        else:
            result = None

        return result

