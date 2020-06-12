import numpy as np
import time
import socket
import os
import csv
from collections import OrderedDict
from astropy.time import Time
from ..conf import read_conf
from .._version import __version__
from .db_yaml import (fetch_ordered_tables, fetch_ordered_indexes,
                      creat_schema_pgsql, creat_schema_index)

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

try:
    import psycopg2
    from psycopg2.extensions import register_adapter, AsIs
except ImportError:
    pass


class DB_pgsql:
    """
    PostgreSQL database class.

    """

    def __init__(self):
        self.host = 'localhost'
        self.port = '5432'
        self.user = ''
        self.password = ''
        self.database = ''
        self.schema = ''

        self.schema_dict = None
        self.trigger_dict = None
        self.index_dict = None

        self.db = None
        self.cursor = None

        self.write_db_source_dir = ''
        self.write_db_source_calib_dir = ''

        # Register AsIs adapter for numpy data types
        # Credit: https://github.com/musically-ut/psycopg2_numpy_ext
        for t in [np.int8, np.int16, np.int32, np.int64,
                  np.float32, np.float64]:
            register_adapter(t, AsIs)

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

        for attr in zip(['host', 'port', 'user', 'database', 'password',
                         'schema'],
                        ['output_db_host', 'output_db_port', 'output_db_user',
                         'output_db_name', 'output_db_passwd',
                         'output_db_schema']):
            try:
                setattr(self, attr[0], conf.get('Database', attr[1]))
            except configparser.Error:
                pass

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
            d1, d2 = fetch_ordered_tables(path_yaml, 'pgsql', True)
            self.schema_dict = d1
            self.trigger_dict = d2
            self.index_dict = fetch_ordered_indexes(path_yaml, 'pgsql', True)

    def get_schema_sql(self, schema=None, mode='create_schema'):
        """
        Return schema creation or drop SQL statements.

        Parameters
        ----------
        schema : str
            Database schema
        mode : str
            Controls which statements to return ('create_schema',
            'drop_schema', 'create_indexes', 'drop_indexes')
        """

        self.read_schema(schema=schema)

        pdict = OrderedDict()
        creat_schema_pgsql(self.schema_dict, self.trigger_dict, pdict)
        creat_schema_index(self.index_dict, pdict)

        if mode in pdict:
            return pdict[mode]
        else:
            return ''

    def open_connection(self, host=None, port=None, user=None, password=None, database=None):
        """
        Open pgsql database connection.

        Parameters
        ----------

        host : str
            pgsql database host name
        port : int
            pgsql database port number
        user : str
            pgsql database user name
        password : str
            pgsql database password
        database : str
            pgsql database name

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

        while True:
            try:
                self.db = psycopg2.connect(user=user, password=password,
                                           host=host, port=port,
                                           database=database)
                self.host = host
                self.port = port
                self.user = user
                self.password = password
                self.database = database

                break
            except (Exception, psycopg2.Error) as e:
                if (e.args):
#                if (e.args[0] == 1040):
                    print('pgsql server reports exception,  trying again')
                    print(e.args)
                    time.sleep(10)
                elif (e.args[1]):
                    print('pgsql error {:d}: {}'.format(e.args[0], e.args[1]))
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

        # Default return value
        val = None

        try:
            self.cursor.execute(*args)

            if 'RETURNING' in args[0] or args[0].startswith('SELECT'):
                val = self.cursor.fetchone()

                if val is not None:
                    if len(val) == 1:
                        val = val[0]

            self.db.commit()
        except AttributeError:
            pass
        except (Exception, psycopg2.OperationalError) as e:
            raise

            if (e.args):
#            if (e.args[0] == 08006):
                print('pgsql server has xception, trying to reconnect')
                print(e.args)

                # Wait for 20 seconds, then open new connection and execute 
                # query again
                time.sleep(20)
                self.open_connection()
                self.cursor.execute(*args)
            else:
                raise

        """
        Not yet really tested, whether these errors are those we want to check 
        """

        return val

    def executemany_query(self, *args):
        """
        Execute SQL query with mutliple data rows and reopen connection if 
        connection has been lost.

        """

        try:
            self.cursor.executemany(*args)
            self.db.commit()
        except AttributeError:
            pass
        except (Exception, psycopg2.Error) as e:
            raise

            if (e.args[0] == 2006):
                print('pgsql server has gone away, trying to reconnect')

                # Wait for 20 seconds, then open new connection and execute 
                # query again
                time.sleep(20)
                self.open_connection()
                self.cursor.executemany(*args)
            else:
                raise

        """
        Not yet really tested, whether these errors are those we want to check 
        """
        return None

    def close_connection(self):
        """
        Close pgsql database connection.

        """

        if self.db is not None:
            self.cursor.close()
            self.db.commit()
            self.db.close()
