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

    def __init__(self, **kwargs):
        self.host = 'localhost'
        self.port = '5432'
        self.user = ''
        self.password = ''
        self.database = ''
        self.schema = kwargs.pop('schema', None)
        self.yaml = kwargs.pop('yaml', None)

        self.schema_dict = None
        self.trigger_dict = None
        self.index_dict = None

        self.db = None
        self.cursor = None

        # Register AsIs adapter for numpy data types
        # Credit: https://github.com/musically-ut/psycopg2_numpy_ext
        for t in [np.int8, np.int16, np.int32, np.int64,
                  np.float32, np.float64]:
            register_adapter(t, AsIs)

    def assign_conf(self, conf, section='DB_pgsql'):
        """
        Assign and parse configuration.

        Parameters
        ----------
        section : str
            Configuration file section to be read from

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        self.conf = conf

        for attr in ['host', 'user', 'password', 'database', 'schema', 'yaml']:
            try:
                setattr(self, attr, conf.get(section, attr))
            except configparser.Error:
                pass

            if attr == 'schema' and self.schema == '':
                self.schema = None

        for attr in ['port']:
            try:
                setattr(self, attr, conf.getint(section, attr))
            except configparser.Error:
                pass

    def read_schema(self, schema=None, yaml=None):
        """Read schema from schema YAML file.

        Parameters
        ----------
        schema : str
            Database schema name
        yaml : str
            Schema YAML file
        """

        if schema is None:
            schema = self.schema

        if yaml is None:
            yaml = self.yaml

        if yaml is None and schema is not None:
            yaml = '{}.yaml'.format(self.schema)

        path_yaml = os.path.join(os.path.dirname(__file__), yaml)
        d1, d2 = fetch_ordered_tables(path_yaml, 'pgsql', True,
                                      new_name=schema)
        self.schema_dict = d1
        self.trigger_dict = d2
        self.index_dict = fetch_ordered_indexes(path_yaml, 'pgsql', True,
                                                new_name=schema)

    def get_schema_sql(self, schema=None, table=None, mode='create_schema'):
        """
        Return schema creation or drop SQL statements.

        Parameters
        ----------
        schema : str
            Database schema
        table : str
            Table name, if only single table statement is required
        mode : str
            Controls which statements to return ('create_schema',
            'drop_schema', 'create_indexes', 'drop_indexes')
        """

        if (self.schema_dict is None or self.trigger_dict is None or
            self.index_dict is None):
            self.read_schema(schema=schema)

        if table is None:
            schema_dict = self.schema_dict.copy()
            trigger_dict = self.trigger_dict.copy()
        else:
            schema_table = '{}.{}'.format(self.schema_dict['schema'], table)
            schema_dict = OrderedDict()

            if schema_table in self.schema_dict:
                schema_dict[schema_table] = self.schema_dict[schema_table]

            trigger_dict = OrderedDict()

            if schema_table in self.trigger_dict:
                trigger_dict[schema_table] = self.trigger_dict[schema_table]

        pdict = OrderedDict()
        creat_schema_pgsql(schema_dict, trigger_dict, pdict)
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

    def execute_select_query(self, *args):
        """
        Execute SQL SELECT query and return all results.

        Parameters
        ----------
        args : tuple
            Query arguments that will be passed to the database cursor
        """

        self.cursor.execute(*args)
        val = self.cursor.fetchall()
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
