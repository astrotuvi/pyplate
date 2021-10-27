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
                      creat_schema_mysql, creat_schema_index)

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

try:
    import pymysql
except ImportError:
    pass


class DB_mysql:
    """
    MySQL database class.

    """

    def __init__(self, **kwargs):
        self.host = 'localhost'
        self.port = 3306
        self.socket = None
        self.user = ''
        self.password = ''
        self.database = ''
        self.schema = kwargs.pop('schema', None)
        self.yaml = kwargs.pop('yaml', None)

        self.schema_dict = None
        self.index_dict = None

        self.db = None
        self.cursor = None

        # https://stackoverflow.com/a/52949184
        pymysql.converters.encoders[np.float32] = pymysql.converters.escape_float
        pymysql.converters.encoders[np.float64] = pymysql.converters.escape_float
        pymysql.converters.encoders[np.int8] = pymysql.converters.escape_int
        pymysql.converters.encoders[np.int16] = pymysql.converters.escape_int
        pymysql.converters.encoders[np.int32] = pymysql.converters.escape_int
        pymysql.converters.encoders[np.int64] = pymysql.converters.escape_int
        pymysql.converters.conversions = pymysql.converters.encoders.copy()
        pymysql.converters.conversions.update(pymysql.converters.decoders)

    def assign_conf(self, conf, section='DB_mysql'):
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

        for attr in ['host', 'socket', 'user', 'password', 'database',
                     'schema', 'yaml']:
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
        d1, _ = fetch_ordered_tables(path_yaml, 'mysql', True,
                                      new_name=schema)
        self.schema_dict = d1
        self.index_dict = fetch_ordered_indexes(path_yaml, 'mysql', True,
                                                new_name=schema)

    def get_schema_sql(self, schema=None, table=None, mode='create_schema'):
        """
        Return schema creation or drop SQL statements.

        Parameters
        ----------
        schema : str
            Database schema
        table : str
            Table name, if only single table is required
        mode : str
            Controls which statements to return ('create_schema',
            'drop_schema', 'create_indexes', 'drop_indexes')
        """

        if self.schema_dict is None or self.index_dict is None:
            self.read_schema(schema=schema)

        if table is None:
            schema_dict = self.schema_dict.copy()
        else:
            schema_table = '{}.{}'.format(self.schema_dict['schema'], table)
            schema_dict = OrderedDict()

            if schema_table in self.schema_dict:
                schema_dict[schema_table] = self.schema_dict[schema_table]

        pdict = OrderedDict()
        creat_schema_mysql(schema_dict, pdict)
        creat_schema_index(self.index_dict, pdict)

        if mode in pdict:
            return pdict[mode]
        else:
            return ''

    def open_connection(self, host=None, port=None, socket=None,
                        user=None, password=None, database=None):
        """
        Open mysql database connection.

        Parameters
        ----------

        host : str
            mysql database host name
        port : str
            mysql database port number
        socket : str
            mysql database socket
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

        if socket is None:
            socket = self.socket

        if user is None:
            user = self.user

        if password is None:
            password = self.password

        if database is None:
            database = self.database

        while True:
            try:
                self.db = pymysql.connect(host=host, port=port,
                                          unix_socket=socket,
                                          user=user, password=password,
                                          database=database)
                self.host = host
                self.port = port
                self.socket = socket
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

        # Default return value
        val = None

        try:
            if len(args) == 1 and args[0].count(';') > 1:
                for sql in args[0].split(';'):
                    if sql.strip() != '':
                        numrows = self.cursor.execute(sql)
            elif 'RETURNING' in args[0]:
                sql = args[0].split('RETURNING')[0].rstrip()
                args = (sql,) + args[1:]
                numrows = self.cursor.execute(*args)
                val = self.cursor.lastrowid
            elif args[0].startswith('SELECT'):
                numrows = self.cursor.execute(*args)
                val = self.cursor.fetchone()

                if val is not None:
                    if len(val) == 1:
                        val = val[0]
            else:
                numrows = self.cursor.execute(*args)

            self.db.commit()
        except AttributeError:
            raise
        except pymysql.IntegrityError as e:
            if e.args[0] == 1062:
                print(e.args[1])
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

        return val

    def executemany_query(self, *args):
        """
        Execute SQL query with mutliple data rows and reopen connection if 
        connection has been lost.

        """

        try:
            numrows = self.cursor.executemany(*args)
        except AttributeError:
            raise
            #numrows = None
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
