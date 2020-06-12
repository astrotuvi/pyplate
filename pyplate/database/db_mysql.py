import numpy as np
import time
import socket
import os
import csv
from collections import OrderedDict
from astropy.time import Time
from ..conf import read_conf
from .._version import __version__
from .db_yaml import fetch_ordered_tables, fetch_ordered_indexes

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
