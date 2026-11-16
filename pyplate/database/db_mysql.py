"""MySQL/MariaDB backend (pymysql)."""
import time
from collections.abc import Sequence

from astropy.time import Time

from . import ddl
from .backend import Backend, plain

try:
    import pymysql
except ImportError:
    pymysql = None


class MysqlBackend(Backend):
    """MySQL or MariaDB.

    As in the old code: too many connections makes the connection retry
    every 10 s; a lost connection is reopened after 20 s and the statement
    repeated; a duplicate key is reported and skipped.
    """

    rdbms = 'mysql'

    def connect(self) -> None:
        if pymysql is None:
            raise ImportError('MySQL needs pymysql (pip install pyplate[mysql])')

        cfg = self.cfg

        while True:
            try:
                self.connection = pymysql.connect(
                    host=cfg.host, port=cfg.port or 3306,
                    unix_socket=cfg.socket, user=cfg.user,
                    password=cfg.password, database=cfg.database)
                break
            except pymysql.OperationalError as e:
                if e.args[0] == 1040:
                    print('MySQL server reports too many connections, '
                          'trying again')
                    time.sleep(10)
                else:
                    raise

        self.cursor = self.connection.cursor()

    def _retry(self, method, *args):
        try:
            method(*args)
        except pymysql.IntegrityError as e:
            if e.args[0] != 1062:
                raise

            print(e.args[1])
        except pymysql.OperationalError as e:
            if e.args[0] != 2006:
                raise

            print('MySQL server has gone away, trying to reconnect')
            time.sleep(20)
            self.connect()
            method(*args)

    def execute(self, sql: str, params: Sequence = ()) -> None:
        if pymysql is None:
            return super().execute(sql, params)

        self._retry(super().execute, sql, params)

    def executemany(self, sql: str, rows: Sequence[Sequence]) -> None:
        if pymysql is None:
            return super().executemany(sql, rows)

        self._retry(super().executemany, sql, rows)

    def insert_returning_id(self, sql: str, params: Sequence,
                            id_col: str) -> int | None:
        self.execute(sql, params)
        return self.cursor.lastrowid

    def upsert_sql(self, table: str, columns: list[str],
                   key_cols: list[str]) -> str:
        # VALUES() in ON DUPLICATE KEY UPDATE, which MariaDB supports (the
        # row alias of MySQL 8.0.20 is not available there)
        values = ','.join([self.placeholder] * len(columns))
        updates = ', '.join('{0} = VALUES({0})'.format(c) for c in columns
                            if c not in key_cols)
        return ('INSERT INTO {} ({}) VALUES ({}) ON DUPLICATE KEY UPDATE {}'
                .format(self.qualify(table), ','.join(columns), values,
                        updates))

    def prepare_value(self, column, value):
        """A DATE value before year 1000 or not a valid date becomes None."""
        if value is None or ddl.column_type(column, 'mysql') != 'DATE':
            return value

        try:
            if Time(plain(value), scale='tai') >= Time('1000-01-01',
                                                       scale='tai'):
                return value
        except ValueError:
            pass

        return None
