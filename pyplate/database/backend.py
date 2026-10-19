"""Backend interface: what differs between the database systems."""
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..config import ConnectionConfig
from . import ddl
from .schema import Schema


def plain(value: Any) -> Any:
    """NumPy scalars as plain Python values, for the database drivers."""
    if isinstance(value, np.generic):
        return value.item()

    return value


class Backend:
    """Connection and SQL dialect of one database system.

    Subclasses set :attr:`rdbms`, implement :meth:`connect` and override the
    dialect where it differs. ``connection`` and ``cursor`` are the DB-API
    connection and cursor; tests may set them directly.

    Parameters
    ----------
    cfg
        Connection settings.
    schema_name
        Schema (PostgreSQL) or database (MySQL) name used as the table
        prefix; None for no prefix.
    """

    rdbms: str = ''
    placeholder: str = '%s'
    now_sql: str = 'NOW()'

    def __init__(self, cfg: ConnectionConfig | None = None,
                 schema_name: str | None = None):
        self.cfg = cfg
        self.schema_name = schema_name
        self.connection = None
        self.cursor = None

    def connect(self) -> None:
        """Open the connection and a cursor."""
        raise NotImplementedError

    def close(self) -> None:
        """Commit and close the connection."""
        if self.connection is not None:
            self.cursor.close()
            self.connection.commit()
            self.connection.close()
            self.connection = None
            self.cursor = None

    def qualify(self, table: str) -> str:
        """Table name with the schema prefix."""
        if self.schema_name:
            return '{}.{}'.format(self.schema_name, table)

        return table

    def execute(self, sql: str, params: Sequence = ()) -> None:
        """Execute a statement and commit."""
        self.cursor.execute(sql, tuple(plain(p) for p in params))
        self.connection.commit()

    def query(self, sql: str, params: Sequence = ()) -> tuple | None:
        """Execute a statement, fetch the first row, then commit."""
        self.cursor.execute(sql, tuple(plain(p) for p in params))
        row = self.cursor.fetchone()
        self.connection.commit()
        return row

    def select_all(self, sql: str, params: Sequence = ()) -> list[tuple]:
        """Execute a query and return all rows."""
        self.cursor.execute(sql, tuple(plain(p) for p in params))
        rows = self.cursor.fetchall()
        self.connection.commit()
        return rows

    def fetchone(self) -> tuple | None:
        return self.cursor.fetchone()

    def fetchall(self) -> list[tuple]:
        return self.cursor.fetchall()

    def executemany(self, sql: str, rows: Sequence[Sequence]) -> None:
        """Execute a statement for each row and commit."""
        self.cursor.executemany(sql, [tuple(plain(v) for v in row)
                                      for row in rows])
        self.connection.commit()

    def insert_returning_id(self, sql: str, params: Sequence,
                            id_col: str) -> int | None:
        """Execute an INSERT and return the ID of the new row."""
        row = self.query('{} RETURNING {}'.format(sql, id_col), params)
        return row[0] if row is not None else None

    def upsert_sql(self, table: str, columns: list[str],
                   key_cols: list[str]) -> str:
        """INSERT that updates the row if one with the same key exists."""
        values = ','.join([self.placeholder] * len(columns))
        updates = ', '.join('{0} = EXCLUDED.{0}'.format(c) for c in columns
                            if c not in key_cols)
        return ('INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}) DO UPDATE '
                'SET {}'.format(self.qualify(table), ','.join(columns),
                                values, ', '.join(key_cols), updates))

    def prepare_value(self, column, value: Any) -> Any:
        """Backend-specific conversion of a value before writing."""
        return value

    def ddl(self, schema: Schema, mode: str) -> list[str]:
        """Statements for one of :data:`pyplate.database.ddl.MODES`."""
        return ddl.statements(schema, self.rdbms, mode)
