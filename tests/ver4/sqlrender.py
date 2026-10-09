"""Recording database cursor and SQL rendering, without pyplate imports.

These define how the expected SQL in tests/ver4/queries was produced, so the
test of the redesigned code must use them unchanged. See README.md.
"""
import math
import os
from collections import defaultdict

import numpy as np

QUERIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'queries')


class RecordingCursor:
    """
    Records executed queries instead of running them.

    An INSERT gets a sequential ID per table (1, 2, 3, ...; the table name as
    written in the SQL, including the schema prefix), returned as lastrowid
    (MySQL) and by fetchone() (PostgreSQL RETURNING). With select_ids=True, a
    SELECT finds one row with a mock ID (9001, 9002, ...) for each selected
    column; with select_ids=False, a SELECT finds no rows.

    executemany() records a comment line with the number of rows, followed by
    one entry per row.
    """

    def __init__(self, select_ids=False):
        self.select_ids = select_ids
        self.queries = []
        self.next_id = defaultdict(int)
        self.lastrowid = None
        self._last = None

    def execute(self, sql, params=None):
        self.queries.append((sql, params))

        if 'INTO' in sql:
            table = sql.split('INTO')[1].split()[0]
            self.next_id[table] += 1
            self.lastrowid = self.next_id[table]
            self._last = (self.lastrowid,)
        elif self.select_ids and sql.startswith('SELECT'):
            num_cols = len(sql[6:].split('FROM')[0].split(','))
            self._last = tuple(9000 + i for i in range(1, num_cols + 1))
        else:
            self._last = None

    def executemany(self, sql, rows):
        self.queries.append(('-- executemany: {:d} rows'.format(len(rows)),
                             None))
        self.queries.extend((sql, row) for row in rows)

    def fetchone(self):
        return self._last


class FakeConnection:
    def commit(self):
        pass


def decode(val):
    """
    Convert special-value markers in mock_data.json back to Python values,
    recursively: {"nan": true}, {"inf": 1 or -1}, {"quantity": value,
    "unit": unit} and {"header": [[keyword, value, comment], ...]}.
    """
    import astropy.units as u
    from astropy.io import fits

    if isinstance(val, list):
        return [decode(v) for v in val]
    if not isinstance(val, dict):
        return val
    if 'nan' in val:
        return np.nan
    if 'inf' in val:
        return np.inf * val['inf']
    if 'quantity' in val:
        return val['quantity'] * u.Unit(val['unit'])
    if 'header' in val:
        h = fits.Header()

        for keyword, value, comment in val['header']:
            h.append((keyword, value, comment))

        return h
    return {k: decode(v) for k, v in val.items()}


def literal(val, rdbms):
    """Render a Python value as an SQL literal."""
    if val is None:
        return 'NULL'
    if isinstance(val, (bool, np.bool_)):
        if rdbms == 'pgsql':
            return 'TRUE' if val else 'FALSE'
        return '1' if val else '0'
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        val = float(val)
        if math.isnan(val):
            return "'NaN'" if rdbms == 'pgsql' else 'NULL'
        return repr(val)
    s = str(val)
    if rdbms == 'mysql':
        s = s.replace('\\', '\\\\')
    return "'{}'".format(s.replace("'", "''"))


def render(sql, params, rdbms):
    """Insert rendered parameter values into the SQL statement."""
    if not params:
        return sql
    parts = sql.split('%s')
    assert len(parts) == len(params) + 1
    out = parts[0]
    for p, rest in zip(params, parts[1:]):
        out += literal(p, rdbms) + rest
    return out


def format_queries(rdbms, blocks):
    """
    Return recorded queries as text, in the format of queries/*/mock.sql.

    blocks is a list of (label, queries) where queries is a list of
    (sql, params) tuples.
    """
    lines = []

    for label, queries in blocks:
        lines.append('-- {}'.format(label))
        lines.extend(sql if sql.startswith('--')
                     else render(sql, params, rdbms) + ';'
                     for sql, params in queries)
        lines.append('')

    return '\n'.join(lines)


def write_queries(rdbms, filename, blocks):
    """Write recorded queries to tests/ver4/queries/<rdbms>/<filename>."""
    out_dir = os.path.join(QUERIES_DIR, rdbms)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, filename), 'w') as f:
        f.write(format_queries(rdbms, blocks))
