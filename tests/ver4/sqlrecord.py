"""Set up the DR4 PlateDB to record SQL instead of running it.

Used by gen_mock_queries.py. PlateDB keeps its real DB_pgsql / DB_mysql
backend, so the backend logic in execute_query() runs (e.g. MySQL removes
RETURNING), but the database cursor is replaced by RecordingCursor from
sqlrender.py.

Runs against the pyplate code as of commit ed2a193 (DR4 schema, before the
redesign described in DESIGN.md).
"""
import os

from pyplate.database.database import PlateDB
from pyplate.database.db_pgsql import DB_pgsql
from pyplate.database.db_mysql import DB_mysql
from sqlrender import RecordingCursor, FakeConnection

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BACKENDS = [('pgsql', DB_pgsql), ('mysql', DB_mysql)]


def recording_platedb(rdbms, cls, select_ids=False):
    """Return a PlateDB whose backend records queries instead of running them."""
    # Bypass __init__, which needs the database driver
    backend = cls.__new__(cls)
    backend.schema = 'applause_dr4'
    backend.yaml = None
    backend.cursor = RecordingCursor(select_ids=select_ids)
    backend.db = FakeConnection()

    platedb = PlateDB()
    platedb.rdbms = rdbms
    platedb.schema = 'applause_dr4'
    platedb.db = backend
    platedb.read_schema()
    return platedb
