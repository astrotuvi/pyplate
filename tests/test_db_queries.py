"""The database layer sends the same SQL as the old code (DESIGN.md 7.2).

tests/ver4/queries/*/mock.sql holds the SQL of every write, update and
lookup method of the old PlateDB, called with the mock data in
tests/ver4/mock_data.json (DR4 column names). The same calls go through
the new PlateDB with the DR4 schema, with the recording cursor and the
rendering of tests/ver4/sqlrender.py. See tests/ver4/README.md.
"""
import json
import os
import sys
from types import SimpleNamespace

import pytest

from pyplate.config import DatabaseConfig
from pyplate.database.database import PlateDB
from pyplate.database.db_mysql import MysqlBackend
from pyplate.database.db_pgsql import PgsqlBackend
from pyplate.database.schema import load_schema
from pyplate.metadata import (Exposure, Logbook, Logpage, Plate, Preview,
                              Scan, SubExposure)

HERE = os.path.dirname(os.path.abspath(__file__))
VER4 = os.path.join(HERE, 'ver4')
sys.path.insert(0, VER4)

from sqlrender import (FakeConnection, RecordingCursor, decode,  # noqa: E402
                       format_queries)

with open(os.path.join(VER4, 'mock_data.json')) as f:
    DATA = json.load(f)


def recording_db(rdbms, cursor):
    """PlateDB with the DR4 schema and a recording cursor."""
    backend_cls = {'pgsql': PgsqlBackend, 'mysql': MysqlBackend}[rdbms]
    backend = backend_cls(None, schema_name=DATA['schema'])
    backend.cursor = cursor
    backend.connection = FakeConnection()
    return PlateDB(DatabaseConfig(dr_num=DATA['dr_num']),
                   schema=load_schema('applause_dr4.yaml'), backend=backend,
                   hostname=DATA['hostname'], version=DATA['pyplate_version'])


# --- One adapter per method of the old PlateDB -------------------------------

def write_plate(db, d):
    plate = Plate.from_row(d)

    for e in d['exposures']:
        exp = Exposure.from_row(e)
        exp.subexposures = [SubExposure.from_row(s)
                            for s in e['subexposures']]
        plate.exposures.append(exp)

    db.write_plate(plate)


def write_plate_logpage(db, d):
    plate = Plate(plate_id=d['plate_id'], plate_num=d['plate_num'],
                  archive_id=d['archive_id'], wfpdb_id=d['wfpdb_id'],
                  logpages=[Logpage(filename=fn) for fn in d['logpages']])
    db.write_plate_logpages(plate)


ADAPTERS = {
    'write_plate': write_plate,
    'write_plate_logpage': write_plate_logpage,
    'write_scan': lambda db, d: db.write_scan(
        Scan.from_row(d), plate_num=d['plate_num'], wfpdb_id=d['wfpdb_id']),
    'update_scan': lambda db, d, **kw: db.update_scan(Scan.from_row(d), **kw),
    'write_preview': lambda db, d: db.write_preview(
        Preview.from_row(d), plate_num=d['plate_num'],
        wfpdb_id=d['wfpdb_id']),
    'write_logbook': lambda db, d: db.write_logbook(Logbook.from_row(d)),
    'write_logpage': lambda db, d: db.write_logpage(
        Logpage.from_row(d), logbook_num=d['logbook_num']),
    'write_platesolution': lambda db, d, **kw: db.write_platesolution(
        SimpleNamespace(**d), **kw),
}

# Methods with the same arguments as in the old code
for _name in ['write_solution', 'write_solution_healpix',
              'write_scanner_pattern', 'write_phot_calib',
              'write_calib_curve', 'write_sources', 'write_source_xmatches',
              'write_process_start', 'update_process', 'write_process_end',
              'write_processlog', 'get_plate_id', 'get_plate_id_wfpdb',
              'get_scan_id', 'get_logbook_id', 'get_logpage_id',
              'get_plate_epoch']:
    ADAPTERS[_name] = (lambda name: lambda db, *args, **kw:
                       getattr(db, name)(*args, **kw))(_name)


@pytest.mark.parametrize('rdbms', ['pgsql', 'mysql'])
def test_same_sql_as_old_code(rdbms):
    cursor = RecordingCursor(select_ids=True)
    db = recording_db(rdbms, cursor)
    blocks = []

    for case in DATA['cases']:
        cursor.select_ids = case['lookups_find']
        start = len(cursor.queries)
        ADAPTERS[case['method']](db, *decode(case['args']),
                                 **decode(case['kwargs']))
        blocks.append((case['label'], cursor.queries[start:]))

    with open(os.path.join(VER4, 'queries', rdbms, 'mock.sql')) as f:
        expected = f.read()

    assert format_queries(rdbms, blocks) == expected
