"""New behaviour of the database layer (DESIGN.md sections 7.2 and 7.3).

No database server is needed: SQLite runs in memory, and the PostgreSQL and
MySQL backends use the recording cursor of tests/ver4/sqlrender.py.
"""
import os
import sys

import numpy as np
import pytest

from pyplate.config import ConnectionConfig, DatabaseConfig
from pyplate.database.database import PlateDB
from pyplate.database.db_mysql import MysqlBackend
from pyplate.database.db_pgsql import PgsqlBackend
from pyplate.metadata import (Archive, Exposure, Logbook, Logpage, Plate,
                              Preview, Scan, SubExposure)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'ver4'))

from sqlrender import FakeConnection, RecordingCursor  # noqa: E402


def recording_db(rdbms, select_ids=False):
    cls = {'pgsql': PgsqlBackend, 'mysql': MysqlBackend}[rdbms]
    backend = cls(None, schema_name='applause_dr5')
    backend.cursor = RecordingCursor(select_ids=select_ids)
    backend.connection = FakeConnection()
    return PlateDB(DatabaseConfig(dr_num=5), backend=backend)


@pytest.fixture
def sqlite_db():
    cfg = DatabaseConfig(connection=ConnectionConfig(rdbms='sqlite',
                                                     file=':memory:'))
    db = PlateDB(cfg)
    db.open_connection()

    for mode in ['create_schema', 'create_indexes']:
        db.run_schema_sql(mode, execute=True)

    yield db
    db.close_connection()


def select(db, table):
    db.backend.execute('SELECT * FROM {}'.format(table))
    names = [d[0] for d in db.backend.cursor.description]
    return [dict(zip(names, row)) for row in db.backend.fetchall()]


def mock_plate():
    exposures = [
        Exposure(archive_id=7, exposure_num=1, object_name='Field A',
                 ra_orig='10:00:00', dec_orig='+20:00:00', ra_icrs=150.1,
                 dec_icrs=19.9, date_orig_start='1950-01-02',
                 time_orig_start='21:30:00', time_orig_end='22:00:00',
                 jd_mid_orig=2433284.40625, ut_start='1950-01-02T21:30:00',
                 jd_start=2433284.39583, exptime=1800., num_sub=1,
                 method_code=2),
        Exposure(archive_id=7, exposure_num=2, object_name='Field B',
                 date_orig_start='1950-01-02',
                 time_orig_start='22:10:00|22:17:00',
                 time_orig_end='22:15:00|22:22:00', exptime=600., num_sub=2,
                 method_code=2,
                 subexposures=[
                     SubExposure(exposure_num=2, subexposure_num=1,
                                 time_orig_start='22:10:00',
                                 time_orig_end='22:15:00', exptime=300.),
                     SubExposure(exposure_num=2, subexposure_num=2,
                                 time_orig_start='22:17:00',
                                 time_orig_end='22:22:00', exptime=300.)])]
    return Plate(archive_id=7, plate_num='A123', wfpdb_id='MCK001_000123',
                 date_orig='1950-01-02', tz_orig='UT', numexp=2,
                 method_code=2, observer='Jõgi Müller',
                 exposures=exposures,
                 scans=[Scan(archive_id=7, filename_scan='a123.fits',
                             naxis1=100, naxis2=200, file_size=12345)],
                 previews=[Preview(archive_id=7, filename='a123.jpg',
                                   preview_type=1)],
                 logpages=[Logpage(archive_id=7, filename='cover.jpg'),
                           Logpage(archive_id=7, filename='page1.jpg')])


def test_sqlite_round_trip(sqlite_db):
    db = sqlite_db
    archive = Archive(archive_id=7, archive_name='Mock archive',
                      institute='Mock institute')
    db.write_archive(archive)

    # Upsert: the second write updates the row
    archive.institute = 'Other institute'
    db.write_archive(archive)
    rows = select(db, 'archive')
    assert [Archive.from_row(r) for r in rows] == [archive]
    assert rows[0]['timestamp_insert'] is not None

    logbook = Logbook(archive_id=7, logbook_num='1', logbook_title='Journal')
    db.write_logbook(logbook)
    plate = mock_plate()

    for lp in plate.logpages:
        lp.logbook_num = '1'
        db.write_logpage(lp)

    plate_id = db.write_plate(plate)
    assert plate_id == plate.plate_id == 1

    assert [Plate.from_row(r) for r in select(db, 'plate')] == \
        [Plate.from_row(plate.to_row())]
    assert [Exposure.from_row(r) for r in select(db, 'exposure')] == \
        [Exposure.from_row(e.to_row()) for e in plate.exposures]
    # Sub-exposures are written without RETURNING, so they get no ID
    subs = plate.exposures[1].subexposures
    assert [SubExposure.from_row(r) for r in select(db, 'exposure_sub')] == \
        [SubExposure.from_row({**s.to_row(), 'subexposure_id': i + 1})
         for i, s in enumerate(subs)]
    assert all(s.exposure_id == plate.exposures[1].exposure_id
               for s in subs)
    assert [Scan.from_row(r) for r in select(db, 'scan')] == \
        [Scan.from_row(plate.scans[0].to_row())]
    assert [Preview.from_row(r) for r in select(db, 'preview')] == \
        [Preview.from_row(plate.previews[0].to_row())]
    assert [Logpage.from_row(r) for r in select(db, 'logpage')] == \
        [Logpage.from_row(lp.to_row()) for lp in plate.logpages]
    assert all(lp.logbook_id == logbook.logbook_id for lp in plate.logpages)

    links = select(db, 'plate_logpage')
    assert [(r['plate_id'], r['logpage_id'], r['logpage_order'])
            for r in links] == [(1, 1, 0), (1, 2, 1)]

    # Lookups
    assert db.get_plate_id('A123', 7) == 1
    assert db.get_plate_id_wfpdb('MCK001_000123') == 1
    assert db.get_scan_id('a123.fits', 7) == (1, 1)
    assert db.get_plate_epoch(1) is None


def test_sqlite_update_trigger(sqlite_db):
    db = sqlite_db
    db.write_archive(Archive(archive_id=1, archive_name='A'))
    db.backend.execute("UPDATE archive SET timestamp_update='2000-01-01'")
    db.write_archive(Archive(archive_id=1, archive_name='B'))
    (row,) = select(db, 'archive')
    assert row['archive_name'] == 'B'
    assert row['timestamp_update'] > '2000-01-01'


def test_sqlite_process(sqlite_db):
    db = sqlite_db
    pid = db.write_process_start(scan_id=1, plate_id=2, archive_id=3,
                                 filename='a.fits', use_psf=True)
    db.update_process(pid, num_sources=10, sky=None)
    db.write_process_end(pid, completed=1, duration=5)
    (row,) = select(db, 'process')
    assert (row['num_sources'], row['completed'], row['duration']) == \
        (10, 1, 5)
    assert row['timestamp_end'] is not None


@pytest.mark.parametrize('rdbms, expected', [
    ('pgsql', 'INSERT INTO applause_dr5.archive (archive_id,archive_name,'
              'institute) VALUES (%s,%s,%s) ON CONFLICT (archive_id) DO '
              'UPDATE SET archive_name = EXCLUDED.archive_name, '
              'institute = EXCLUDED.institute'),
    ('mysql', 'INSERT INTO applause_dr5.archive (archive_id,archive_name,'
              'institute) VALUES (%s,%s,%s) ON DUPLICATE KEY UPDATE '
              'archive_name = VALUES(archive_name), '
              'institute = VALUES(institute)')])
def test_archive_upsert_sql(rdbms, expected):
    db = recording_db(rdbms)
    db.write_archive(Archive(archive_id=7, archive_name='A', institute='B'))
    assert db.backend.cursor.queries == [(expected, (7, 'A', 'B'))]


def test_lookup_with_none_key_sends_nothing():
    db = recording_db('pgsql', select_ids=True)
    assert db.get_plate_id(None, 1) is None
    assert db.get_plate_id('A123', None) is None
    assert db.get_plate_id_wfpdb(None) is None
    assert db.get_scan_id(None, 1) == (None, None)
    assert db.get_logbook_id(None, 1) is None
    assert db.get_logpage_id('a.jpg', None) is None
    assert db.get_plate_epoch(None) is None
    assert db.backend.cursor.queries == []

    # A scan of a plate without plate_num and wfpdb_id: no lookups
    db.write_scan(Scan(archive_id=1, filename_scan='a.fits'))
    assert [q[0].split()[0] for q in db.backend.cursor.queries] == ['INSERT']


@pytest.mark.parametrize('record', [Archive(), Plate(), Scan(), Preview(),
                                    Logbook(), Logpage()])
def test_record_without_archive_id_refused(record):
    db = recording_db('pgsql')
    method = {'Archive': db.write_archive, 'Plate': db.write_plate,
              'Scan': db.write_scan, 'Preview': db.write_preview,
              'Logbook': db.write_logbook,
              'Logpage': db.write_logpage}[type(record).__name__]

    with pytest.raises(ValueError, match='archive_id'):
        method(record)

    assert db.backend.cursor.queries == []


def test_source_batches():
    db = recording_db('pgsql')
    n = 2500
    sources = [{'source_num': i + 1, 'x_source': float(i),
                'healpix256': i % 7 - 1, 'gaiadr4_id': i % 3}
               for i in range(n)]
    db.write_sources(sources, process_id=12, scan_id=1, plate_id=2,
                     archive_id=3)
    queries = db.backend.cursor.queries
    counts = [q[0] for q in queries if q[0].startswith('--')]
    assert counts == ['-- executemany: {} rows'.format(k)
                      for k in [1000, 1000, 1000, 1000, 500, 500]]
    tables = [q[0].split()[2] for q in queries if not q[0].startswith('--')]
    assert tables == (['applause_dr5.source'] * 1000 +
                      ['applause_dr5.source_calib'] * 1000) * 2 + \
        ['applause_dr5.source'] * 500 + ['applause_dr5.source_calib'] * 500

    for table in ['source', 'source_calib']:
        rows = [q[1] for q in queries
                if q[0].startswith('INSERT INTO applause_dr5.{} '
                                   .format(table))]
        ids = [r[0] for r in rows]
        assert ids == [db.source_id(12, i + 1) for i in range(n)]

    calib = [q for q in queries
             if q[0].startswith('INSERT INTO applause_dr5.source_calib ')]
    cols = calib[0][0].split('(')[1].split(')')[0].split(',')
    row = dict(zip(cols, calib[0][1]))
    assert row['healpix256'] is None and row['gaiadr4_id'] is None


class TypeRecordingCursor(RecordingCursor):
    def execute(self, sql, params=None):
        self.types = [type(p) for p in params]
        super().execute(sql, params)


@pytest.mark.parametrize('rdbms', ['pgsql', 'mysql'])
def test_numpy_values_passed_as_python(rdbms):
    db = recording_db(rdbms)
    db.backend.cursor = TypeRecordingCursor()
    db.update_process(1, sky=np.float32(1.5), num_sources=np.int64(7),
                      solved=np.bool_(True))
    assert db.backend.cursor.types == [float, int, bool, int]
