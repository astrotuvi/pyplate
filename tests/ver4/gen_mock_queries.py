"""Record the SQL that every PlateDB write, update and lookup method sends,
using mock data, into tests/ver4/queries/*/mock.sql (no DB connection).

The mock data fill every column of the applause_dr4 tables that a method can
write. Values are deterministic and distinct: integers 1, 2, 3, ..., floats
n + 0.5, timestamps and dates derived from n, and strings made from the
column (or metadata key) name. Values are generated from the PostgreSQL
column types, so both backends get the same values. Several methods are
called twice to cover both of their branches (e.g. with and without a known
plate_id).

A SELECT finds one row with mock IDs (9001, 9002, ...), so that methods
which depend on a lookup also write their rows. Cases called with
lookups_find=False get no rows from a SELECT instead, to cover the branches
where a lookup fails. The hostname and the pyplate
version written by write_process_start() are replaced by fixed values.

The inputs of each call are also written to tests/ver4/mock_data.json, in
database column names and without pyplate classes, so that the same calls
can be repeated with the redesigned code and compared with mock.sql.

See sqlrecord.py for how queries are recorded and rendered. The
create_schema() and drop_schema() SQL is in tests/ver4/ddl.

Runs against the pyplate code as of commit ed2a193 (DR4 schema, before the
redesign described in DESIGN.md). Run from the repository root:

    PYTHONPATH=. python tests/ver4/gen_mock_queries.py
"""
import json
import os
import re
import socket
from types import SimpleNamespace

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table

from pyplate import metadata
from pyplate.database import database
from pyplate.database.db_pgsql import DB_pgsql
from sqlrecord import ROOT, BACKENDS, recording_platedb
from sqlrender import write_queries

# Fixed values instead of the machine name and the package version
socket.gethostname = lambda: 'mockhost'
database.__version__ = 'mock-version'

# Metadata keys that differ from column names, as in PlateDB.write_plate()
PLATE_KEYS = {'plate_id': 'db_plate_id', 'flag_coord': 'coord_flag',
              'ra_icrs': 'ra_deg', 'dec_icrs': 'dec_deg',
              'date_orig_start': 'date_orig', 'date_orig_end': 'date_orig_end',
              'time_orig_start': 'tms_orig', 'time_orig_end': 'tme_orig',
              'flag_time': 'time_flag', 'ut_start': 'date_obs',
              'ut_mid': 'date_avg', 'ut_weighted': 'date_weighted',
              'ut_end': 'date_end', 'year_start': 'year',
              'year_mid': 'year_avg', 'jd_start': 'jd', 'jd_mid': 'jd_avg',
              'hjd_mid': 'hjd_avg', 'num_sub': 'numsub'}

# Metadata keys that differ from column names, as in PlateDB.write_scan()
SCAN_KEYS = {'plate_id': 'db_plate_id', 'filename_scan': 'filename',
             'filename_wedge': 'fn_wedge', 'naxis1': 'fits_naxis1',
             'naxis2': 'fits_naxis2', 'minval': 'fits_minval',
             'maxval': 'fits_maxval', 'pixel_size1': 'pix_size1',
             'pixel_size2': 'pix_size2', 'scan_date': 'datescan',
             'file_datetime': 'fits_datetime', 'file_size': 'fits_size'}

# Column values that the write methods take as arguments
ID_COLUMNS = ['source_id', 'solution_id', 'solutionset_id', 'process_id',
              'scan_id', 'plate_id', 'archive_id']

# Arguments passed to the write methods
IDS = dict(process_id=21, scan_id=22, plate_id=23, archive_id=24)

types_db = recording_platedb('pgsql', DB_pgsql)


def column_types(table):
    """Return {column: PostgreSQL type} for a DR4 table."""
    return types_db.get_table_dict(table)


def data_columns(table, exclude=()):
    """Columns that a mock object should contain."""
    return {c: t for c, t in column_types(table).items()
            if not c.startswith('timestamp_') and 'serial' not in t
            and c not in exclude}


class MockValues:
    """Deterministic, distinct mock values by column type."""

    def __init__(self):
        self.n = 0
        self.cache = {}

    def get(self, key, dtype, i=0, ns=''):
        if (ns, key, i) in self.cache:
            return self.cache[(ns, key, i)]

        self.n += 1
        n = self.n
        t = dtype.lower()

        if t.startswith(('integer', 'smallint', 'bigint')):
            val = n
        elif t.startswith(('real', 'double')):
            val = n + 0.5
        elif t.startswith('timestamp'):
            val = '2001-01-01T00:{:02d}:{:02d}'.format(n // 60 % 60, n % 60)
        elif key.startswith('date') or key.endswith('_date'):
            val = '1950-01-{:02d}'.format(n % 28 + 1)
        else:
            val = '{}{}{}'.format(ns, key, i)
            m = re.search(r'\((\d+)\)', t)

            if m:
                val = val[:int(m.group(1))]

        self.cache[(ns, key, i)] = val
        return val


mock = MockValues()


def fill_meta(meta, table, keymap, n=1, ns='', skip=()):
    """Fill metadata object keys that correspond to table columns."""
    for col, dtype in data_columns(table).items():
        key = keymap.get(col, col)

        if key in skip or key not in meta:
            continue

        if key in metadata._keyword_meta and metadata._keyword_meta[key][1]:
            meta[key] = [mock.get(key, dtype, i, ns) for i in range(n)]
        else:
            meta[key] = mock.get(key, dtype, 0, ns)


def mock_header(name):
    h = fits.Header()
    h['CTYPE1'] = 'RA---TAN'
    h['COMMENT'] = '{} header'.format(name)
    return h


def mock_plate(db_plate_id=None, date_orig=None):
    """Plate with two exposures; the second has two sub-exposures."""
    plate = metadata.Plate()
    fill_meta(plate, 'plate', PLATE_KEYS, skip=['db_plate_id'])
    fill_meta(plate, 'exposure', PLATE_KEYS, n=2, skip=['db_plate_id'])
    plate['numexp'] = 2
    plate['numsub'] = [1, 2]
    plate['db_plate_id'] = db_plate_id

    if date_orig is not None:
        plate['date_orig'] = [date_orig, date_orig]

    sub = metadata.Plate()
    fill_meta(sub, 'exposure_sub', PLATE_KEYS, n=2, ns='sub_',
              skip=['db_plate_id'])
    plate.exposures = [None, sub]
    return plate


def mock_plate_logpage(db_plate_id=None):
    plate = metadata.Plate()
    plate['plate_num'] = 'plate_num0'
    plate['archive_id'] = IDS['archive_id']
    plate['wfpdb_id'] = 'wfpdb_id0'
    plate['db_plate_id'] = db_plate_id
    plate['fn_cover'] = 'fn_cover0'
    plate['fn_log'] = ['fn_log0', 'fn_log1']
    return plate


def mock_scan(scan_id=None, db_plate_id=None):
    plate = metadata.Plate()
    fill_meta(plate, 'scan', SCAN_KEYS, skip=['db_plate_id', 'scan_id'])
    plate['plate_num'] = 'plate_num0'
    plate['wfpdb_id'] = 'wfpdb_id0'
    plate['scan_id'] = scan_id
    plate['db_plate_id'] = db_plate_id
    return plate


def mock_preview(preview_id=None, db_plate_id=None):
    preview = metadata.Preview(filename='preview.jpg')
    fill_meta(preview, 'preview', {}, skip=['plate_id', 'preview_id'])
    preview['plate_num'] = 'plate_num0'
    preview['wfpdb_id'] = 'wfpdb_id0'
    preview['preview_id'] = preview_id
    preview['db_plate_id'] = db_plate_id
    return preview


def mock_logbook():
    logbook = metadata.Logbook()
    fill_meta(logbook, 'logbook', {})
    return logbook


def mock_logpage(logbook_id=None):
    logpage = metadata.Logpage(filename='logpage.jpg')
    fill_meta(logpage, 'logpage', {})
    logpage['logbook_num'] = 'logbook_num0'
    logpage['logbook_id'] = logbook_id
    return logpage


def mock_dict(table, exclude=ID_COLUMNS):
    """Dictionary with a value for each data column of the table."""
    return {c: mock.get(c, t, ns=table + '.')
            for c, t in data_columns(table, exclude).items()}


def mock_platesolution():
    """Plate solution with Quantity, Header and non-finite values."""
    d = mock_dict('solution_set')
    d['mean_pixel_scale'] = d['mean_pixel_scale'] * u.arcsec / u.pixel
    d['source_density'] = np.nan
    d['header_wcs'] = mock_header('solution_set')
    return SimpleNamespace(**d)


def mock_solution():
    """Solution with Quantity, Header, bool and non-finite values."""
    d = mock_dict('solution')
    d['pixel_scale'] = d['pixel_scale'] * u.arcsec / u.pixel
    d['fov1'] = d['fov1'] * u.deg
    d['plate_mirrored'] = True
    d['ncp_on_plate'] = False
    d['scamp_chi2'] = np.inf
    d['scamp_sigma_mean'] = np.nan

    for h in ['header_wcs', 'header_anet', 'header_scamp']:
        d[h] = mock_header(h)

    return d


def mock_table(columns, nrows, ns):
    """Astropy table with one column per {column: type}."""
    t = Table()

    for c, dtype in columns.items():
        t[c] = [mock.get(c, dtype, i, ns) for i in range(nrows)]

    return t


def mock_sources():
    """Two sources; the second has non-finite and special values."""
    cols = data_columns('source', ID_COLUMNS)
    cols.update(data_columns('source_calib', ID_COLUMNS))
    t = mock_table(cols, 2, 'source.')
    t['mag_auto'][1] = np.nan
    t['natmag'][1] = np.inf
    t['healpix256'][1] = -1
    t['gaiaedr3_id'][1] = 0
    return t


def mock_xmatches():
    cols = {'source_num': 'integer'}
    cols.update(data_columns('source_xmatch', ID_COLUMNS))
    t = mock_table(cols, 2, 'source_xmatch.')
    t['dist'][1] = np.nan
    return t


# Export of the mock inputs to mock_data.json, in database column names and
# independent of the pyplate classes

def encode(val):
    """Convert a value to JSON; mark values that JSON cannot hold."""
    if isinstance(val, np.generic):
        val = val.item()
    if isinstance(val, u.Quantity):
        return {'quantity': encode(val.value), 'unit': val.unit.to_string()}
    if isinstance(val, fits.Header):
        return {'header': [[c.keyword, c.value, c.comment] for c in val.cards]}
    if isinstance(val, float):
        if np.isnan(val):
            return {'nan': True}
        if np.isinf(val):
            return {'inf': 1 if val > 0 else -1}
    return val


def encode_row(row):
    return {k: encode(v) for k, v in row.items()}


def meta_row(meta, table, keymap, i=0, skip=()):
    """Values of the table columns from a metadata object (exposure i)."""
    row = {}

    for col in data_columns(table):
        key = keymap.get(col, col)

        if col in skip or key not in meta:
            continue

        val = meta[key]

        if isinstance(val, list):
            val = val[i] if len(val) > i else None

        row[col] = encode(val)

    return row


def export_plate(plate):
    """Plate with its exposures and sub-exposures."""
    out = meta_row(plate, 'plate', PLATE_KEYS, skip=['plate_id'])
    out['plate_id'] = plate['db_plate_id']
    out['exposures'] = []

    for i in range(plate['numexp']):
        exp = meta_row(plate, 'exposure', PLATE_KEYS, i, skip=['plate_id'])
        exp['exposure_num'] = i + 1
        exp['subexposures'] = []

        if plate['numsub'][i] > 1:
            for j in range(plate['numsub'][i]):
                sub = meta_row(plate.exposures[i], 'exposure_sub',
                               PLATE_KEYS, j, skip=['plate_id'])
                sub['subexposure_num'] = j + 1
                exp['subexposures'].append(sub)

        out['exposures'].append(exp)

    return out


def export_plate_logpage(plate):
    """Plate keys and its logpage filenames (cover first)."""
    return {'plate_id': plate['db_plate_id'],
            'plate_num': plate['plate_num'],
            'archive_id': plate['archive_id'],
            'wfpdb_id': plate['wfpdb_id'],
            'logpages': [plate['fn_cover']] + plate['fn_log']}


def export_scan(plate):
    """Scan row plus the plate keys used to look up plate_id."""
    out = meta_row(plate, 'scan', SCAN_KEYS, skip=['plate_id', 'scan_id'])
    out['scan_id'] = plate['scan_id']
    out['plate_id'] = plate['db_plate_id']
    out['plate_num'] = plate['plate_num']
    out['wfpdb_id'] = plate['wfpdb_id']
    return out


def export_arg(method, arg):
    if isinstance(arg, metadata.Plate):
        if method == 'write_plate':
            return export_plate(arg)
        if method == 'write_plate_logpage':
            return export_plate_logpage(arg)
        return export_scan(arg)
    if isinstance(arg, metadata.Preview):
        out = meta_row(arg, 'preview', {}, skip=['plate_id', 'preview_id'])
        out['preview_id'] = arg['preview_id']
        out['plate_id'] = arg['db_plate_id']
        out['plate_num'] = arg['plate_num']
        out['wfpdb_id'] = arg['wfpdb_id']
        return out
    if isinstance(arg, metadata.Logbook):
        return meta_row(arg, 'logbook', {})
    if isinstance(arg, metadata.Logpage):
        out = meta_row(arg, 'logpage', {})
        out['logbook_num'] = arg['logbook_num']
        return out
    if isinstance(arg, SimpleNamespace):
        return encode_row(vars(arg))
    if isinstance(arg, Table):
        return [encode_row(dict(zip(arg.colnames, r))) for r in arg]
    if hasattr(arg, 'colnames'):
        # Table row
        return encode_row(dict(zip(arg.colnames, arg)))
    if isinstance(arg, dict):
        return encode_row(arg)
    return encode(arg)


def run_cases(platedb, cases=None):
    """
    Call each PlateDB method; return a list of (label, queries).

    If cases is a list, the inputs of each call are appended to it.
    """
    queries = platedb.db.cursor.queries
    blocks = []

    def case(label, func, *args, lookups_find=True, **kwargs):
        if cases is not None:
            # Export before the call, because some methods modify their input
            name = func.__name__
            cases.append({'label': label, 'method': name,
                          'lookups_find': lookups_find,
                          'args': [export_arg(name, a) for a in args],
                          'kwargs': {k: export_arg(name, v)
                                     for k, v in kwargs.items()}})

        platedb.db.cursor.select_ids = lookups_find
        start = len(queries)
        func(*args, **kwargs)
        blocks.append((label, queries[start:]))
        platedb.db.cursor.select_ids = True

    case('write_plate: new plate', platedb.write_plate, mock_plate())
    case('write_plate: plate_id given', platedb.write_plate,
         mock_plate(db_plate_id=31))
    case('write_plate: date_orig before year 1000', platedb.write_plate,
         mock_plate(date_orig='0900-01-01'))
    case('write_plate_logpage: plate_id given', platedb.write_plate_logpage,
         mock_plate_logpage(db_plate_id=31))
    case('write_plate_logpage: plate_id looked up',
         platedb.write_plate_logpage, mock_plate_logpage())
    case('write_scan: plate_id looked up', platedb.write_scan, mock_scan())
    case('write_scan: plate_id lookups find nothing', platedb.write_scan,
         mock_scan(), lookups_find=False)
    case('write_scan: scan_id and plate_id given', platedb.write_scan,
         mock_scan(scan_id=32, db_plate_id=31))
    case('update_scan: all columns, scan_id given', platedb.update_scan,
         mock_scan(scan_id=32, db_plate_id=31))
    case('update_scan: file columns, scan_id looked up', platedb.update_scan,
         mock_scan(), filecols=True)
    case('write_preview: plate_id looked up', platedb.write_preview,
         mock_preview())
    case('write_preview: preview_id and plate_id given',
         platedb.write_preview, mock_preview(preview_id=33, db_plate_id=31))
    case('write_logbook', platedb.write_logbook, mock_logbook())
    case('write_logpage: logbook_id looked up', platedb.write_logpage,
         mock_logpage())
    case('write_logpage: logbook_id given', platedb.write_logpage,
         mock_logpage(logbook_id=34))
    case('write_platesolution', platedb.write_platesolution,
         mock_platesolution(), **IDS)
    case('write_solution', platedb.write_solution, mock_solution(),
         solutionset_id=35, **IDS)
    case('write_solution_healpix', platedb.write_solution_healpix,
         mock_table(data_columns('solution_healpix', ID_COLUMNS
                                 + ['solution_num']), 2, 'solution_healpix.'),
         solution_id=36, solutionset_id=35, solution_num=1, **IDS)
    case('write_scanner_pattern', platedb.write_scanner_pattern,
         mock_table(data_columns('scanner_pattern', ID_COLUMNS), 1,
                    'scanner_pattern.')[0],
         solutionset_id=35, **IDS)
    case('write_phot_calib', platedb.write_phot_calib,
         mock_dict('phot_calib'), **IDS)
    case('write_calib_curve', platedb.write_calib_curve,
         mock_dict('phot_calib_curve'), **IDS)
    case('write_sources', platedb.write_sources, mock_sources(), **IDS)
    case('write_source_xmatches', platedb.write_source_xmatches,
         mock_xmatches(), **IDS)
    case('write_process_start', platedb.write_process_start,
         scan_id=22, plate_id=23, archive_id=24, filename='mock.fits',
         use_psf=True)

    # All process columns, plus an unknown keyword and a None value, which
    # update_process() skips
    process_values = mock_dict('process', exclude=['process_id'])
    process_values['not_a_column'] = 1
    process_values['sky'] = None
    case('update_process: all columns except sky', platedb.update_process,
         21, **process_values)
    case('update_process: sky', platedb.update_process, 21, sky=40.5)
    case('write_process_end', platedb.write_process_end, 21, completed=1,
         duration=12.5)
    case('write_processlog', platedb.write_processlog, 3,
         "Quote ' and backslash \\ in a message", event=30, solution_num=1,
         **IDS)
    case('get_plate_id', platedb.get_plate_id, 'plate_num0', 24)
    case('get_plate_id_wfpdb', platedb.get_plate_id_wfpdb, 'wfpdb_id0')
    case('get_scan_id', platedb.get_scan_id, 'mock.fits', 24)
    case('get_logbook_id', platedb.get_logbook_id, 'logbook_num0', 24)
    case('get_logpage_id', platedb.get_logpage_id, 'fn_log0', 24)
    case('get_plate_epoch', platedb.get_plate_epoch, 23)
    return blocks


exported = {}

for rdbms, cls in BACKENDS:
    mock = MockValues()
    platedb = recording_platedb(rdbms, cls, select_ids=True)
    platedb.dr_num = 4
    exported[rdbms] = []
    blocks = run_cases(platedb, cases=exported[rdbms])
    write_queries(rdbms, 'mock.sql', blocks)
    print(rdbms, len(platedb.db.cursor.queries), 'queries')

# Both backends must have been given the same inputs
assert exported['pgsql'] == exported['mysql']

mock_data = {
    'description': 'Inputs of the PlateDB method calls recorded in '
                   'queries/*/mock.sql, generated by gen_mock_queries.py. '
                   'Values are in database column names (applause_dr4). '
                   'Special values: {"nan": true}, {"inf": 1 or -1}, '
                   '{"quantity": value, "unit": unit}, '
                   '{"header": [[keyword, value, comment], ...]}.',
    'schema': 'applause_dr4',
    'dr_num': 4,
    'hostname': 'mockhost',
    'pyplate_version': 'mock-version',
    'select_result': 'A SELECT finds one row with the values 9001, 9002, ... '
                     'for the selected columns. In cases with '
                     '"lookups_find": false, a SELECT finds no rows.',
    'insert_ids': 'An INSERT gets an ID 1, 2, 3, ... per table '
                  '(lastrowid in MySQL, RETURNING in PostgreSQL).',
    'cases': exported['pgsql'],
}

with open(os.path.join(ROOT, 'tests', 'ver4', 'mock_data.json'), 'w') as f:
    json.dump(mock_data, f, indent=1)
    f.write('\n')
