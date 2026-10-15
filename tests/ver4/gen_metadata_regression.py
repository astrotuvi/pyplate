"""Record the metadata that the old code computes from archive input files,
into tests/ver4/metadata/expected.json.

For each archive in tests/ver4/metadata (old-format configuration, WFPDB and
CSV input files), reads the plates with Archive.get_platemeta(), runs
Plate.compute_values() and records:

- columns: the values that PlateDB.write_plate() would write, in DR4 column
  names, nested as plate -> exposures -> subexposures (taken with
  Plate.get_value() exactly as write_plate() does);
- other: metadata keys without a database column that have a value (for
  example the RA/Dec strings, object type text, original time zone).

Scans in a scan CSV are recorded the same way, with the columns that
PlateDB.write_scan() would write.

compute_values() sets the key 'date' (FITS keyword DATE) to the current UTC
time; the time is replaced by a fixed value.

Runs against the pyplate code as of commit ed2a193 (DR4 schema, before the
redesign described in DESIGN.md). Run from the repository root:

    PYTHONPATH=. python tests/ver4/gen_metadata_regression.py
"""
import datetime
import json
import math
import os
from types import SimpleNamespace

import numpy as np

from pyplate import metadata
from pyplate.database.db_pgsql import DB_pgsql
from sqlrecord import ROOT, recording_platedb

META_DIR = os.path.join(ROOT, 'tests', 'ver4', 'metadata')
UTCNOW = datetime.datetime(2001, 2, 3, 4, 5, 6)


class FixedDatetime(datetime.datetime):
    @classmethod
    def utcnow(cls):
        return UTCNOW


# Fixed value instead of the current time
metadata.dt = SimpleNamespace(datetime=FixedDatetime)

# (name, configuration file, input subdirectory, input type)
ARCHIVES = [('pot015', 'pot015.conf', 'wfpdb', 'wfpdb'),
            ('my_archive', 'my_archive.conf', 'csv', 'csv'),
            ('mock', 'mock_archive.conf', 'csv', 'csv')]

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

_db = recording_platedb('pgsql', DB_pgsql)
COLUMNS = {t: [c for c in _db.get_table_dict(t)
               if not c.startswith('timestamp_')]
           for t in ['plate', 'exposure', 'exposure_sub', 'scan']}


def encode(val):
    """Convert to JSON: NumPy scalars to Python, NaN/inf to markers."""
    if isinstance(val, list):
        return [encode(v) for v in val]
    if isinstance(val, np.generic):
        val = val.item()
    if isinstance(val, float):
        if math.isnan(val):
            return {'nan': True}
        if math.isinf(val):
            return {'inf': 1 if val > 0 else -1}
    return val


def plain_numbers(platemeta):
    """
    Work around a NumPy 2 bug in Plate.get_value(): np.float64 == [] is
    ambiguous. Convert NumPy scalars to Python numbers (same values).
    """
    for meta in [platemeta] + [e for e in platemeta.exposures or []
                               if e is not None]:
        for k, v in meta.items():
            if isinstance(v, np.generic):
                meta[k] = v.item()
            elif isinstance(v, list):
                meta[k] = [x.item() if isinstance(x, np.generic) else x
                           for x in v]


def row(meta, table, keymap, exp=0, skip=('plate_id',)):
    """Column values as PlateDB.write_plate() / write_scan() take them."""
    out = {}

    for col in COLUMNS[table]:
        key = keymap.get(col, col)

        if col not in skip and key in meta:
            out[col] = encode(meta.get_value(key, exp=exp))

    return out


def plate_columns(pm):
    out = row(pm, 'plate', PLATE_KEYS)
    out['exposures'] = []

    for exp in range(pm['numexp']):
        e = row(pm, 'exposure', PLATE_KEYS, exp)
        e['exposure_num'] = exp + 1
        e['subexposures'] = []

        if len(pm['numsub']) > exp and pm['numsub'][exp] > 1:
            expmeta = pm.exposures[exp]

            for sub in range(pm['numsub'][exp]):
                s = row(expmeta, 'exposure_sub', PLATE_KEYS, sub)
                s['subexposure_num'] = sub + 1
                e['subexposures'].append(s)

        out['exposures'].append(e)

    return out


def other_values(pm, keymaps):
    """Metadata keys without a database column that have a value."""
    used = set()

    for table, keymap in keymaps:
        used.update(keymap.get(c, c) for c in COLUMNS[table])

    return {k: encode(v) for k, v in pm.items()
            if k not in used and v not in (None, [], '')}


def read_archive(conf_file, input_dir, input_type):
    archive = metadata.Archive()
    archive.assign_conf(os.path.join(META_DIR, conf_file))
    path = os.path.join(META_DIR, input_dir)

    if input_type == 'wfpdb':
        archive.conf.set('Files', 'wfpdb_dir', path)
        archive.read_wfpdb()
    else:
        archive.conf.set('Files', 'csv_dir', path)
        archive.read_csv()

    return archive


def record_archive(name, conf_file, input_dir, input_type):
    archive = read_archive(conf_file, input_dir, input_type)
    plates = []

    for input_id in archive.get_platelist():
        pm = archive.get_platemeta(plate_id=input_id)
        pm.compute_values()
        plain_numbers(pm)
        plates.append({
            'input_id': input_id,
            'columns': plate_columns(pm),
            'other': other_values(pm, [('plate', PLATE_KEYS),
                                       ('exposure', PLATE_KEYS),
                                       ('exposure_sub', PLATE_KEYS)])})

    scans = []

    for filename in archive.get_scanlist():
        pm = archive.get_platemeta(filename=filename)
        pm.compute_values()
        plain_numbers(pm)
        scans.append({
            'input_filename': filename,
            'columns': row(pm, 'scan', SCAN_KEYS,
                           skip=('plate_id', 'scan_id')),
            'plate_num': pm['plate_num'],
            'other': other_values(pm, [('plate', PLATE_KEYS),
                                       ('exposure', PLATE_KEYS),
                                       ('scan', SCAN_KEYS)])})

    return {'name': name, 'conf': conf_file,
            'input': os.path.join(input_dir, ''), 'plates': plates,
            'scans': scans}


if __name__ == '__main__':
    result = {
        'description': 'Metadata computed by the old code from the archive '
                       'inputs in tests/ver4/metadata, generated by '
                       'gen_metadata_regression.py. See README.md.',
        'utcnow': UTCNOW.strftime('%Y-%m-%dT%H:%M:%S'),
        'archives': [record_archive(*a) for a in ARCHIVES],
    }

    with open(os.path.join(META_DIR, 'expected.json'), 'w') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
        f.write('\n')

    for a in result['archives']:
        print(a['name'], len(a['plates']), 'plates,', len(a['scans']),
              'scans')
