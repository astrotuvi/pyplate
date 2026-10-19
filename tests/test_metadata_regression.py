"""The metadata of the redesigned code equals that of the old code.

tests/ver4/metadata/expected.json holds what the old code computed from the
archive inputs in tests/ver4/metadata (see tests/ver4/README.md, section
"Metadata regression"). Values are compared exactly, apart from the
intended differences listed in the README.
"""
import calendar
import dataclasses
import datetime
import glob
import json
import math
import os
import sys

import pytest

from pyplate.config import load_config
from pyplate.metadata import ArchiveReader

HERE = os.path.dirname(os.path.abspath(__file__))
META_DIR = os.path.join(HERE, 'ver4', 'metadata')
sys.path.insert(0, os.path.join(HERE, 'ver4'))

from sqlrender import decode  # noqa: E402

with open(os.path.join(META_DIR, 'expected.json')) as f:
    EXPECTED = json.load(f)

ARCHIVES = {a['name']: a for a in EXPECTED['archives']}
# [WFPDB] maindata_timezone of the WFPDB archives
reader_tz = {'pot015': 'ST', 'full': 'UT'}

CONFIGS = {'pot015': 'pot015.toml', 'my_archive': 'my_archive.toml',
           'mock': 'mock_archive.toml', 'full': 'full_archive.toml',
           'times': 'times_archive.toml'}


def set_file_times():
    t = datetime.datetime.strptime(EXPECTED['file_mtime'],
                                   '%Y-%m-%dT%H:%M:%S')
    t = calendar.timegm(t.timetuple())

    for d in ['fits', 'previews', 'logpages', 'covers']:
        for fn in glob.glob(os.path.join(META_DIR, d, '*')):
            os.utime(fn, (t, t))


def config(name, **wfpdb):
    """The archive configuration with the directories of expected.json."""
    archive = ARCHIVES[name]
    cfg = load_config(os.path.join(META_DIR, CONFIGS[name]))
    dirs = {k: os.path.join(META_DIR, v) for k, v in archive['dirs'].items()}
    cfg = dataclasses.replace(cfg, files=dataclasses.replace(cfg.files,
                                                             **dirs))

    if wfpdb:
        cfg = dataclasses.replace(cfg, wfpdb=dataclasses.replace(cfg.wfpdb,
                                                                 **wfpdb))

    return cfg


@pytest.fixture(scope='module')
def readers():
    set_file_times()
    return {name: ArchiveReader(config(name)) for name in ARCHIVES}


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))

    return a == b


def compare(record, expected, where, skip=()):
    diffs = []

    for col, val in expected.items():
        if col in skip:
            continue

        new = getattr(record, col)

        if not same(new, val):
            diffs.append('{}.{}: {!r} != {!r}'.format(where, col, new, val))

    return diffs


def plate_input(archive, input_id):
    """Arguments of get_plate() for an input identifier of the old code."""
    if archive['name'] in ('pot015', 'full'):
        return {'wfpdb_id': input_id}

    return {'plate_num': input_id}


def compare_plate(plate, expected, archive):
    name = archive['name']
    cols = decode(expected['columns'])
    other = decode(expected['other'])
    where = '{} {}'.format(name, expected['input_id'])
    diffs = []

    # Intended: plate_num is the CSV identifier for CSV plates
    if name in ('my_archive', 'mock', 'times'):
        assert cols['plate_num'] is None
        cols['plate_num'] = expected['input_id']

    diffs += compare(plate, cols, where, skip=('exposures',))

    # Intended: tz_orig is a plate column; for WFPDB records it is the file
    # default also without a recorded time (POT015_004252, MCK001_000003)
    if name in ('pot015', 'full') and 'tz_orig' not in other:
        assert all(e['time_orig_start'] is None for e in cols['exposures'])
        other['tz_orig'] = reader_tz[name]

    if not same(plate.tz_orig, other.get('tz_orig')):
        diffs.append('{}.tz_orig: {!r} != {!r}'.format(where, plate.tz_orig,
                                                       other.get('tz_orig')))

    assert len(plate.exposures) == len(cols['exposures'])

    for i, (exp, ecols) in enumerate(zip(plate.exposures, cols['exposures'])):
        ewhere = '{} exposure {}'.format(where, i + 1)
        ecols = dict(ecols)
        subs = ecols.pop('subexposures')

        # Intended: HJD of every exposure with coordinates and a mid-point
        # (the old code computed it only for the given coordinates)
        if (name, expected['input_id']) == ('mock', 'm1') and i > 0:
            assert ecols['hjd_mid'] is None
            ecols.pop('hjd_mid')
            ecols.pop('hjd_weighted')

        diffs += compare(exp, ecols, ewhere)
        assert len(exp.subexposures) == len(subs), ewhere

        for j, (sub, scols) in enumerate(zip(exp.subexposures, subs)):
            diffs += compare(sub, scols, '{} sub {}'.format(ewhere, j + 1))

        # Derived values, given for the computed coordinates
        for key, attr in [('ra', 'ra'), ('dec', 'dec'),
                          ('object_type', 'object_type')]:
            if key in other and i < len(other[key]):
                if not same(getattr(exp, attr), other[key][i]):
                    diffs.append('{}.{}: {!r} != {!r}'.format(
                        ewhere, attr, getattr(exp, attr), other[key][i]))

    if 'method' in other and plate.method != other['method']:
        diffs.append('{}.method: {!r} != {!r}'.format(where, plate.method,
                                                      other['method']))

    jd_orig = other.get('jd_avg_orig')

    if isinstance(jd_orig, list):
        assert [e.jd_mid_orig for e in plate.exposures] == jd_orig
    elif jd_orig is not None:
        assert plate.exposures[0].jd_mid_orig == jd_orig

    cover = other.get('fn_cover')
    logs = other.get('fn_log', [])

    if cover or logs:
        assert [lp.filename if lp else None for lp in plate.logpages] == \
            [cover] + logs
    else:
        assert plate.logpages == []

    return diffs


@pytest.mark.parametrize('name', list(ARCHIVES))
def test_plates(readers, name):
    reader = readers[name]
    archive = ARCHIVES[name]
    plates = list(reader.iter_plates())
    assert len(plates) == len(archive['plates'])
    diffs = []

    for plate, expected in zip(plates, archive['plates']):
        plate.compute_values(reader.cfg.metadata.evening_date)
        diffs += compare_plate(plate, expected, archive)

    assert diffs == []


@pytest.mark.parametrize('name', list(ARCHIVES))
def test_scans(readers, name):
    reader = readers[name]
    archive = ARCHIVES[name]
    diffs = []

    for expected in archive['scans']:
        fn = expected['input_filename']
        plate = reader.get_plate(filename_scan=fn)
        (scan,) = [s for s in plate.scans if s.filename_scan == fn]
        diffs += compare(scan, decode(expected['columns']),
                         '{} {}'.format(name, fn))

        # Intended: the new code links scans to CSV plates by plate_num
        if expected['plate_num'] is None:
            assert plate.plate_num == expected['other']['plate_id']
        else:
            assert plate.plate_num == expected['plate_num']

    assert diffs == []
    assert [s.filename_scan for s in reader.iter_scans()] == \
        [s['input_filename'] for s in archive['scans']]


@pytest.mark.parametrize('name', list(ARCHIVES))
def test_previews_logbooks_logpages(readers, name):
    reader = readers[name]
    archive = ARCHIVES[name]
    diffs = []
    previews = list(reader.iter_previews())
    assert len(previews) == len(archive['previews'])

    for preview, expected in zip(previews, archive['previews']):
        diffs += compare(preview, expected['columns'], preview.filename)
        plate = reader.get_plate(plate_num=expected['input_plate_id'])
        assert plate.plate_num == expected['plate_num']
        assert plate.wfpdb_id == expected['wfpdb_id']
        assert preview.filename in [p.filename for p in plate.previews]

    logbooks = reader.archive.logbooks
    assert len(logbooks) == len(archive['logbooks'])

    for logbook, expected in zip(logbooks, archive['logbooks']):
        diffs += compare(logbook, expected['columns'], expected['input_id'])

    logpages = list(reader.iter_logpages())
    assert len(logpages) == len(archive['logpages'])

    for logpage, expected in zip(logpages, archive['logpages']):
        diffs += compare(logpage, expected['columns'], logpage.filename)
        assert logpage.logbook_num == expected['logbook_num']

    assert diffs == []


def test_latin1_wfpdb_file():
    # The original Latin-1 bytes of the POT015 records, read with
    # [WFPDB] encoding, give the results of archive pot015 (UTF-8 copy)
    utf8 = ArchiveReader(config('pot015'))
    cfg = config('pot015', encoding='latin-1')
    cfg = dataclasses.replace(cfg, files=dataclasses.replace(
        cfg.files, wfpdb_dir=os.path.join(META_DIR, 'wfpdb_latin1')))
    latin1 = ArchiveReader(cfg)

    for a, b in zip(utf8.iter_plates(), latin1.iter_plates(), strict=True):
        a.compute_values()
        b.compute_values()
        assert a == b
