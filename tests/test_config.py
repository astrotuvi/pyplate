"""Configuration: TOML parsing, rejected keys, and ini2toml (DESIGN.md 3)."""
import configparser
import os

import pytest

from pyplate.config import ConfigError, config_from_dict, load_config
from pyplate.config.ini2toml import convert, read_ini

HERE = os.path.dirname(os.path.abspath(__file__))
VER4 = os.path.join(HERE, 'ver4')

# Old-format configurations in tests/ver4 and their TOML versions
CONVERSIONS = [
    ('metadata/pot015.conf', 'metadata/pot015.toml'),
    ('metadata/my_archive.conf', 'metadata/my_archive.toml'),
    ('metadata/mock_archive.conf', 'metadata/mock_archive.toml'),
    ('metadata/full_archive.conf', 'metadata/full_archive.toml'),
    ('metadata/times_archive.conf', 'metadata/times_archive.toml'),
    ('header/header.conf', 'header/header.toml'),
    ('plates/plates.conf', 'plates/plates.toml'),
]


@pytest.mark.parametrize('ini,toml', CONVERSIONS)
def test_ini2toml_gives_committed_toml(ini, toml):
    text, _ = convert(read_ini([os.path.join(VER4, ini)]))

    with open(os.path.join(VER4, toml)) as f:
        assert text == f.read()

    load_config(os.path.join(VER4, toml))


def test_ini2toml_dropped_settings():
    _, dropped = convert(read_ini([os.path.join(VER4, 'plates',
                                                'plates.conf')]))
    assert [d.split(':')[0] for d in dropped] == [
        '[plate_archive_hamburger_sternwarte_LA_20171128_updated.csv] '
        'skycond',
        '[plate_archive_hamburger_sternwarte_LA_20171128_updated.csv] method',
        '[LA_scans.csv] scan_id', '[LA_logbooks.csv] logbook_id',
        '[LA_logpages.csv] logpage_id', '[Database] use_gaia_fits',
        '[Database] use_tycho2_fits']


def ini(text):
    conf = configparser.ConfigParser()
    conf.read_string(text)
    return conf


def test_ini2toml_merges_files_and_sections(tmp_path):
    (tmp_path / 'a.conf').write_text('[Archive]\narchive_id = 5\n'
                                     'logbooks = LB1\n[LB1]\nlogbook_num = 1\n'
                                     '[Database]\noutput_db = DB_pgsql\n'
                                     '[DB_pgsql]\nrdbms = pgsql\n'
                                     'schema = applause_dr5\n'
                                     '[Keyword lookup]\nobserver = true\n'
                                     'emulsion = false\n'
                                     '[observer]\nJ. Mets = Jaan Mets\n'
                                     '[emulsion]\nx = y\n')
    (tmp_path / 'b.conf').write_text('[DB_pgsql]\nport = 5433\n'
                                     '[Programs]\nsextractor_path = /opt/sex\n')
    text, _ = convert(read_ini([str(tmp_path / 'a.conf'),
                                str(tmp_path / 'b.conf')]))
    cfg = config_from_dict(__import__('tomllib').loads(text))
    assert cfg.archive.logbooks == {'LB1': {'logbook_num': '1'}}
    assert cfg.database.connection.rdbms == 'pgsql'
    assert cfg.database.connection.port == 5433
    assert cfg.programs.sextractor_path == '/opt/sex'
    assert cfg.metadata.lookup == {'observer': {'j. mets': 'Jaan Mets'}}


@pytest.mark.parametrize('text,message', [
    ('[Files]\nfits_dri = x\n', "unknown key 'fits_dri'"),
    ('[Solve]\nthreshold = 3\n', "unknown key 'threshold'"),
    ('[Files]\nplate_csv = p.csv\n[p.csv]\nplate_id = 1\nobsrever = 2\n',
     "unknown key 'obsrever'"),
    ('[Keyword values]\nobservatory_name = x\n', "unknown key"),
    ('[Mystery]\na = 1\n', r'Unknown section \[Mystery\]'),
    ('[Solve]\nsip = three\n', 'cannot convert'),
])
def test_ini2toml_stops_on_unknown_keys(text, message):
    with pytest.raises(ConfigError, match=message):
        convert(ini(text))


def test_ini2toml_csv_identifier_and_renames():
    text, _ = convert(ini('[Files]\npreview_csv = pv.csv\nscan_csv = s.csv\n'
                          '[pv.csv]\nplate_id = 2\n'
                          '[s.csv]\nfilename = 1\nplate_id = 2\n'
                          'plate_num = 2\ndatescan = 3\n'))
    data = __import__('tomllib').loads(text)
    # The old code took the identifier from the first column
    assert data['CSV']['pv.csv'] == {'filename': 1, 'plate_num': 2}
    assert data['CSV']['s.csv'] == {'filename_scan': 1, 'plate_num': 2,
                                    'scan_date': 3}


def test_load_config_values():
    cfg = load_config(os.path.join(VER4, 'plates', 'plates.toml'))
    assert cfg.archive.archive_id == 0
    assert cfg.pipeline.processes == 1
    assert cfg.pipeline.wait_start == 10.
    assert cfg.extraction.threshold_sigma == 3.
    assert cfg.extraction.use_psf is False
    assert cfg.solve.plate_epoch == 1970.
    assert cfg.crossmatch.crossmatch_nsigma == 10.
    assert cfg.database.dr_num == 5
    assert cfg.database.connection is None
    assert cfg.header.detector == 'photographic plate'
    assert cfg.metadata.keyword_values['site_elevation'] == 41
    assert isinstance(cfg.metadata.keyword_values['site_elevation'], int)
    table = cfg.csv['parsed_LA_UID_times.csv']
    assert table.kind == 'plate'
    assert table.csv_list_delimiter == ';'
    assert table.columns == {'plate_num': 1, 'tz_orig': 2,
                             'time_orig_start': 3, 'time_orig_end': 4}


def test_defaults():
    cfg = config_from_dict({})
    assert cfg.programs.sextractor_path == 'sex'
    assert cfg.solve.plate_epoch == 1950.
    assert cfg.wfpdb.encoding == 'utf-8'


@pytest.mark.parametrize('data,message', [
    ({'Files': {'fits_dri': 'x'}}, r"\[Files\]: unknown key 'fits_dri'"),
    ({'Mystery': {}}, r'Unknown section \[Mystery\]'),
    ({'Solve': {'sip': 'three'}}, r'\[Solve\] sip: expected int'),
    ({'Solve': {'use_psf': 1}}, r'\[Solve\] use_psf: expected bool'),
    ({'Archive': {'archive_id': 1.5}}, 'expected int'),
    ({'CSV': {'p.csv': {'plate_num': 1}}}, 'not named in'),
    ({'Files': {'plate_csv': ['p.csv']},
      'CSV': {'p.csv': {'plate_num': 1, 'scan_author': 2}}},
     "unknown key 'scan_author'"),
    ({'Files': {'plate_csv': ['p.csv']},
      'CSV': {'p.csv': {'plate_num': 1, 'exposure': {'observer': 2}}}},
     r'\.exposure\]: unknown key'),
    ({'Files': {'plate_csv': ['p.csv']},
      'CSV': {'p.csv': {'plate_id': 1}}}, "unknown key 'plate_id'"),
    ({'Keyword values': {'archive_id': 1}}, "unknown key 'archive_id'"),
    ({'Lookup': {'nonsense': {'a': 'b'}}}, "unknown key 'nonsense'"),
    ({'Archive': {'logbooks': ['LB1']}}, 'Logbook'),
    ({'Database': {'connection': {'rdbms': 'pgsql', 'hots': 'x'}}},
     r"\[Database\.connection\]: unknown key 'hots'"),
])
def test_unknown_keys_and_types(data, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict(data)
