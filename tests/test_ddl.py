"""Generated table and index SQL (DESIGN.md sections 5.3 and 7.2).

DR4: the SQL generated from the neutral applause_dr4.yaml must equal the SQL
of the old code in tests/ver4/ddl, statement by statement with whitespace
normalised, except DROP INDEX, which is corrected on purpose.

DR5: the PostgreSQL and MySQL SQL is compared with the reviewed copy in
tests/data/ddl_dr5 (only SQLite executes DR5 in the tests).
"""
import os
import re

import pytest

from pyplate.database import ddl
from pyplate.database.schema import Column, load_schema

HERE = os.path.dirname(os.path.abspath(__file__))
DDL_DR4 = os.path.join(HERE, 'ver4', 'ddl')
DDL_DR5 = os.path.join(HERE, 'data', 'ddl_dr5')


def split_statements(text):
    """Statements of an SQL text, split at semicolons, whitespace normalised.

    The PostgreSQL trigger function contains semicolons; splitting the
    generated and the expected SQL the same way still compares it fully.
    """
    return [' '.join(s.split()) for s in text.split(';') if s.strip()]


def generated(schema, rdbms, mode):
    return split_statements(';\n'.join(ddl.statements(schema, rdbms, mode)))


def read(path):
    with open(path) as f:
        return f.read()


@pytest.mark.parametrize('rdbms', ['pgsql', 'mysql'])
@pytest.mark.parametrize('mode', ['create_schema', 'drop_schema',
                                  'create_indexes'])
def test_dr4_same_as_old_code(rdbms, mode):
    schema = load_schema('applause_dr4.yaml')
    expected = split_statements(read(os.path.join(DDL_DR4, rdbms,
                                                  mode + '.sql')))
    assert generated(schema, rdbms, mode) == expected


@pytest.mark.parametrize('rdbms', ['pgsql', 'mysql'])
def test_dr4_drop_indexes_corrected(rdbms):
    # Old code: DROP INDEX IF EXISTS <name> for both backends. Expected:
    # the same index names in the same order, with the schema prefix
    # (PostgreSQL) or the table (MySQL).
    tables = dict(re.findall(r'CREATE INDEX (\w+) ON applause_dr4\.(\w+)',
                             read(os.path.join(DDL_DR4, rdbms,
                                               'create_indexes.sql'))))
    old = split_statements(read(os.path.join(DDL_DR4, rdbms,
                                             'drop_indexes.sql')))
    names = [s.split()[-1] for s in old]
    assert all(s == f'DROP INDEX IF EXISTS {n}' for s, n in zip(old, names))

    if rdbms == 'pgsql':
        expected = [f'DROP INDEX IF EXISTS applause_dr4.{n}' for n in names]
    else:
        expected = [f'DROP INDEX {n} ON applause_dr4.{tables[n]}'
                    for n in names]

    schema = load_schema('applause_dr4.yaml')
    assert generated(schema, rdbms, 'drop_indexes') == expected


@pytest.mark.parametrize('rdbms', ['pgsql', 'mysql'])
@pytest.mark.parametrize('mode', ddl.MODES)
def test_dr5_same_as_reviewed(rdbms, mode):
    schema = load_schema('applause_dr5.yaml')
    expected = split_statements(read(os.path.join(DDL_DR5, rdbms,
                                                  mode + '.sql')))
    assert generated(schema, rdbms, mode) == expected


def test_dr5_is_dr4_plus_changes():
    # DESIGN.md section 5.4: Gaia DR4 names (also in index names and the
    # source_xmatch primary key), two new columns, and no index on the
    # missing column source.exposure_id
    renames = {'gaiaedr3_id': 'gaiadr4_id', 'gaiaiedr3_gmag': 'gaiadr4_gmag',
               'gaiaedr3_bp_rp': 'gaiadr4_bp_rp',
               'gaiaedr3_dist': 'gaiadr4_dist',
               'gaiaedr3_neighbors': 'gaiadr4_neighbors',
               'num_gaia_edr3': 'num_gaia_dr4'}
    renamed_tables = {'gaiaedr3_id': {'source_calib', 'source_xmatch'},
                      'num_gaia_edr3': {'solution', 'phot_calib', 'process'}}
    new_columns = {'plate': ('date_orig', 'tz_orig', 'varchar', 40),
                   'exposure': ('time_orig_end', 'jd_mid_orig', 'double',
                                None)}

    schema = load_schema('applause_dr4.yaml')
    schema.name = 'applause_dr5'

    for table in schema.tables.values():
        for col in table.columns:
            if col.name in renames:
                tables = renamed_tables.get(col.name, {'source_calib'})
                assert table.name in tables
                col.name = renames[col.name]

        table.primary_key = [renames.get(c, c) for c in table.primary_key]

        for index in table.indexes:
            index.name = index.name.replace('gaiaedr3', 'gaiadr4')
            index.columns = [renames.get(c, c) for c in index.columns]

        if table.name in new_columns:
            after, name, typ, length = new_columns[table.name]
            pos = table.column_names.index(after) + 1
            table.columns.insert(pos, Column(name=name, type=typ,
                                             length=length))

    source = schema['source']
    source.indexes = [i for i in source.indexes if i.name != 'exposure_ind']

    dr5 = load_schema('applause_dr5.yaml')
    assert list(dr5.tables) == list(schema.tables)
    assert dr5 == schema


def test_dr5_keys_and_indexes_have_columns():
    for table in load_schema('applause_dr5.yaml').tables.values():
        for col in table.primary_key + [c for i in table.indexes
                                        for c in i.columns]:
            assert col in table.column_names, (table.name, col)


def test_schema_name_override():
    schema = load_schema('applause_dr5.yaml', name='my_schema')
    sql = ddl.create_schema(schema, 'pgsql')
    assert sql[0] == 'CREATE schema my_schema'
    assert sql[2].startswith('CREATE TABLE my_schema.archive (')


def test_unknown_column_key(tmp_path):
    fn = tmp_path / 'bad.yaml'
    fn.write_text('schema: s\ntables:\n- name: t\n  columns:\n'
                  '  - {name: a, type: int, nul: false}\n')

    with pytest.raises(ValueError, match="nul"):
        load_schema(str(fn))
