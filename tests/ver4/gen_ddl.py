"""Generate the DR4 schema SQL into tests/ver4/ddl (no DB connection).

Runs against the pyplate code as of commit ed2a193 (DR4 schema, before the
redesign described in DESIGN.md). Run from the repository root:

    PYTHONPATH=. python tests/ver4/gen_ddl.py
"""
import os
from pyplate.database.db_pgsql import DB_pgsql
from pyplate.database.db_mysql import DB_mysql

out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ddl')
modes = ['create_schema', 'drop_schema', 'create_indexes', 'drop_indexes']

for name, cls in [('pgsql', DB_pgsql), ('mysql', DB_mysql)]:
    # Bypass __init__, which needs the database driver
    db = cls.__new__(cls)
    db.schema = None
    db.yaml = 'applause_dr4.yaml'
    db.schema_dict = None
    db.trigger_dict = None
    db.index_dict = None

    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    for mode in modes:
        sql = db.get_schema_sql(mode=mode)
        with open(os.path.join(out_dir, mode + '.sql'), 'w') as f:
            f.write(sql)
        print(name, mode, len(sql))
