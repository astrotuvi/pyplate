"""SQL for creating and dropping the tables and indexes of a schema.

Each function returns a list of statements without the final semicolon.
"""
from .schema import Column, Schema, Table

MODES = ('create_schema', 'drop_schema', 'create_indexes', 'drop_indexes')

# Function used by the PostgreSQL update triggers
PGSQL_TIMESTAMP_FUNCTION = """CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
NEW.timestamp_update = NOW();
RETURN NEW;
END;
$$ LANGUAGE plpgsql"""

MYSQL_TABLE_OPTIONS = 'ENGINE=Aria CHARACTER SET=utf8 COLLATE=utf8_unicode_ci'

_MYSQL_TYPES = {'tinyint': 'TINYINT', 'smallint': 'SMALLINT', 'int': 'INT',
                'bigint': 'BIGINT', 'bool': 'TINYINT(1)', 'real': 'FLOAT',
                'double': 'DOUBLE', 'char': 'CHAR', 'varchar': 'VARCHAR',
                'text': 'TEXT', 'date': 'DATE', 'datetime': 'DATETIME'}

_PGSQL_TYPES = {'tinyint': 'smallint', 'smallint': 'smallint',
                'int': 'integer', 'bigint': 'bigint', 'bool': 'smallint',
                'real': 'real', 'double': 'double precision',
                'char': 'character', 'varchar': 'character varying',
                'text': 'text', 'date': 'date',
                'datetime': 'timestamp without time zone'}

_SQLITE_TYPES = {'tinyint': 'INTEGER', 'smallint': 'INTEGER',
                 'int': 'INTEGER', 'bigint': 'INTEGER', 'bool': 'INTEGER',
                 'real': 'REAL', 'double': 'REAL', 'char': 'TEXT',
                 'varchar': 'TEXT', 'text': 'TEXT', 'date': 'TEXT',
                 'datetime': 'TEXT', 'timestamp': 'TEXT'}


def _length(col: Column) -> str:
    return f'({col.length})' if col.length is not None else ''


def _mysql_type(col: Column) -> str:
    if col.type == 'serial':
        return 'INT UNSIGNED NOT NULL PRIMARY KEY AUTO_INCREMENT'

    if col.type == 'timestamp':
        # TIMESTAMP columns are NOT NULL unless declared NULL
        if col.default == 'now':
            return 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
        return 'TIMESTAMP NULL' if col.null else 'TIMESTAMP'

    sql = _MYSQL_TYPES[col.type] + _length(col)

    if col.unsigned:
        sql += ' UNSIGNED'

    if not col.null:
        sql += ' NOT NULL'

    if col.default is not None:
        sql += f' DEFAULT {col.default}'

    return sql


def _pgsql_type(col: Column) -> str:
    if col.type == 'serial':
        return 'serial'

    if col.type == 'timestamp':
        sql = 'timestamp with time zone'

        if not col.null:
            sql += ' not null'

        if col.default == 'now':
            sql += ' default NOW()'

        return sql

    sql = _PGSQL_TYPES[col.type] + _length(col)

    if col.type in ('char', 'varchar', 'text'):
        sql += ' COLLATE ucs_basic'

    if not col.null:
        sql += ' NOT NULL'

    if col.default is not None:
        sql += f' DEFAULT {col.default}'

    return sql


def _sqlite_type(col: Column) -> str:
    if col.type == 'serial':
        return 'INTEGER PRIMARY KEY'

    sql = _SQLITE_TYPES[col.type]

    if not col.null:
        sql += ' NOT NULL'

    if col.default == 'now':
        sql += ' DEFAULT CURRENT_TIMESTAMP'
    elif col.default is not None:
        sql += f' DEFAULT {col.default}'

    return sql


def column_type(col: Column, rdbms: str) -> str:
    """Return the column definition (after the name) for a backend."""
    if rdbms in col.overrides:
        return col.overrides[rdbms]

    if rdbms == 'mysql':
        return _mysql_type(col)
    if rdbms == 'pgsql':
        return _pgsql_type(col)
    if rdbms == 'sqlite':
        return _sqlite_type(col)

    raise ValueError(f'Unknown rdbms {rdbms!r}')


def qualified(schema: Schema, table: str, rdbms: str) -> str:
    """Table name with the schema prefix (none for SQLite)."""
    if rdbms == 'sqlite':
        return table

    return f'{schema.name}.{table}'


def _create_table(schema: Schema, table: Table, rdbms: str) -> str:
    lines = ['    {:20s} {}'.format(c.name, column_type(c, rdbms))
             for c in table.columns]
    serial = table.serial

    # SQLite: primary key inside CREATE TABLE, unless it is the serial column
    if (rdbms == 'sqlite' and table.primary_key and
            not (serial and table.primary_key == [serial.name])):
        lines.append('    PRIMARY KEY ({})'
                     .format(', '.join(table.primary_key)))

    sql = 'CREATE TABLE {} (\n{}\n)'.format(qualified(schema, table.name,
                                                      rdbms),
                                            ',\n'.join(lines))

    if rdbms == 'mysql':
        sql += ' ' + MYSQL_TABLE_OPTIONS

    return sql


def _trigger(schema: Schema, table: Table, rdbms: str) -> str | None:
    name = qualified(schema, table.name, rdbms)

    if rdbms == 'pgsql':
        return ('CREATE TRIGGER set_timestamp BEFORE UPDATE ON {} '
                'FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp()'
                .format(name))

    if rdbms == 'sqlite':
        # Recursive triggers are off by default, so the UPDATE inside the
        # trigger does not fire it again
        return ('CREATE TRIGGER {0}_set_timestamp AFTER UPDATE ON {0} '
                'FOR EACH ROW BEGIN UPDATE {0} SET timestamp_update = '
                'CURRENT_TIMESTAMP WHERE rowid = NEW.rowid; END'.format(name))

    # MySQL: no trigger (as in the DR4 schema)
    return None


def create_schema(schema: Schema, rdbms: str) -> list[str]:
    """CREATE statements for the schema, its tables and triggers."""
    sql = []

    if rdbms == 'pgsql':
        sql.append(f'CREATE schema {schema.name}')

        if any(t.timestamps for t in schema.tables.values()):
            sql.append(PGSQL_TIMESTAMP_FUNCTION)
    elif rdbms == 'mysql':
        sql.append(f'CREATE DATABASE {schema.name}')

    for table in schema.tables.values():
        sql.append(_create_table(schema, table, rdbms))

        if table.timestamps:
            trigger = _trigger(schema, table, rdbms)

            if trigger:
                sql.append(trigger)

    return sql


def drop_schema(schema: Schema, rdbms: str) -> list[str]:
    """DROP statements for the tables and the schema."""
    sql = ['DROP TABLE IF EXISTS {}'.format(qualified(schema, t, rdbms))
           for t in schema.tables]

    if rdbms == 'pgsql':
        sql.append(f'DROP schema {schema.name} CASCADE')
    elif rdbms == 'mysql':
        sql.append(f'DROP DATABASE {schema.name}')

    return sql


def create_indexes(schema: Schema, rdbms: str) -> list[str]:
    """Primary keys (PostgreSQL, MySQL) and indexes."""
    sql = []

    for table in schema.tables.values():
        name = qualified(schema, table.name, rdbms)

        if table.primary_key and rdbms != 'sqlite':
            sql.append('ALTER TABLE {} ADD CONSTRAINT {} PRIMARY KEY ({})'
                       .format(name, table.pk_name,
                               ', '.join(table.primary_key)))
            serial = table.serial

            if rdbms == 'mysql' and serial is not None:
                sql.append('ALTER TABLE {} MODIFY {} INT UNSIGNED NOT NULL '
                           'AUTO_INCREMENT'.format(name, serial.name))

        for index in table.indexes:
            sql.append('CREATE INDEX {}_{} ON {} ({})'
                       .format(table.name, index.name, name,
                               ', '.join(index.columns)))

    return sql


def drop_indexes(schema: Schema, rdbms: str) -> list[str]:
    """DROP INDEX statements for the indexes (not the primary keys)."""
    sql = []

    for table in schema.tables.values():
        for index in table.indexes:
            index_name = f'{table.name}_{index.name}'

            if rdbms == 'mysql':
                sql.append('DROP INDEX {} ON {}'.format(
                    index_name, qualified(schema, table.name, rdbms)))
            elif rdbms == 'pgsql':
                sql.append(f'DROP INDEX IF EXISTS {schema.name}.{index_name}')
            else:
                sql.append(f'DROP INDEX IF EXISTS {index_name}')

    return sql


def statements(schema: Schema, rdbms: str, mode: str) -> list[str]:
    """Statements for one of the modes in :data:`MODES`."""
    functions = {'create_schema': create_schema, 'drop_schema': drop_schema,
                 'create_indexes': create_indexes,
                 'drop_indexes': drop_indexes}

    if mode not in functions:
        raise ValueError(f'Unknown mode {mode!r}; one of {MODES}')

    return functions[mode](schema, rdbms)
