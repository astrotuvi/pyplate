"""Backend-neutral database schema, read from a schema YAML file.

The format is described in DESIGN.md, section 5.3. The SQL for each backend
is generated from the schema by :mod:`pyplate.database.ddl`.
"""
import os
import re
from dataclasses import dataclass, field

import yaml

SCHEMA_DIR = os.path.dirname(os.path.abspath(__file__))

BACKENDS = ('pgsql', 'mysql', 'sqlite')

TYPES = ('serial', 'tinyint', 'smallint', 'int', 'bigint', 'real', 'double',
         'bool', 'char', 'varchar', 'text', 'date', 'datetime', 'timestamp')

# Columns added to tables with `timestamps: true`
TIMESTAMP_COLUMNS = ('timestamp_insert', 'timestamp_update')

_COLUMN_KEYS = {'name', 'type', 'unsigned', 'null', 'default'} | set(BACKENDS)
_TABLE_KEYS = {'name', 'timestamps', 'primary_key', 'primary_key_name',
               'columns', 'indexes'}


@dataclass
class Column:
    """A table column.

    ``type`` is the base type (for example ``varchar``) and ``length`` its
    length, if any. ``overrides`` holds the full column definitions given for
    single backends (the ``pgsql:``, ``mysql:`` and ``sqlite:`` keys).
    """

    name: str
    type: str
    length: int | None = None
    unsigned: bool = False
    null: bool = True
    default: str | None = None
    overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class Index:
    """A table index. The index name in SQL is ``<table>_<name>``."""

    name: str
    columns: list[str]


@dataclass
class Table:
    """A table: columns in order, primary key and indexes."""

    name: str
    columns: list[Column]
    primary_key: list[str] = field(default_factory=list)
    primary_key_name: str | None = None
    indexes: list[Index] = field(default_factory=list)
    timestamps: bool = False

    @property
    def column_names(self) -> list[str]:
        """Names of all columns, including the generated timestamps."""
        return [c.name for c in self.columns]

    def column(self, name: str) -> Column:
        """Return the column with the given name."""
        for c in self.columns:
            if c.name == name:
                return c

        raise KeyError(f'No column {name} in table {self.name}')

    @property
    def serial(self) -> Column | None:
        """The auto-incremented column, if the table has one."""
        for c in self.columns:
            if c.type == 'serial':
                return c

        return None

    @property
    def pk_name(self) -> str:
        """Name of the primary-key constraint."""
        return self.primary_key_name or f'{self.name}_pky'


@dataclass
class Schema:
    """A database schema: its name and its tables in order."""

    name: str
    tables: dict[str, Table]

    def __getitem__(self, table: str) -> Table:
        return self.tables[table]

    def __contains__(self, table: str) -> bool:
        return table in self.tables


def _parse_column(d: dict, table: str) -> Column:
    # YAML reads the key `null` as None
    if None in d:
        d = {('null' if k is None else k): v for k, v in d.items()}

    unknown = set(d) - _COLUMN_KEYS

    if unknown:
        raise ValueError(f'Unknown keys {sorted(unknown)} in column '
                         f'{d.get("name")} of table {table}')

    m = re.fullmatch(r'(\w+)(?:\((\d+)\))?', d['type'])

    if m is None or m.group(1) not in TYPES:
        raise ValueError(f'Unknown type {d["type"]!r} in column {d["name"]} '
                         f'of table {table}')

    default = d.get('default')

    return Column(name=d['name'], type=m.group(1),
                  length=int(m.group(2)) if m.group(2) else None,
                  unsigned=d.get('unsigned', False), null=d.get('null', True),
                  default=None if default is None else str(default),
                  overrides={b: d[b] for b in BACKENDS if b in d})


def _parse_table(d: dict) -> Table:
    unknown = set(d) - _TABLE_KEYS

    if unknown:
        raise ValueError(f'Unknown keys {sorted(unknown)} in table '
                         f'{d.get("name")}')

    columns = [_parse_column(c, d['name']) for c in d['columns']]

    if d.get('timestamps', False):
        columns += [Column(name=n, type='timestamp', null=False,
                           default='now') for n in TIMESTAMP_COLUMNS]

    return Table(name=d['name'], columns=columns,
                 primary_key=list(d.get('primary_key', [])),
                 primary_key_name=d.get('primary_key_name'),
                 indexes=[Index(i['name'], list(i['columns']))
                          for i in d.get('indexes', [])],
                 timestamps=d.get('timestamps', False))


def load_schema(yaml_file: str, name: str | None = None) -> Schema:
    """Read a schema YAML file.

    Parameters
    ----------
    yaml_file
        Path of the YAML file, or the name of a file in the
        ``pyplate/database`` directory (``applause_dr5.yaml``).
    name
        Schema name to use instead of the one in the file.

    Raises
    ------
    ValueError
        If the file has unknown keys or types.
    """
    if not os.path.exists(yaml_file):
        yaml_file = os.path.join(SCHEMA_DIR, yaml_file)

    with open(yaml_file) as f:
        d = yaml.safe_load(f)

    tables = [_parse_table(t) for t in d['tables']]
    return Schema(name=name or d['schema'], tables={t.name: t for t in tables})
