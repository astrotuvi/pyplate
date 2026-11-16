"""Read a TOML configuration file into a :class:`Config`.

Every key is checked: a key that its section does not accept, or a value of
the wrong type, raises :class:`ConfigError` naming the section and the key.
"""
import dataclasses
import tomllib
import types
import typing
from typing import Any

from .sections import (ArchiveConfig, CatalogConfig, Config, ConnectionConfig,
                       CrossmatchConfig, CsvConfig, DatabaseConfig,
                       ExtractionConfig, FilesConfig, HeaderConfig,
                       ImageConfig, MetadataConfig, PipelineConfig,
                       ProgramsConfig, SolveConfig, WfpdbConfig)


class ConfigError(ValueError):
    """An error in a configuration file."""


# Sections read directly into one dataclass
SIMPLE_SECTIONS = {'Files': FilesConfig, 'WFPDB': WfpdbConfig,
                   'Image': ImageConfig, 'Programs': ProgramsConfig,
                   'Pipeline': PipelineConfig}

# [Solve] is split between the components that use it
SOLVE_SECTIONS = {'extraction': ExtractionConfig, 'solve': SolveConfig,
                  'crossmatch': CrossmatchConfig}

# Header sections with fixed FITS keyword values or comments
HEADER_VALUE_SECTIONS = {'Integer values': ('integer_values', int),
                         'Float values': ('float_values', float),
                         'String values': ('string_values', str),
                         'FITS keyword comments': ('keyword_comments', str)}

SECTIONS = (set(SIMPLE_SECTIONS) | set(HEADER_VALUE_SECTIONS) |
            {'Archive', 'Logbook', 'CSV', 'Metadata', 'Keyword values',
             'Lookup', 'Header', 'FITS keyword order', 'Solve', 'Catalog',
             'Database'})

# CSV parsing settings (not columns)
CSV_SETTINGS = ('csv_delimiter', 'csv_quotechar', 'csv_list_delimiter',
                'csv_timezone', 'csv_encoding')

# [Files] lists that name CSV files, and the kind of each
CSV_KINDS = {'plate_csv': 'plate', 'scan_csv': 'scan',
             'preview_csv': 'preview', 'logbook_csv': 'logbook',
             'logpage_csv': 'logpage'}

# Plate CSV keys that are not columns: logpage, preview and scan links, and
# start and end times given as UT or sidereal time (DESIGN.md section 4.2)
PLATE_LINK_KEYS = ('fn_cover', 'fn_log', 'fn_pre', 'fn_scan')
PLATE_TIME_KEYS = ('ut_start_orig', 'ut_end_orig', 'st_start_orig',
                   'st_end_orig')


def _type_name(t: Any) -> str:
    return getattr(t, '__name__', str(t))


def convert(value: Any, annotation: Any, where: str,
            keep_int: bool = False) -> Any:
    """Check a TOML value against a field type and convert it.

    Integers are accepted for floats and converted, unless ``keep_int``
    (metadata values keep the type of the input, as in the old code).
    ``X | None`` accepts ``X``.

    Raises
    ------
    ConfigError
        If the value does not have the type.
    """
    if isinstance(annotation, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        annotation = args[0]

    origin = typing.get_origin(annotation)

    if origin is list:
        (item,) = typing.get_args(annotation)

        if not isinstance(value, list):
            raise ConfigError(f'{where}: expected a list, got {value!r}')

        return [convert(v, item, where, keep_int) for v in value]

    if annotation is Any:
        return value

    if annotation is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        value = float(value) if ok and not keep_int else value
    elif annotation is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    else:
        ok = isinstance(value, annotation)

    if not ok:
        raise ConfigError(f'{where}: expected {_type_name(annotation)}, '
                          f'got {value!r}')

    return value


def field_types(cls: type) -> dict[str, Any]:
    """Field names and types of a dataclass."""
    return {f.name: f.type for f in dataclasses.fields(cls)}


def _build(cls: type, table: dict, section: str, skip: tuple = ()) -> Any:
    types_ = field_types(cls)
    values = {}

    for key, value in table.items():
        if key in skip:
            continue

        if key not in types_ or isinstance(value, dict):
            raise ConfigError(f'[{section}]: unknown key {key!r}')

        values[key] = convert(value, types_[key], f'[{section}] {key}')

    try:
        return cls(**values)
    except TypeError as e:
        raise ConfigError(f'[{section}]: {e}') from None


def metadata_columns() -> dict[str, dict[str, Any]]:
    """Accepted column keys and their types, by record kind.

    Database IDs (``*_id`` except ``wfpdb_id``) and ``exposure_num`` are not
    read from input files (DESIGN.md section 4.2).
    """
    # Imported here: the metadata subpackage imports the config sections
    from ..metadata import (Exposure, Logbook, Logpage, Plate, Preview,
                            Scan)

    def columns(cls):
        types_ = field_types(cls)
        return {c: types_[c] for c in cls.columns()
                if not (c.endswith('_id') and c != 'wfpdb_id')
                and c != 'exposure_num'}

    return {'plate': columns(Plate), 'exposure': columns(Exposure),
            'scan': columns(Scan), 'preview': columns(Preview),
            'logbook': columns(Logbook), 'logpage': columns(Logpage)}


def csv_keys(kind: str, cols: dict[str, dict[str, Any]]) -> set[str]:
    """Keys accepted in a CSV table of the given kind (besides settings)."""
    if kind == 'plate':
        return (set(cols['plate']) | set(cols['exposure']) |
                set(PLATE_LINK_KEYS) | set(PLATE_TIME_KEYS))
    if kind in ('scan', 'preview'):
        return set(cols[kind]) | {'plate_num', 'wfpdb_id'}
    if kind == 'logpage':
        return set(cols['logpage']) | {'logbook_num'}
    return set(cols[kind])


def plate_value_columns(cols: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Columns that ``["Keyword values"]`` and lookups can set."""
    return {**cols['scan'], **cols['exposure'], **cols['plate']}


def _csv_tables(data: dict, files: FilesConfig,
                cols: dict[str, dict[str, Any]]) -> dict[str, CsvConfig]:
    kinds = {}

    for attr, kind in CSV_KINDS.items():
        for fn in getattr(files, attr):
            kinds[fn] = kind

    out = {}

    for fn, table in data.items():
        section = f'CSV."{fn}"'

        if not isinstance(table, dict):
            raise ConfigError(f'[CSV]: unknown key {fn!r}')

        if fn not in kinds:
            raise ConfigError(f'[{section}]: the file is not named in '
                              f'[Files]')

        kind = kinds[fn]
        accepted = csv_keys(kind, cols)
        settings = {}
        columns = {}
        exposure_columns = {}

        for key, value in table.items():
            where = f'[{section}] {key}'

            if key in CSV_SETTINGS:
                settings[key] = convert(value, field_types(CsvConfig)[key],
                                        where)
            elif key == 'exposure' and kind == 'plate':
                if not isinstance(value, dict):
                    raise ConfigError(f'{where}: expected a table')

                for k, v in value.items():
                    if k not in cols['exposure']:
                        raise ConfigError(f'[{section}.exposure]: unknown '
                                          f'key {k!r}')

                    exposure_columns[k] = convert(
                        v, int, f'[{section}.exposure] {k}')
            elif key in accepted:
                columns[key] = convert(value, int, where)
            else:
                raise ConfigError(f'[{section}]: unknown key {key!r}')

        out[fn] = CsvConfig(filename=fn, kind=kind, columns=columns,
                            exposure_columns=exposure_columns, **settings)

    return out


def _archive(data: dict, cols: dict[str, dict[str, Any]]) -> ArchiveConfig:
    archive = _build(ArchiveConfig, data.get('Archive', {}), 'Archive',
                     skip=('logbooks',))

    if 'logbooks' in data.get('Archive', {}):
        raise ConfigError('[Archive]: unknown key \'logbooks\' (logbooks '
                          'are [Logbook.<name>] tables)')

    logbooks = {}

    for name, table in data.get('Logbook', {}).items():
        if not isinstance(table, dict):
            raise ConfigError(f'[Logbook]: unknown key {name!r}')

        values = {}

        for key, value in table.items():
            if key not in cols['logbook']:
                raise ConfigError(f'[Logbook.{name}]: unknown key {key!r}')

            values[key] = convert(value, cols['logbook'][key],
                                  f'[Logbook.{name}] {key}', keep_int=True)

        logbooks[name] = values

    return dataclasses.replace(archive, logbooks=logbooks)


def _metadata(data: dict, cols: dict[str, dict[str, Any]]) -> MetadataConfig:
    meta = _build(MetadataConfig, data.get('Metadata', {}), 'Metadata',
                  skip=())
    accepted = plate_value_columns(cols)
    keyword_values = {}

    for key, value in data.get('Keyword values', {}).items():
        if key not in accepted:
            raise ConfigError(f'["Keyword values"]: unknown key {key!r}')

        keyword_values[key] = convert(value, accepted[key],
                                      f'["Keyword values"] {key}',
                                      keep_int=True)

    lookup = {}

    for key, table in data.get('Lookup', {}).items():
        if key not in accepted or not isinstance(table, dict):
            raise ConfigError(f'[Lookup]: unknown key {key!r}')

        lookup[key] = {k.lower(): convert(v, str, f'[Lookup.{key}] {k}')
                       for k, v in table.items()}

    return dataclasses.replace(meta, keyword_values=keyword_values,
                               lookup=lookup)


def _header(data: dict) -> HeaderConfig:
    header = _build(HeaderConfig, data.get('Header', {}), 'Header',
                    skip=())
    values = {}

    for section, (attr, typ) in HEADER_VALUE_SECTIONS.items():
        values[attr] = {k.upper(): convert(v, typ, f'["{section}"] {k}')
                        for k, v in data.get(section, {}).items()}

    order = data.get('FITS keyword order')

    if order is not None:
        unknown = set(order) - {'keywords'}

        if unknown:
            raise ConfigError(f'["FITS keyword order"]: unknown key '
                              f'{sorted(unknown)[0]!r}')

        values['keyword_order'] = convert(order.get('keywords', []),
                                          list[str],
                                          '["FITS keyword order"] keywords')

    return dataclasses.replace(header, **values)


def _with_connection(cls: type, table: dict, section: str) -> Any:
    conn = table.get('connection')
    obj = _build(cls, table, section, skip=('connection',))

    if conn is not None:
        if not isinstance(conn, dict):
            raise ConfigError(f'[{section}] connection: expected a table')

        obj = dataclasses.replace(obj, connection=_build(
            ConnectionConfig, conn, f'{section}.connection'))

    return obj


def _solve(table: dict) -> dict[str, Any]:
    known = {}

    for attr, cls in SOLVE_SECTIONS.items():
        for name in field_types(cls):
            known[name] = attr

    split = {attr: {} for attr in SOLVE_SECTIONS}

    for key, value in table.items():
        if key not in known:
            raise ConfigError(f'[Solve]: unknown key {key!r}')

        split[known[key]][key] = value

    return {attr: _build(cls, split[attr], 'Solve')
            for attr, cls in SOLVE_SECTIONS.items()}


def config_from_dict(data: dict) -> Config:
    """Build a :class:`Config` from parsed TOML data.

    Raises
    ------
    ConfigError
        For an unknown section or key, or a value of the wrong type.
    """
    for section in data:
        if section not in SECTIONS:
            raise ConfigError(f'Unknown section [{section}]')

    cols = metadata_columns()
    files = _build(FilesConfig, data.get('Files', {}), 'Files')
    simple = {name.lower(): _build(cls, data.get(name, {}), name)
              for name, cls in SIMPLE_SECTIONS.items() if name != 'Files'}

    return Config(archive=_archive(data, cols), files=files,
                  csv=_csv_tables(data.get('CSV', {}), files, cols),
                  metadata=_metadata(data, cols), header=_header(data),
                  database=_with_connection(DatabaseConfig,
                                            data.get('Database', {}),
                                            'Database'),
                  catalog=_with_connection(CatalogConfig,
                                           data.get('Catalog', {}),
                                           'Catalog'),
                  **_solve(data.get('Solve', {})), **simple)


def load_config(path: str) -> Config:
    """Read a TOML configuration file.

    Raises
    ------
    ConfigError
        For an unknown section or key, or a value of the wrong type.
    """
    with open(path, 'rb') as f:
        data = tomllib.load(f)

    return config_from_dict(data)
