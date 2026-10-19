"""Convert old INI configurations to the TOML format (DESIGN.md section 3.1).

Usage::

    python -m pyplate.config.ini2toml old.conf [more.conf ...] new.toml

Several input files are merged in order, as ``ConfigParser.read()`` does.
Keys are renamed to the new names (the column names of the database
tables), logbook, CSV and lookup sections become ``[Logbook.<name>]``,
``[CSV."<file name>"]`` and ``[Lookup.<keyword>]`` tables, and the section
named by ``[Database] output_db`` becomes ``[Database.connection]``.
Settings of removed features are dropped and listed. Any other key that the
new configuration does not accept stops the conversion. Comments are not
carried over.
"""
import configparser
import re
import sys
import tomllib
import types
import typing
from typing import Any

from .loader import (CSV_KINDS, CSV_SETTINGS, HEADER_VALUE_SECTIONS,
                     SIMPLE_SECTIONS, SOLVE_SECTIONS, ConfigError,
                     config_from_dict, csv_keys, field_types,
                     metadata_columns, plate_value_columns)
from .sections import (ArchiveConfig, CatalogConfig, ConnectionConfig,
                       CsvConfig, DatabaseConfig, MetadataConfig)

# Old metadata keys and their new names (the database columns)
EXPOSURE_RENAMES = {'date_orig': 'date_orig_start',
                    'tms_orig': 'time_orig_start',
                    'tme_orig': 'time_orig_end', 'coord_flag': 'flag_coord',
                    'time_flag': 'flag_time', 'ra_deg': 'ra_icrs',
                    'dec_deg': 'dec_icrs', 'date_obs': 'ut_start',
                    'date_avg': 'ut_mid', 'date_weighted': 'ut_weighted',
                    'date_end': 'ut_end', 'year': 'year_start',
                    'year_avg': 'year_mid', 'jd': 'jd_start',
                    'jd_avg': 'jd_mid', 'hjd_avg': 'hjd_mid',
                    'numsub': 'num_sub', 'jd_avg_orig': 'jd_mid_orig'}
SCAN_RENAMES = {'filename': 'filename_scan', 'fn_wedge': 'filename_wedge',
                'fits_naxis1': 'naxis1', 'fits_naxis2': 'naxis2',
                'fits_minval': 'minval', 'fits_maxval': 'maxval',
                'pix_size1': 'pixel_size1', 'pix_size2': 'pixel_size2',
                'datescan': 'scan_date', 'fits_datetime': 'file_datetime',
                'fits_size': 'file_size'}
CSV_RENAMES = {'plate': {'plate_id': 'plate_num', **EXPOSURE_RENAMES},
               'scan': {'plate_id': 'plate_num', **SCAN_RENAMES},
               'preview': {'plate_id': 'plate_num'},
               'logbook': {}, 'logpage': {}}
KEYWORD_VALUE_RENAMES = {**EXPOSURE_RENAMES, **SCAN_RENAMES}

# Column that identifies the rows; the old code always used the first
# column of the file
ID_COLUMNS = {'plate': 'plate_num', 'scan': 'filename_scan',
              'preview': 'filename', 'logpage': 'filename',
              'logbook': 'logbook_num'}

# [Keyword values] settings that move to [Header]
HEADER_KEYS = {'detector': 'detector', 'licence': 'licence',
               'fits_acknowledgements': 'acknowledgements'}

# Settings of removed features, with the reason
DROPPED = {
    'Database': dict.fromkeys(
        ['use_tycho2_fits', 'use_ucac4_db', 'use_apass_db', 'ucac4_db_host',
         'ucac4_db_user', 'ucac4_db_name', 'ucac4_db_passwd',
         'ucac4_db_table', 'apass_db_host', 'apass_db_user', 'apass_db_name',
         'apass_db_passwd', 'apass_db_table'],
        'Tycho-2, UCAC4 and APASS are no longer used')
    | {'use_gaia_fits': 'the Gaia file is always used'}
    | dict.fromkeys(['output_db_host', 'output_db_user', 'output_db_name',
                     'output_db_passwd'], 'not used'),
    'Files': {'tycho2_dir': 'Tycho-2 is no longer used'},
    'Solve': dict.fromkeys(['astref_catalog', 'photref_catalog'],
                           'Gaia is the only catalogue')
    | dict.fromkeys(['subfield_distort', 'max_recursion_depth',
                     'force_recursion_depth'],
                    'the sub-field recursion was removed'),
    'Programs': dict.fromkeys(['build_index_path', 'xy2sky_path'],
                              'never read by the old code'),
}
DROPPED_CSV = (dict.fromkeys(['scan_id', 'preview_id', 'logbook_id',
                              'logpage_id'],
                             'database IDs are not read from input files')
               | {'method': 'derived from method_code',
                  'skycond': 'never read by the old code (the column is '
                             'sky_conditions)'})
DROPPED_SECTIONS = {'Parse keywords': 'header parsing was removed',
                    'Rename FITS keywords': 'header renaming was removed'}

# Fields filled from other sections or tables, not from INI keys
NESTED_FIELDS = ('connection', 'logbooks', 'keyword_values', 'lookup')

# Output order of the sections
ORDER = ['Archive', 'Logbook', 'Files', 'CSV', 'WFPDB', 'Image', 'Metadata',
         'Keyword values', 'Lookup', 'Header', 'Integer values',
         'Float values', 'String values', 'FITS keyword comments',
         'FITS keyword order', 'Pipeline', 'Solve', 'Catalog', 'Programs',
         'Database']


class Converter:
    """Converts a ConfigParser with an old configuration."""

    def __init__(self, conf: configparser.ConfigParser):
        self.conf = conf
        self.out: dict[str, Any] = {}
        self.dropped: list[str] = []
        self.done: set[str] = set()
        self.cols = metadata_columns()

    # --- Values ----------------------------------------------------------

    def typed(self, value: str, typ: Any, where: str,
              keep_int: bool = False) -> Any:
        """Convert an INI string to the type of a field.

        With ``keep_int``, an integer for a float field stays an integer
        (metadata values, read by the old code as written).
        """
        if isinstance(typ, types.UnionType):
            typ = [a for a in typing.get_args(typ) if a is not type(None)][0]

        if keep_int and typ is float and re.fullmatch(r'\s*[+-]?\d+\s*',
                                                      value):
            typ = int

        try:
            if typing.get_origin(typ) is list:
                return [v.strip() for v in value.split(',') if v.strip()]
            if typ is bool:
                return self.conf.BOOLEAN_STATES[value.lower()]
            if typ is int:
                return int(value)
            if typ is float:
                return float(value)
        except (KeyError, ValueError):
            raise ConfigError(f'{where}: cannot convert {value!r} to '
                              f'{getattr(typ, "__name__", typ)}') from None

        return value

    def drop(self, section: str, key: str, reason: str) -> None:
        self.dropped.append(f'[{section}] {key}: {reason}')

    def section(self, name: str) -> list[tuple[str, str]]:
        self.done.add(name)
        return list(self.conf.items(name, raw=False))

    def simple(self, name: str, cls: type,
               skip: tuple = ()) -> dict[str, Any]:
        """A section read into one dataclass, with dropped keys removed.

        Keys in ``skip`` are left to the caller.
        """
        fields = field_types(cls)
        out = {}

        for key, value in self.section(name):
            if key in skip:
                continue
            elif key in DROPPED.get(name, {}):
                self.drop(name, key, DROPPED[name][key])
            elif key in fields and key not in NESTED_FIELDS:
                out[key] = self.typed(value, fields[key], f'[{name}] {key}')
            else:
                raise ConfigError(f'[{name}]: unknown key {key!r}')

        return out

    # --- Sections --------------------------------------------------------

    def connection(self, name: str, where: str) -> dict[str, Any]:
        if not self.conf.has_section(name):
            raise ConfigError(f'{where} names the section [{name}], which '
                              f'does not exist')

        return self.simple(name, ConnectionConfig)

    def archive(self) -> None:
        if not self.conf.has_section('Archive'):
            return

        types = field_types(ArchiveConfig)
        archive = {}
        logbooks = []

        for key, value in self.section('Archive'):
            if key == 'logbooks':
                logbooks = [lb.strip() for lb in value.split(',')
                            if lb.strip()]
            elif key in types and key != 'logbooks':
                archive[key] = self.typed(value, types[key],
                                          f'[Archive] {key}')
            else:
                raise ConfigError(f'[Archive]: unknown key {key!r}')

        self.out['Archive'] = archive
        cols = self.cols['logbook']

        for name in logbooks:
            values = {}

            for key, value in self.section(name):
                if key == 'logbook_id':
                    self.drop(name, key, 'database IDs are not read from '
                                         'input files')
                elif key in cols:
                    values[key] = self.typed(value, cols[key],
                                             f'[{name}] {key}', keep_int=True)
                else:
                    raise ConfigError(f'[{name}]: unknown key {key!r}')

            self.out.setdefault('Logbook', {})[name] = values

    def csv(self, files: dict[str, Any]) -> None:
        settings_types = field_types(CsvConfig)

        for attr, kind in CSV_KINDS.items():
            for fn in files.get(attr, []):
                section = fn.split('/')[-1]

                if not self.conf.has_section(section):
                    continue

                renames = CSV_RENAMES[kind]
                accepted = csv_keys(kind, self.cols)
                table = {}

                for key, value in self.section(section):
                    where = f'[{section}] {key}'

                    if key in CSV_SETTINGS:
                        table[key] = self.typed(value, settings_types[key],
                                                where)
                        continue

                    if key in DROPPED_CSV:
                        self.drop(section, key, DROPPED_CSV[key])
                        continue

                    new = renames.get(key, key)

                    if new not in accepted:
                        raise ConfigError(f'[{section}]: unknown key {key!r}')

                    pos = self.typed(value, int, where)

                    if new in table and table[new] != pos:
                        raise ConfigError(f'[{section}]: {key!r} and another '
                                          f'key both give {new!r}')

                    table[new] = pos

                # The old code took the identifier from the first column
                id_key = ID_COLUMNS[kind]

                if id_key not in table:
                    table = {id_key: 1, **table}

                self.out.setdefault('CSV', {})[fn] = table

    def keyword_values(self) -> None:
        if not self.conf.has_section('Keyword values'):
            return

        accepted = plate_value_columns(self.cols)
        values = {}
        header = {}

        for key, value in self.section('Keyword values'):
            if key in HEADER_KEYS:
                header[HEADER_KEYS[key]] = value
                continue

            new = KEYWORD_VALUE_RENAMES.get(key, key)

            if new not in accepted:
                raise ConfigError(f'[Keyword values]: unknown key {key!r}')

            values[new] = self.typed(value, accepted[new],
                                     f'[Keyword values] {key}', keep_int=True)

        if values:
            self.out['Keyword values'] = values

        if header:
            self.out.setdefault('Header', {}).update(header)

    def lookup(self) -> None:
        if not self.conf.has_section('Keyword lookup'):
            return

        accepted = plate_value_columns(self.cols)

        for key, value in self.section('Keyword lookup'):
            if not self.typed(value, bool, f'[Keyword lookup] {key}'):
                if self.conf.has_section(key):
                    self.done.add(key)
                    self.dropped.append(f'[{key}]: the lookup is switched '
                                        f'off in [Keyword lookup]')

                continue

            new = KEYWORD_VALUE_RENAMES.get(key, key)

            if new not in accepted:
                raise ConfigError(f'[Keyword lookup]: unknown key {key!r}')

            if self.conf.has_section(key):
                self.out.setdefault('Lookup', {})[new] = dict(
                    self.section(key))

    def header_values(self) -> None:
        types = {'Integer values': int, 'Float values': float,
                 'String values': str, 'FITS keyword comments': str}

        for name in HEADER_VALUE_SECTIONS:
            if self.conf.has_section(name):
                self.out[name] = {k.upper(): self.typed(v, types[name],
                                                        f'[{name}] {k}')
                                  for k, v in self.section(name)}

        if self.conf.has_section('FITS keyword order'):
            items = dict(self.section('FITS keyword order'))
            unknown = set(items) - {'keywords'}

            if unknown:
                raise ConfigError(f'[FITS keyword order]: unknown key '
                                  f'{sorted(unknown)[0]!r}')

            self.out['FITS keyword order'] = {
                'keywords': [k.strip() for k in
                             items.get('keywords', '').split('\n')
                             if k.strip()]}

    def database(self) -> None:
        if not self.conf.has_section('Database'):
            return

        items = dict(self.conf.items('Database'))
        output_db = items.pop('output_db', None)
        self.out['Database'] = self.simple('Database', DatabaseConfig,
                                           skip=('output_db',))

        for table in ['ucac4_db_table', 'apass_db_table']:
            name = items.get(table, table.split('_')[0])

            if self.conf.has_section(name):
                self.done.add(name)
                self.dropped.append(f'[{name}]: column names of a removed '
                                    f'catalogue')

        if output_db:
            self.out['Database']['connection'] = self.connection(
                output_db, '[Database] output_db')

    def catalog(self) -> None:
        if not self.conf.has_section('Catalog'):
            return

        items = dict(self.conf.items('Catalog'))
        db = items.pop('db', None)
        self.out['Catalog'] = self.simple('Catalog', CatalogConfig,
                                          skip=('db',))

        if db:
            self.out['Catalog']['connection'] = self.connection(
                db, '[Catalog] db')

    def solve(self) -> None:
        if not self.conf.has_section('Solve'):
            return

        types = {}

        for cls in SOLVE_SECTIONS.values():
            types.update(field_types(cls))

        out = {}

        for key, value in self.section('Solve'):
            if key in DROPPED['Solve']:
                self.drop('Solve', key, DROPPED['Solve'][key])
            elif key in types:
                out[key] = self.typed(value, types[key], f'[Solve] {key}')
            else:
                raise ConfigError(f'[Solve]: unknown key {key!r}')

        self.out['Solve'] = out

    def convert(self) -> dict[str, Any]:
        """Return the new configuration as TOML data.

        Raises
        ------
        ConfigError
            For a key or section that the new configuration does not
            accept.
        """
        self.archive()

        if self.conf.has_section('Files'):
            self.out['Files'] = self.simple('Files', SIMPLE_SECTIONS['Files'])
            self.csv(self.out['Files'])

        for name, cls in SIMPLE_SECTIONS.items():
            if name != 'Files' and self.conf.has_section(name):
                self.out[name] = self.simple(name, cls)

        if self.conf.has_section('Metadata'):
            self.out['Metadata'] = self.simple('Metadata', MetadataConfig)

        self.keyword_values()
        self.lookup()
        self.header_values()
        self.solve()
        self.catalog()
        self.database()

        for name, reason in DROPPED_SECTIONS.items():
            if self.conf.has_section(name):
                self.done.add(name)
                self.dropped.append(f'[{name}]: {reason}')

        for name in self.conf.sections():
            if name not in self.done:
                raise ConfigError(f'Unknown section [{name}]')

        # The result must be a valid configuration
        config_from_dict(self.out)
        return {k: self.out[k] for k in ORDER if k in self.out}


# --- TOML text ---------------------------------------------------------------

def _key(key: str) -> str:
    if re.fullmatch(r'[A-Za-z0-9_-]+', key):
        return key

    return _string(key)


def _string(s: str) -> str:
    out = []

    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == '\\':
            out.append('\\\\')
        elif ch == '\n':
            out.append('\\n')
        elif ch == '\t':
            out.append('\\t')
        elif ord(ch) < 32 or ord(ch) == 127:
            out.append('\\u{:04x}'.format(ord(ch)))
        else:
            out.append(ch)

    return '"' + ''.join(out) + '"'


def _value(v: Any) -> str:
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return _string(v)
    if isinstance(v, list):
        return '[' + ', '.join(_value(x) for x in v) + ']'

    raise TypeError(v)


def _table(lines: list[str], path: list[str], table: dict) -> None:
    lines.append('[{}]'.format('.'.join(_key(p) for p in path)))
    subtables = []

    for key, value in table.items():
        if isinstance(value, dict):
            subtables.append((key, value))
        else:
            lines.append('{} = {}'.format(_key(key), _value(value)))

    lines.append('')

    for key, value in subtables:
        _table(lines, path + [key], value)


def to_toml(data: dict[str, Any]) -> str:
    """TOML text of the converted configuration.

    Tables whose values are all tables (``Logbook``, ``CSV``, ``Lookup``)
    are written as their sub-tables only.
    """
    lines = []

    for name, table in data.items():
        if table and all(isinstance(v, dict) for v in table.values()):
            for key, value in table.items():
                _table(lines, [name, key], value)
        else:
            _table(lines, [name], table)

    return '\n'.join(lines).rstrip('\n') + '\n'


def read_ini(filenames: list[str]) -> configparser.ConfigParser:
    """Read INI files in order, as the old ``read_conf()`` did."""
    conf = configparser.ConfigParser()

    for fn in filenames:
        with open(fn) as f:
            conf.read_file(f, fn)

    return conf


def convert(conf: configparser.ConfigParser) -> tuple[str, list[str]]:
    """Convert an old configuration.

    Returns
    -------
    text
        The TOML text.
    dropped
        Descriptions of the dropped settings.

    Raises
    ------
    ConfigError
        For a key or section that the new configuration does not accept.
    """
    converter = Converter(conf)
    data = converter.convert()
    text = to_toml(data)
    assert tomllib.loads(text) == data
    return text, converter.dropped


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv

    if len(args) < 2:
        sys.exit(__doc__.split('Usage::')[1].split('Several')[0].strip())

    try:
        text, dropped = convert(read_ini(args[:-1]))
    except ConfigError as e:
        sys.exit('Error: {}'.format(e))

    with open(args[-1], 'w') as f:
        f.write(text)

    for line in dropped:
        print('Dropped', line)


if __name__ == '__main__':
    main()
