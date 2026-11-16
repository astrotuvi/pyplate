"""CSV files mapped by column position (``[CSV."<file name>"]`` tables)."""
import csv
import os
from typing import Any

from ...config import CsvConfig
from ..timeutils import normalise_tz
from .values import LIST_KEYS, PlateValues
from .wfpdb import str_to_num

# Column that identifies the rows of each kind of CSV file
ID_COLUMNS = {'plate': 'plate_num', 'scan': 'filename_scan',
              'preview': 'filename', 'logpage': 'filename',
              'logbook': 'logbook_num'}


def read_csv_file(csv_dir: str, cfg: CsvConfig) -> dict[str, list[str]]:
    """Rows of a CSV file by the value of their identifier column.

    Empty lines are skipped; if an identifier occurs more than once, the
    last row is used.
    """
    pos = cfg.columns[ID_COLUMNS[cfg.kind]]

    with open(os.path.join(csv_dir, cfg.filename), encoding=cfg.csv_encoding,
              newline='') as f:
        reader = csv.reader(f, delimiter=cfg.csv_delimiter,
                            quotechar=cfg.csv_quotechar)
        return {row[pos - 1]: row for row in reader
                if row and len(row) >= pos}


def _value(raw: str, typ: Any, is_list: bool, list_delimiter: str) -> Any:
    numeric = typ in (int, float)

    if is_list:
        if numeric:
            return [str_to_num(x.strip()) for x in raw.split(list_delimiter)]

        val = [x.strip() for x in raw.split(list_delimiter)]
        return [] if val == [''] else val

    return str_to_num(raw) if numeric else raw


def apply_row(target: dict, row: list[str], cfg: CsvConfig,
              types: dict[str, Any], columns: dict[str, int] | None = None,
              list_keys: tuple = ()) -> None:
    """Write the mapped values of a CSV row into ``target``.

    Parameters
    ----------
    target
        Values by key; list values are replaced.
    row
        The CSV row.
    cfg
        The CSV table of the file.
    types
        Value type (``int``, ``float``, ``str``) by key; numbers are
        converted, an empty number becomes None.
    columns
        Key → position mapping to use instead of ``cfg.columns``.
    list_keys
        Keys whose values are lists (split at ``csv_list_delimiter``).
    """
    for key, pos in (cfg.columns if columns is None else columns).items():
        if not 0 < pos <= len(row):
            continue

        val = _value(row[pos - 1], types.get(key, str), key in list_keys,
                     cfg.csv_list_delimiter)

        if key == 'numexp' and not val:
            val = 1

        target[key] = val


def apply_plate_row(v: PlateValues, row: list[str], cfg: CsvConfig,
                    types: dict[str, Any], keep_plate_num: bool) -> None:
    """Write a plate CSV row into the plate values.

    The default time zone of the file (``csv_timezone``) comes before the
    columns, so a ``tz_orig`` column wins. With ``keep_plate_num``, the
    identifier column does not change ``plate_num`` (the row was found by
    the WFPDB ID).
    """
    if cfg.csv_timezone:
        v['tz_orig'] = normalise_tz(cfg.csv_timezone)

    columns = dict(cfg.columns)

    if keep_plate_num:
        columns.pop('plate_num', None)

    apply_row(v, row, cfg, types, columns, LIST_KEYS)

    if 'tz_orig' in columns and isinstance(v.get('tz_orig'), str):
        v['tz_orig'] = normalise_tz(v['tz_orig'])

    exposure = {}
    apply_row(exposure, row, cfg, types, cfg.exposure_columns, LIST_KEYS)
    v.exposure.update(exposure)
    v.combine_times()
