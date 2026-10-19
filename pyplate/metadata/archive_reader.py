"""ArchiveReader: reads the metadata input files of an archive."""
import os
from collections.abc import Iterator
from typing import Any

from ..config import Config, CsvConfig
from ..config.loader import field_types
from .archive import Archive
from .exposure import Exposure
from .logbook import Logbook
from .logpage import Logpage
from .plate import Plate
from .preview import Preview
from .readers import wfpdb
from .readers.csv import apply_plate_row, apply_row, read_csv_file
from .readers.exif import format_from_name, image_values
from .readers.fits import read_fits
from .readers.values import EXPOSURE_LIST_COLUMNS, LIST_KEYS, PlateValues
from .scan import Scan
from .subexposure import SubExposure
from .timeutils import split_orig_times

# Computed exposure values: a single value is not copied to the other
# exposures (DESIGN.md section 4.3)
COMPUTED_COLUMNS = ('ut_start', 'ut_mid', 'ut_weighted', 'ut_end',
                    'jd_start', 'jd_mid', 'jd_weighted', 'jd_end',
                    'hjd_mid', 'hjd_weighted', 'year_start', 'year_mid',
                    'year_weighted', 'year_end')

# A single value of these gives the first exposure only, when the plate
# has a date and several exposures
FIRST_EXPOSURE_COLUMNS = ('time_orig_start', 'time_orig_end', 'exptime',
                          'jd_mid_orig')

_IDS = ('plate_id', 'exposure_id', 'archive_id', 'exposure_num')


def _types() -> dict[str, Any]:
    """Value types of all keys that the input files can give."""
    types = {}

    for cls in [Scan, Exposure, Plate, Preview, Logpage, Logbook]:
        for name, typ in field_types(cls).items():
            args = getattr(typ, '__args__', (typ,))
            types[name] = args[0]

    types.update({'exptime': float, 'jd_mid_orig': float, 'focus': float,
                  'num_sub': int})
    return types


TYPES = _types()


def _spread(values: list, i: int, column: str) -> Any:
    if len(values) > i:
        return values[i]

    if len(values) == 1 and column not in COMPUTED_COLUMNS:
        return values[0]

    return None


class ArchiveReader:
    """Reads the WFPDB and CSV files named in the configuration.

    Values are merged in the order WFPDB maindata, quality, observer,
    notes, plate CSV files, scan CSV file, ``["Keyword values"]`` and
    ``[Lookup.<keyword>]``; later sources win. ``archive_id`` is copied into
    every record.

    Parameters
    ----------
    cfg
        The configuration.
    read_wfpdb, read_csv
        Read the WFPDB files and the CSV files named in ``[Files]``.
    """

    def __init__(self, cfg: Config, read_wfpdb: bool = True,
                 read_csv: bool = True):
        self.cfg = cfg
        files = cfg.files
        self.archive_id = cfg.archive.archive_id
        names = {'maindata': files.wfpdb_maindata,
                 'quality': files.wfpdb_quality,
                 'notes': files.wfpdb_notes,
                 'observer': files.wfpdb_observer}

        if not read_wfpdb:
            names = {k: None for k in names}

        self.wfpdb = wfpdb.read_files(files.wfpdb_dir, names,
                                      cfg.wfpdb.encoding)
        self.csv: dict[str, list[tuple[CsvConfig, dict]]] = {
            kind: [] for kind in ['plate', 'scan', 'preview', 'logbook',
                                  'logpage']}

        if read_csv:
            for attr in ['plate_csv', 'scan_csv', 'preview_csv',
                         'logbook_csv', 'logpage_csv']:
                for fn in getattr(files, attr):
                    table = cfg.csv.get(fn)

                    if table is None:
                        raise ValueError('No [CSV."{}"] table for the file '
                                         'in [Files] {}'.format(fn, attr))

                    self.csv[table.kind].append(
                        (table, read_csv_file(files.csv_dir, table)))

        self.archive = Archive(archive_id=self.archive_id,
                               archive_name=cfg.archive.archive_name,
                               institute=cfg.archive.institute,
                               logbooks=self._logbooks())

    # --- Rows ----------------------------------------------------------------

    def _rows(self, kind: str, key: str | None) -> list[tuple[CsvConfig,
                                                               list]]:
        if not key:
            return []

        return [(table, rows[key]) for table, rows in self.csv[kind]
                if key in rows]

    def _linked_rows(self, kind: str, keys: set) -> list[tuple[CsvConfig,
                                                                list]]:
        """Rows of scan or preview CSV files linked to one of the plate
        identifiers in ``keys``."""
        out = []

        for table, rows in self.csv[kind]:
            for row in rows.values():
                link = {}
                apply_row(link, row, table, TYPES,
                          {k: p for k, p in table.columns.items()
                           if k in ('plate_num', 'wfpdb_id')})

                if keys & {v for v in link.values() if v}:
                    out.append((table, row))

        return out

    def _record_values(self, kind: str, rows) -> dict:
        values = {}

        for table, row in rows:
            apply_row(values, row, table, TYPES)

        return values

    # --- Related records -------------------------------------------------------

    def _logbooks(self) -> list[Logbook]:
        if self.cfg.archive.logbooks:
            return [Logbook(archive_id=self.archive_id, **values)
                    for values in self.cfg.archive.logbooks.values()]

        return [self._logbook(num) for _, rows in self.csv['logbook']
                for num in rows]

    def _logbook(self, num: str) -> Logbook:
        values = self._record_values('logbook', self._rows('logbook', num))
        cols = Logbook.columns()
        return Logbook(**{k: v for k, v in values.items() if k in cols},
                       archive_id=self.archive_id)

    def _scan(self, filename: str, values: dict | None = None) -> Scan:
        """Scan from its scan CSV row, or from the given values (FITS file
        and CSV row), and ``["Keyword values"]``."""
        if values is None:
            values = {}

            for table, row in self._rows('scan', filename):
                apply_row(values, row, table, TYPES,
                          {k: p for k, p in table.columns.items()
                           if k not in ('plate_num', 'wfpdb_id')})

        values = dict(values)
        self._keyword_values(values, Scan.columns())
        cols = Scan.columns()
        scan = Scan(**{k: v for k, v in values.items() if k in cols})
        scan.archive_id = self.archive_id

        if scan.filename_scan is None:
            scan.filename_scan = filename

        return scan

    def _preview(self, filename: str) -> Preview:
        values = self._record_values('preview', self._rows('preview',
                                                           filename))
        values['filename'] = filename
        values.setdefault('file_format', format_from_name(filename))
        values.update(image_values(
            os.path.join(self.cfg.files.preview_dir, filename),
            self.cfg.image.preview_exif_timezone))
        cols = Preview.columns()
        preview = Preview(**{k: v for k, v in values.items() if k in cols})
        preview.archive_id = self.archive_id
        return preview

    def _logpage(self, filename: str) -> Logpage:
        rows = self._rows('logpage', filename)
        values = self._record_values('logpage', rows)
        values['filename'] = filename
        values.setdefault('file_format', format_from_name(filename))
        files = self.cfg.files
        path = os.path.join(files.logpage_dir, filename)

        if not os.path.exists(path):
            path = os.path.join(files.cover_dir, filename)

        values.update(image_values(path, self.cfg.image.logpage_exif_timezone))
        cols = Logpage.columns()
        logpage = Logpage(**{k: v for k, v in values.items() if k in cols})
        logpage.archive_id = self.archive_id
        logpage.logbook_num = values.get('logbook_num')
        return logpage

    # --- Plates ----------------------------------------------------------------

    def get_plate(self, plate_num: str | None = None,
                  wfpdb_id: str | None = None,
                  filename_scan: str | None = None) -> Plate:
        """Metadata of a plate, given by plate number, WFPDB ID or scan.

        Parameters
        ----------
        plate_num
            Identifier of the plate in the plate CSV files (or a WFPDB ID).
        wfpdb_id
            WFPDB ID of the plate.
        filename_scan
            File name of a scan; the plate is found from the scan CSV file
            or the file name, and the FITS file in ``[Files] fits_dir`` is
            read for the scan and the plate.

        Returns
        -------
        Plate
            The plate with its exposures, scans, previews and logpages. Its
            times are not computed yet (:meth:`Plate.compute_values`).
        """
        v = PlateValues()
        scan_values = {}
        ident = plate_num
        fn_base = None

        if filename_scan:
            path = os.path.join(self.cfg.files.fits_dir, filename_scan)
            _, plate_fits, scan_values = read_fits(path)
            v.update(plate_fits)
            fn_base = os.path.basename(filename_scan)
            link = {}

            for table, row in self._rows('scan', fn_base):
                apply_row(link, row, table, TYPES,
                          {k: p for k, p in table.columns.items()
                           if k in ('plate_num', 'wfpdb_id')})

            if not ident and link.get('plate_num'):
                ident = link['plate_num']

            if not wfpdb_id and link.get('wfpdb_id'):
                wfpdb_id = link['wfpdb_id']

            # The plate CSV may give the WFPDB ID
            if ident and not wfpdb_id:
                for table, row in self._rows('plate', ident)[:1]:
                    found = {}
                    apply_row(found, row, table, TYPES,
                              {k: p for k, p in table.columns.items()
                               if k == 'wfpdb_id'})
                    wfpdb_id = found.get('wfpdb_id') or None

        maindata = self.wfpdb['maindata']

        if ident and not wfpdb_id and ident in maindata:
            wfpdb_id = ident

        # A scan without CSV row: the file name may be a WFPDB ID or a plate
        # identifier
        if not ident and not wfpdb_id and fn_base:
            part = os.path.splitext(fn_base)[0]

            if part in maindata:
                ident = wfpdb_id = part

            if any(part in rows for _, rows in self.csv['plate']):
                ident = part

        if wfpdb_id and not ident:
            ident = wfpdb_id

        if wfpdb_id:
            v['wfpdb_id'] = wfpdb_id

        wfpdb_parsed = False

        for kind, parse in [('maindata', wfpdb.parse_maindata),
                            ('quality', wfpdb.parse_quality),
                            ('observer', wfpdb.parse_observer),
                            ('notes', wfpdb.parse_notes)]:
            entries = self.wfpdb[kind]
            key = wfpdb_id if wfpdb_id in entries else ident

            if key in entries:
                if kind in ('maindata', 'notes'):
                    parse(entries[key], v, self.cfg.wfpdb)
                else:
                    parse(entries[key], v)

                wfpdb_parsed = wfpdb_parsed or kind == 'maindata'

        for table, rows in self.csv['plate']:
            if ident in rows:
                key = ident
            elif wfpdb_id in rows:
                key = wfpdb_id
            else:
                continue

            keep = wfpdb_parsed and key == wfpdb_id
            apply_plate_row(v, rows[key], table, TYPES, keep)

        for table, row in self._rows('scan', fn_base):
            apply_row(scan_values, row, table, TYPES,
                      {k: p for k, p in table.columns.items()
                       if k not in ('plate_num', 'wfpdb_id')})

        if fn_base and not scan_values.get('filename_scan'):
            scan_values['filename_scan'] = fn_base

        self._keyword_values(v, Plate.columns() + Exposure.columns())
        plate = self._build_plate(v)
        keys = {k for k in (ident, wfpdb_id, plate.plate_num) if k}

        # Scans: from the plate CSV links, or the scan CSV rows of the plate
        if v['fn_scan']:
            filenames = list(v['fn_scan'])
        else:
            filenames = [row[t.columns['filename_scan'] - 1]
                         for t, row in self._linked_rows('scan', keys)]

        if fn_base and fn_base not in filenames:
            filenames.append(fn_base)

        plate.scans = [self._scan(fn, scan_values if fn == fn_base else None)
                       for fn in filenames]

        if v.get('fn_pre'):
            plate.previews = [self._preview(v['fn_pre'])]
        else:
            plate.previews = [self._preview(row[t.columns['filename'] - 1])
                              for t, row in self._linked_rows('preview',
                                                              keys)]

        cover = v.get('fn_cover')

        if cover or v['fn_log']:
            plate.logpages = ([self._logpage(cover) if cover else None] +
                              [self._logpage(fn) for fn in v['fn_log']])

        for scan in plate.scans:
            scan.plate_id = plate.plate_id

        return plate

    def _keyword_values(self, target: dict, columns: list[str]) -> None:
        """Apply ``["Keyword values"]`` and the lookup tables."""
        meta = self.cfg.metadata

        for key, val in meta.keyword_values.items():
            if key not in columns:
                continue

            # As in the old code, a value for a list is added to it
            if isinstance(target.get(key), list):
                target[key].append(val)
            else:
                target[key] = val

        for key, table in meta.lookup.items():
            val = target.get(key)

            if key in columns and isinstance(val, str) and val:
                target[key] = table.get(val.lower(), val)

    def _build_plate(self, v: PlateValues) -> Plate:
        """Plate and exposures from the collected values (section 4.3)."""
        numexp = v.get('numexp') or 1
        lists = {k: list(v[k]) if isinstance(v[k], list) else v[k]
                 for k in EXPOSURE_LIST_COLUMNS}

        if v['date_orig_start']:
            ntimes = len(lists['time_orig_start']) or 1

            if ntimes > numexp or 1 < ntimes < numexp:
                numexp = ntimes

            if numexp > 1:
                for key in FIRST_EXPOSURE_COLUMNS:
                    if isinstance(lists[key], list) and len(lists[key]) == 1:
                        lists[key] += [None] * (numexp - 1)

        plate_cols = {k: v[k] for k in Plate.columns()
                      if k in v and k not in _IDS and k not in LIST_KEYS}
        dates = v['date_orig_start']
        plate_cols['date_orig'] = dates[0] if dates else None
        plate = Plate(**plate_cols)
        plate.archive_id = self.archive_id
        plate.numexp = numexp

        for i in range(numexp):
            values = {}

            for col in Exposure.columns():
                if col in _IDS:
                    continue

                if col in v.exposure:
                    src = v.exposure[col]
                elif col in lists:
                    src = lists[col]
                elif col in v and col not in Plate.columns():
                    src = v[col]
                else:
                    continue

                values[col] = (_spread(src, i, col) if isinstance(src, list)
                               else src)

            exp = Exposure(**values)
            exp.archive_id = self.archive_id
            exp.exposure_num = i + 1

            if 'method_code' not in v.exposure:
                exp.method_code = plate.method_code

            exp.subexposures = self._subexposures(exp, plate.tz_orig)
            plate.exposures.append(exp)

        return plate

    @staticmethod
    def _subexposures(exp: Exposure, tz: str | None) -> list[SubExposure]:
        """Sub-exposures from ``|``-separated start times."""
        def recorded(combined):
            if not combined:
                return None

            t, ut, st = split_orig_times(combined)
            return t or ut or st

        start = recorded(exp.time_orig_start)

        if not start or '|' not in start:
            return []

        starts = [x.strip() for x in start.split('|')]
        end = recorded(exp.time_orig_end)
        ends = ([x.strip() for x in end.split('|')]
                if end and '|' in end else None)
        subs = []

        for i, t in enumerate(starts):
            subs.append(SubExposure(
                exposure_num=exp.exposure_num, subexposure_num=i + 1,
                date_orig_start=exp.date_orig_start, time_orig_start=t,
                time_orig_end=(ends[i] if i < len(ends) else None)
                if ends is not None else end))

        return subs

    # --- Iteration -------------------------------------------------------------

    def iter_plates(self) -> Iterator[Plate]:
        """Plates of the WFPDB maindata file, or else of the first plate
        CSV file."""
        if self.wfpdb['maindata']:
            for wfpdb_id in self.wfpdb['maindata']:
                yield self.get_plate(wfpdb_id=wfpdb_id)
        elif self.csv['plate']:
            for ident in self.csv['plate'][0][1]:
                yield self.get_plate(plate_num=ident)

    def iter_scans(self) -> Iterator[Scan]:
        """Scans of the scan CSV files, with the values of their FITS
        files."""
        for _, rows in self.csv['scan']:
            for filename in rows:
                plate = self.get_plate(filename_scan=filename)
                yield next(s for s in plate.scans
                           if s.filename_scan == filename)

    def iter_previews(self) -> Iterator[Preview]:
        """Previews of the preview CSV files."""
        for _, rows in self.csv['preview']:
            for filename in rows:
                yield self._preview(filename)

    def iter_logpages(self) -> Iterator[Logpage]:
        """Logpages of the logpage CSV files."""
        for _, rows in self.csv['logpage']:
            for filename in rows:
                yield self._logpage(filename)
