"""PlateDB: writes records and results to the database, for every backend."""
import csv
import os
import socket
from collections.abc import Iterable, Sequence
from typing import Any

import astropy.units as u
import numpy as np
from astropy.io import fits

from .._version import __version__
from ..config import DatabaseConfig, FilesConfig
from ..metadata import (Archive, Logbook, Logpage, Plate, Preview, Scan)
from .backend import Backend
from .db_mysql import MysqlBackend
from .db_pgsql import PgsqlBackend
from .db_sqlite import SqliteBackend
from .schema import Schema, load_schema

BACKENDS = {'pgsql': PgsqlBackend, 'mysql': MysqlBackend,
            'sqlite': SqliteBackend}

# Rows sent with one executemany()
BATCH_SIZE = 1000

# Columns set by the result writers before the table columns
RESULT_IDS = ('process_id', 'scan_id', 'plate_id', 'archive_id')
SOLUTION_IDS = ('solutionset_id',) + RESULT_IDS
SOURCE_IDS = ('source_id',) + RESULT_IDS
HEALPIX_IDS = ('solution_id',) + SOLUTION_IDS + ('solution_num',)


class CsvWriter:
    """CSV writer that writes None as ``\\N`` (NULL for database loading)."""

    def __init__(self, f, *args, **kwargs):
        self.writer = csv.writer(f, *args, **kwargs)

    def writerow(self, row: Sequence) -> None:
        self.writer.writerow(['\\N' if val is None else val for val in row])


def _finite(value: Any) -> Any:
    """None instead of NaN and infinity."""
    try:
        if not np.isfinite(value):
            return None
    except TypeError:
        pass

    return value


def _result_value(value: Any) -> Any:
    """Value of a solution or solution-set attribute for the database."""
    if isinstance(value, u.Quantity):
        value = value.value
    elif isinstance(value, fits.Header):
        value = value.tostring(sep='\\n')
    elif isinstance(value, bool):
        value = int(value)

    return _finite(value)


class PlateDB:
    """Database access, the same for every backend.

    Parameters
    ----------
    cfg
        ``[Database]`` with its ``connection`` table.
    files
        ``[Files]``, for the directories of the CSV output of sources.
    schema
        Schema to use instead of the one named in the configuration.
    backend
        Backend to use instead of the one named in the configuration
        (tests use a backend with a recording cursor).
    hostname, version
        Written by :meth:`write_process_start` instead of the host name and
        the pyplate version.
    """

    def __init__(self, cfg: DatabaseConfig, files: FilesConfig | None = None,
                 schema: Schema | None = None,
                 backend: Backend | None = None,
                 hostname: str | None = None, version: str | None = None):
        conn = cfg.connection
        self.cfg = cfg
        self.files = files or FilesConfig()

        if schema is None:
            name = conn.schema if conn is not None else None
            yaml = conn.yaml if conn is not None else None
            schema = load_schema(yaml or '{}.yaml'.format(name or
                                                          'applause_dr5'),
                                 name=name)

        self.schema = schema

        if backend is None:
            if conn is None:
                raise ValueError('No [Database.connection]')

            if conn.rdbms not in BACKENDS:
                raise ValueError('Unknown rdbms {!r}; one of {}'
                                 .format(conn.rdbms, sorted(BACKENDS)))

            backend = BACKENDS[conn.rdbms](conn, schema_name=schema.name)

        self.backend = backend
        self.hostname = hostname
        self.version = version

    # --- Connection and schema -------------------------------------------

    def open_connection(self) -> None:
        self.backend.connect()

    def close_connection(self) -> None:
        self.backend.close()

    def schema_sql(self, mode: str) -> list[str]:
        """Statements to create or drop the tables or indexes; ``mode`` is
        one of :data:`pyplate.database.ddl.MODES`."""
        return self.backend.ddl(self.schema, mode)

    def run_schema_sql(self, mode: str, execute: bool = False) -> None:
        """Execute the statements of :meth:`schema_sql`, or print them."""
        for sql in self.schema_sql(mode):
            if execute:
                self.backend.execute(sql)
            else:
                print(sql + ';\n')

    def create_schema(self, execute: bool = False) -> None:
        self.run_schema_sql('create_schema', execute)

    def drop_schema(self, execute: bool = False) -> None:
        self.run_schema_sql('drop_schema', execute)

    # --- Generic statements ----------------------------------------------

    def _columns(self, table: str, row: dict, first: Sequence[str]) -> list:
        """Columns to write: ``first``, then the table columns in schema
        order; only those in the row, and without an unset serial ID."""
        t = self.schema[table]
        cols = [c for c in first if c in row and c in t.column_names]
        cols += [c for c in t.column_names if c in row and c not in cols]
        serial = t.serial

        if serial is not None and row.get(serial.name) is None:
            cols = [c for c in cols if c != serial.name]

        return cols

    def _values(self, table: str, row: dict, cols: list[str]) -> list:
        t = self.schema[table]
        return [self.backend.prepare_value(t.column(c), row[c]) for c in cols]

    def _insert_sql(self, table: str, cols: list[str]) -> str:
        return 'INSERT INTO {} ({}) VALUES ({})'.format(
            self.backend.qualify(table), ','.join(cols),
            ','.join([self.backend.placeholder] * len(cols)))

    def insert(self, table: str, row: dict[str, Any],
               first: Sequence[str] = (), return_id: bool = False
               ) -> int | None:
        """Insert a row; return its ID with ``return_id``.

        Parameters
        ----------
        table
            Table name.
        row
            Values by column; keys that are not columns of the table are
            ignored.
        first
            Columns written first, before the others in schema order.
        return_id
            Return the ID of the new row (its serial column).
        """
        cols = self._columns(table, row, first)
        sql = self._insert_sql(table, cols)
        values = self._values(table, row, cols)

        if return_id:
            return self.backend.insert_returning_id(
                sql, values, self.schema[table].serial.name)

        self.backend.execute(sql, values)
        return None

    def insert_many(self, table: str, rows: Iterable[dict[str, Any]],
                    first: Sequence[str] = ()) -> None:
        """Insert rows with ``executemany``, in batches of 1000.

        The columns are those of the first row.
        """
        batch = []
        sql = None
        cols = None

        for row in rows:
            if cols is None:
                cols = self._columns(table, row, first)
                sql = self._insert_sql(table, cols)

            batch.append(self._values(table, row, cols))

            if len(batch) == BATCH_SIZE:
                self.backend.executemany(sql, batch)
                batch = []

        if batch:
            self.backend.executemany(sql, batch)

    def update(self, table: str, row: dict[str, Any],
               key: dict[str, Any]) -> None:
        """Update the row with the given key; ``row`` gives the columns to
        set, in order."""
        t = self.schema[table]
        cols = [c for c in row if c in t.column_names]

        if not cols:
            return

        p = self.backend.placeholder
        sql = 'UPDATE {} SET {} WHERE {}'.format(
            self.backend.qualify(table),
            ','.join('{}={}'.format(c, p) for c in cols),
            ' AND '.join('{}={}'.format(k, p) for k in key))
        self.backend.execute(sql, self._values(table, row, cols) +
                             list(key.values()))

    def upsert(self, table: str, row: dict[str, Any],
               key_cols: Sequence[str]) -> None:
        """Insert a row, or update the row with the same key columns."""
        cols = self._columns(table, row, key_cols)
        sql = self.backend.upsert_sql(table, cols, list(key_cols))
        self.backend.execute(sql, self._values(table, row, cols))

    def _select_one(self, sql: str, params: Sequence) -> tuple | None:
        return self.backend.query(sql, params)

    # --- Metadata records --------------------------------------------------

    @staticmethod
    def _require_archive_id(record) -> None:
        if record.archive_id is None:
            raise ValueError('{} has no archive_id; it is needed for writing '
                             'to the database ([Archive] archive_id)'
                             .format(type(record).__name__))

    def write_archive(self, archive: Archive) -> None:
        """Insert the archive, or update it if it exists."""
        self._require_archive_id(archive)
        self.upsert('archive', archive.to_row(), ['archive_id'])

    def write_plate(self, plate: Plate) -> int:
        """Write a plate with its exposures, sub-exposures, scans, previews
        and plate–logpage links; set the IDs in the records.

        Returns
        -------
        int
            The plate ID.
        """
        self._require_archive_id(plate)
        plate_id = self.insert('plate', plate.to_row(), return_id=True)
        plate.plate_id = plate_id

        for exp in plate.exposures:
            exp.plate_id = plate_id
            exp.exposure_id = self.insert('exposure', exp.to_row(),
                                          first=('exposure_num',),
                                          return_id=True)

            for sub in exp.subexposures:
                sub.exposure_id = exp.exposure_id
                sub.plate_id = plate_id
                sub.exposure_num = exp.exposure_num
                self.insert('exposure_sub', sub.to_row())

        for scan in plate.scans:
            scan.plate_id = plate_id
            self.write_scan(scan)

        for preview in plate.previews:
            preview.plate_id = plate_id
            self.write_preview(preview)

        self.write_plate_logpages(plate)
        return plate_id

    def _plate_id(self, plate_num: str | None, archive_id: int | None,
                  wfpdb_id: str | None) -> int | None:
        plate_id = self.get_plate_id(plate_num, archive_id)

        if plate_id is None:
            plate_id = self.get_plate_id_wfpdb(wfpdb_id)

        return plate_id

    def write_plate_logpages(self, plate: Plate) -> None:
        """Write the plate–logpage links of ``plate.logpages``.

        The list position is ``logpage_order``. The logpages are looked up
        by file name; without ``plate.plate_id`` the plate is looked up by
        ``plate_num`` or ``wfpdb_id`` (once for each logpage, as in the old
        code).
        """
        for order, logpage in enumerate(plate.logpages):
            if logpage is None or not logpage.filename:
                continue

            plate_id = plate.plate_id or self._plate_id(
                plate.plate_num, plate.archive_id, plate.wfpdb_id)
            logpage_id = self.get_logpage_id(logpage.filename,
                                             plate.archive_id)

            if plate_id and logpage_id:
                self.insert('plate_logpage', {'plate_id': plate_id,
                                              'logpage_id': logpage_id,
                                              'logpage_order': order})

    def write_scan(self, scan: Scan, plate_num: str | None = None,
                   wfpdb_id: str | None = None) -> int:
        """Write a scan; without ``scan.plate_id``, the plate is looked up
        by ``plate_num`` or ``wfpdb_id``. Returns the scan ID."""
        self._require_archive_id(scan)

        if not scan.plate_id:
            scan.plate_id = self._plate_id(plate_num, scan.archive_id,
                                           wfpdb_id)

        scan.scan_id = self.insert('scan', scan.to_row(),
                                   first=('plate_id', 'scan_id'),
                                   return_id=True)
        return scan.scan_id

    def update_scan(self, scan: Scan, filecols: bool = False) -> None:
        """Update a scan; with ``filecols`` only the file columns
        (``file_datetime``, ``file_size``, ``fits_checksum``,
        ``fits_datasum``). Without ``scan.scan_id`` the scan is looked up by
        file name."""
        scan_id = scan.scan_id

        if not scan_id:
            scan_id, _ = self.get_scan_id(scan.filename_scan, scan.archive_id)

            if scan_id is None:
                print('Cannot update scan metadata in the database '
                      '(filename={}, archive_id={})'
                      .format(scan.filename_scan, scan.archive_id))
                return

        if filecols:
            cols = ['file_datetime', 'file_size', 'fits_checksum',
                    'fits_datasum']
        else:
            cols = [c for c in self.schema['scan'].column_names
                    if c != 'scan_id']

        row = scan.to_row()
        self.update('scan', {c: row[c] for c in cols if c in row},
                    {'scan_id': scan_id})

    def write_preview(self, preview: Preview, plate_num: str | None = None,
                      wfpdb_id: str | None = None) -> int:
        """Write a preview, as :meth:`write_scan`. Returns its ID."""
        self._require_archive_id(preview)

        if not preview.plate_id:
            preview.plate_id = self._plate_id(plate_num, preview.archive_id,
                                              wfpdb_id)

        preview.preview_id = self.insert('preview', preview.to_row(),
                                         first=('plate_id', 'preview_id'),
                                         return_id=True)
        return preview.preview_id

    def write_logbook(self, logbook: Logbook) -> int:
        """Write a logbook. Returns its ID."""
        self._require_archive_id(logbook)
        logbook.logbook_id = self.insert('logbook', logbook.to_row(),
                                         return_id=True)
        return logbook.logbook_id

    def write_logpage(self, logpage: Logpage,
                      logbook_num: str | None = None) -> int:
        """Write a logpage; without ``logpage.logbook_id``, the logbook is
        looked up by ``logbook_num`` (default ``logpage.logbook_num``).
        Returns the logpage ID."""
        self._require_archive_id(logpage)

        if logpage.logbook_id is None:
            logpage.logbook_id = self.get_logbook_id(
                logbook_num or logpage.logbook_num, logpage.archive_id)

        logpage.logpage_id = self.insert('logpage', logpage.to_row(),
                                         return_id=True)
        return logpage.logpage_id

    # --- Results -------------------------------------------------------------

    def write_platesolution(self, platesolution: Any, process_id=None,
                            scan_id=None, plate_id=None,
                            archive_id=None) -> int:
        """Write a solution set (an object with the columns as
        attributes). Returns its ID."""
        row = dict(zip(RESULT_IDS, (process_id, scan_id, plate_id,
                                    archive_id)))

        for col in self.schema['solution_set'].column_names:
            if col not in row and hasattr(platesolution, col):
                row[col] = _result_value(getattr(platesolution, col))

        return self.insert('solution_set', row, first=RESULT_IDS,
                           return_id=True)

    def write_solution(self, solution: dict, solutionset_id=None,
                       process_id=None, scan_id=None, plate_id=None,
                       archive_id=None) -> int:
        """Write an astrometric solution. Returns its ID."""
        row = dict(zip(SOLUTION_IDS, (solutionset_id, process_id, scan_id,
                                      plate_id, archive_id)))

        for col in self.schema['solution'].column_names:
            if col not in row and col in solution:
                row[col] = _result_value(solution[col])

        return self.insert('solution', row, first=SOLUTION_IDS,
                           return_id=True)

    def _write_rows(self, table: str, rows: Iterable[dict], first: tuple,
                    csv_path: str | None) -> None:
        """Insert rows, or write them to a CSV file."""
        if csv_path is None:
            self.insert_many(table, rows, first)
            return

        with open(csv_path, 'w', newline='') as f:
            writer = None

            for row in rows:
                cols = self._columns(table, row, first)

                if writer is None:
                    writer = CsvWriter(f, delimiter=',', quotechar='"',
                                       quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(cols)

                writer.writerow([row[c] for c in cols])

    def write_solution_healpix(self, solution_healpix, solution_id=None,
                               solutionset_id=None, process_id=None,
                               scan_id=None, plate_id=None, archive_id=None,
                               solution_num=None, write_csv=None) -> None:
        """Write the HEALPix table of a solution (rows with the columns of
        ``solution_healpix``), or write it to a CSV file."""
        ids = dict(zip(HEALPIX_IDS, (solution_id, solutionset_id, process_id,
                                     scan_id, plate_id, archive_id,
                                     solution_num)))
        cols = [c for c in self.schema['solution_healpix'].column_names
                if c not in ids]
        rows = ({**ids, **{c: _finite(r[c]) for c in cols if c in _keys(r)}}
                for r in solution_healpix)
        csv_path = None

        if write_csv:
            csv_path = os.path.join(
                self.files.write_db_solution_healpix_dir,
                '{:06d}_solution_healpix.csv'.format(solution_id))

        self._write_rows('solution_healpix', rows, HEALPIX_IDS, csv_path)

    def write_scanner_pattern(self, row, solutionset_id=None,
                              process_id=None, scan_id=None, plate_id=None,
                              archive_id=None) -> None:
        """Write one row of the scanner pattern."""
        out = dict(zip(SOLUTION_IDS, (solutionset_id, process_id, scan_id,
                                      plate_id, archive_id)))

        for col in self.schema['scanner_pattern'].column_names:
            if col not in out and col in _keys(row):
                out[col] = row[col]

        self.insert('scanner_pattern', out, first=SOLUTION_IDS)

    def write_phot_calib(self, phot_calib: dict, process_id=None,
                         scan_id=None, plate_id=None,
                         archive_id=None) -> int:
        """Write a photometric calibration. Returns its ID."""
        row = {**dict(zip(RESULT_IDS, (process_id, scan_id, plate_id,
                                       archive_id))),
               **{k: v for k, v in phot_calib.items() if k not in RESULT_IDS}}
        return self.insert('phot_calib', row, first=RESULT_IDS,
                           return_id=True)

    def write_calib_curve(self, calib_curve: dict, process_id=None,
                          scan_id=None, plate_id=None,
                          archive_id=None) -> None:
        """Write a point of a photometric calibration curve."""
        row = {**dict(zip(RESULT_IDS, (process_id, scan_id, plate_id,
                                       archive_id))),
               **{k: v for k, v in calib_curve.items()
                  if k not in RESULT_IDS}}
        self.insert('phot_calib_curve', row, first=RESULT_IDS)

    def source_id(self, process_id: int, source_num: int) -> int:
        """Database ID of a source:
        dr_num, process_id and source_num as digits."""
        cfg = self.cfg
        return (cfg.dr_num * 10**(cfg.process_num_digits +
                                  cfg.source_num_digits) +
                process_id * 10**cfg.source_num_digits + source_num)

    def write_sources(self, sources, process_id=None, scan_id=None,
                      plate_id=None, archive_id=None, write_csv=None) -> None:
        """Write sources to the ``source`` and ``source_calib`` tables (or
        CSV files). NaN and infinity become NULL; in ``source_calib`` also
        negative HEALPix numbers and Gaia IDs 0. As in the old code, the
        batches of the two tables alternate."""
        ids = dict(zip(RESULT_IDS, (process_id, scan_id, plate_id,
                                    archive_id)))
        names = _keys(sources)
        tables = ['source', 'source_calib']
        cols = {t: [c for c in self.schema[t].column_names
                    if c in names and c not in ids and c != 'source_id']
                for t in tables}

        def row(s, table):
            out = {'source_id': self.source_id(process_id, s['source_num']),
                   **ids}

            for c in cols[table]:
                val = _finite(s[c])

                if table == 'source_calib':
                    if 'healpix256' in c and val is not None and val < 0:
                        val = None

                    if c.startswith('gaia') and c.endswith('_id') and \
                            val == 0:
                        val = None

                out[c] = val

            return out

        if write_csv:
            for table, attr in [('source', 'write_db_source_dir'),
                                ('source_calib',
                                 'write_db_source_calib_dir')]:
                path = os.path.join(getattr(self.files, attr),
                                    '{:06d}_{}.csv'.format(process_id, table))
                self._write_rows(table, (row(s, table) for s in sources),
                                 SOURCE_IDS, path)

            return

        for start in range(0, len(sources), BATCH_SIZE):
            batch = sources[start:start + BATCH_SIZE]

            for table in tables:
                self.insert_many(table, [row(s, table) for s in batch],
                                 SOURCE_IDS)

    def write_source_xmatches(self, xmatches, process_id=None, scan_id=None,
                              plate_id=None, archive_id=None,
                              write_csv=None) -> None:
        """Write crossmatches to ``source_xmatch`` (or a CSV file)."""
        ids = dict(zip(RESULT_IDS, (process_id, scan_id, plate_id,
                                    archive_id)))
        names = _keys(xmatches)
        cols = [c for c in self.schema['source_xmatch'].column_names
                if c in names and c not in ids and c != 'source_id']
        rows = ({'source_id': self.source_id(process_id, x['source_num']),
                 **ids, **{c: _finite(x[c]) for c in cols}}
                for x in xmatches)
        csv_path = None

        if write_csv:
            csv_path = os.path.join(self.files.write_db_source_xmatch_dir,
                                    '{:06d}_source_xmatch.csv'
                                    .format(process_id))

        self._write_rows('source_xmatch', rows, SOURCE_IDS, csv_path)

    def write_process_start(self, scan_id=None, plate_id=None,
                            archive_id=None, filename=None,
                            use_psf=None) -> int:
        """Write the start of a plate process. Returns the process ID."""
        row = {'scan_id': scan_id, 'plate_id': plate_id,
               'archive_id': archive_id, 'filename': filename,
               'hostname': self.hostname or socket.gethostname(),
               'use_psf': 1 if use_psf else 0,
               'pyplate_version': self.version or __version__}
        return self.insert('process', row, first=list(row), return_id=True)

    def update_process(self, process_id: int, **kwargs) -> None:
        """Set columns of the process; unknown names and None values are
        skipped."""
        row = {k: v for k, v in kwargs.items() if v is not None}
        self.update('process', row, {'process_id': process_id})

    def write_process_end(self, process_id: int, completed=None,
                          duration=None) -> None:
        """Write the end time, duration and completion of a process."""
        p = self.backend.placeholder
        sql = ('UPDATE {} SET timestamp_end={},duration={},completed={} '
               'WHERE process_id={}'.format(self.backend.qualify('process'),
                                             self.backend.now_sql, p, p, p))
        self.backend.execute(sql, (duration, completed, process_id))

    def write_processlog(self, level: int, message: str, event=None,
                         solution_num=None, process_id=None, scan_id=None,
                         plate_id=None, archive_id=None) -> None:
        """Write a log message of a process."""
        row = {'process_id': process_id, 'scan_id': scan_id,
               'plate_id': plate_id, 'archive_id': archive_id,
               'level': level, 'event': event, 'solution_num': solution_num,
               'message': message}
        self.insert('process_log', row, first=list(row))

    # --- Lookups ---------------------------------------------------------------
    # A lookup with a key of None is not sent and finds nothing

    def _lookup(self, sql: str, keys: Sequence) -> tuple | None:
        if any(k is None for k in keys):
            return None

        return self._select_one(sql, keys)

    def get_plate_id(self, plate_num: str | None,
                     archive_id: int | None) -> int | None:
        """Plate ID by archive ID and plate number."""
        row = self._lookup('SELECT plate_id FROM {} WHERE archive_id={} AND '
                           'plate_num={}'.format(self.backend.qualify('plate'),
                                                 *[self.backend.placeholder]
                                                 * 2),
                           (archive_id, plate_num))
        return row[0] if row else None

    def get_plate_id_wfpdb(self, wfpdb_id: str | None) -> int | None:
        """Plate ID by WFPDB ID."""
        row = self._lookup('SELECT plate_id FROM {} WHERE wfpdb_id={}'
                           .format(self.backend.qualify('plate'),
                                   self.backend.placeholder), (wfpdb_id,))
        return row[0] if row else None

    def get_scan_id(self, filename: str | None,
                    archive_id: int | None) -> tuple[int | None, int | None]:
        """Scan ID and plate ID by scan file name and archive ID."""
        p = self.backend.placeholder
        row = self._lookup('SELECT scan_id, plate_id FROM {} '
                           'WHERE filename_scan={} AND archive_id={}'
                           .format(self.backend.qualify('scan'), p, p),
                           (filename, archive_id))
        return tuple(row) if row else (None, None)

    def get_logbook_id(self, logbook_num: str | None,
                       archive_id: int | None) -> int | None:
        """Logbook ID by archive ID and logbook number."""
        p = self.backend.placeholder
        row = self._lookup('SELECT logbook_id FROM {} '
                           'WHERE archive_id={} AND logbook_num={}'
                           .format(self.backend.qualify('logbook'), p, p),
                           (archive_id, logbook_num))
        return row[0] if row else None

    def get_logpage_id(self, filename: str | None,
                       archive_id: int | None) -> int | None:
        """Logpage ID by file name and archive ID."""
        p = self.backend.placeholder
        row = self._lookup('SELECT logpage_id FROM {} '
                           'WHERE filename={} AND archive_id={}'
                           .format(self.backend.qualify('logpage'), p, p),
                           (filename, archive_id))
        return row[0] if row else None

    def get_plate_epoch(self, plate_id: int | None) -> float | None:
        """Earliest start year of the exposures of a plate."""
        row = self._lookup('SELECT year_start FROM {} WHERE plate_id={} '
                           'AND year_start IS NOT NULL ORDER BY year_start'
                           .format(self.backend.qualify('exposure'),
                                   self.backend.placeholder), (plate_id,))
        return row[0] if row else None


def _keys(rows) -> list[str]:
    """Column names of an astropy Table or Row, or keys of a dict or of
    the first dict of a list."""
    if hasattr(rows, 'colnames'):
        return rows.colnames

    if isinstance(rows, dict):
        return list(rows)

    rows = list(rows)
    return list(rows[0]) if rows else []
