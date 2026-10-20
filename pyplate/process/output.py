"""Writing the results of a plate process to the database and to files.

The database writers take an open :class:`~pyplate.database.PlateDB`.
Results are written only with the process, scan, plate and archive IDs of
the process record.
"""
import os

import numpy as np

from ..database.database import PlateDB
from .photometry import PhotCalibCurve
from .process import PlateProcess
from .solve import ScannerPattern


def _result_ids(proc: PlateProcess) -> dict[str, int] | None:
    """Process, scan, plate and archive IDs, or None if one is missing."""
    record = proc.ctx.record
    ids = {'process_id': record.process_id, 'scan_id': record.scan_id,
           'plate_id': record.plate_id, 'archive_id': record.archive_id}

    if any(v is None for v in ids.values()):
        return None

    return ids


def _make_dir(proc: PlateProcess, path: str, event: int) -> None:
    """Create an output directory, if missing."""
    if path and not os.path.isdir(path):
        proc.ctx.log.write('Creating output directory {}'.format(path),
                           level=4, event=event)
        os.makedirs(path)


def write_solution(db: PlateDB, proc: PlateProcess) -> None:
    """Write the solution set, the solutions, their HEALPix tables and the
    scanner pattern to the database; the HEALPix tables to CSV files with
    ``[Database] write_solution_healpix_csv``."""
    log = proc.ctx.log
    cfg = proc.ctx.cfg
    write_csv = cfg.database.write_solution_healpix_csv
    solution_set = proc.solution

    log.to_db(3, 'Writing astrometric solution to the database', event=77)

    if solution_set.num_solutions == 0:
        log.write('No plate solution to write to the database',
                  level=2, event=77)
        return

    if write_csv:
        _make_dir(proc, cfg.files.write_db_solution_healpix_dir, 77)

    kw = _result_ids(proc)

    if kw is None:
        return

    kw['solutionset_id'] = db.write_platesolution(solution_set, **kw)

    for solution in solution_set.solutions:
        sol_id = db.write_solution(solution.to_row(), **kw)

        if solution.healpix_table is not None:
            db.write_solution_healpix(solution.healpix_table,
                                      solution_id=sol_id,
                                      solution_num=solution.solution_num,
                                      write_csv=write_csv, **kw)

    for solution in solution_set.duplicate_solutions:
        db.write_solution(solution.to_row(), **kw)

    if solution_set.pattern_table is not None:
        for row in solution_set.pattern_table:
            pattern = ScannerPattern(**{c: row[c] for c in row.colnames})
            db.write_scanner_pattern(pattern.to_row(), **kw)


def write_calibration(db: PlateDB, proc: PlateProcess) -> None:
    """Write the photometric calibrations to the database, and the
    calibration curves of the last iteration, sampled every 0.1 mag."""
    log = proc.ctx.log
    log.to_db(3, 'Writing photometric calibration to the database',
              event=78)

    if not proc.phot_calib:
        log.write('No photometric calibration to write to the database',
                  level=2, event=78)
        return

    kwargs = _result_ids(proc)

    if kwargs is None:
        log.write('Cannot output calibration data. Missing process_id,'
                  'scan_id, plate_id or archive_id.',
                  level=2, event=78)
        return

    for calib in proc.phot_calib:
        calib_id = db.write_phot_calib(calib.to_row(), **kwargs)

        # For the last iteration, also write calib curves to the database
        if calib.iteration == proc.ctx.record.num_iterations:
            if calib.curve is None:
                log.write('Cannot write photometric calibration '
                          'curves to the database', level=2, event=78)
                continue

            min_mag = np.floor(calib.plate_mag_brightest * 10) / 10.
            max_mag = np.ceil(calib.plate_mag_lim * 10) / 10.
            plate_mags = np.arange(min_mag, max_mag + 0.01, 0.1)

            for m in plate_mags:
                point = PhotCalibCurve(calib_id=calib_id,
                                       solution_num=calib.solution_num,
                                       iteration=calib.iteration,
                                       plate_mag=m,
                                       natmag=calib.curve(m).item())
                db.write_calib_curve(point.to_row(), **kwargs)


def write_sources(db: PlateDB, proc: PlateProcess) -> None:
    """Write the sources and their Gaia crossmatches to the database, or to
    CSV files with ``[Database] write_sources_csv``."""
    log = proc.ctx.log
    cfg = proc.ctx.cfg
    write_csv = cfg.database.write_sources_csv

    if write_csv:
        log.to_db(3, 'Writing sources to database files', event=80)
        _make_dir(proc, cfg.files.write_db_source_dir, 80)
        _make_dir(proc, cfg.files.write_db_source_calib_dir, 80)
        _make_dir(proc, cfg.files.write_db_source_xmatch_dir, 80)
    else:
        log.to_db(3, 'Writing sources to the database', event=80)

    kwargs = _result_ids(proc)

    if kwargs is None:
        log.write('Cannot write source data due to missing '
                  'plate identification number(s).',
                  level=2, event=80)
        return

    db.write_sources(proc.sources, write_csv=write_csv, **kwargs)

    if proc.sources.neighbors_gaia is not None:
        db.write_source_xmatches(proc.sources.neighbors_gaia,
                                 write_csv=write_csv, **kwargs)


def write_sources_csv(proc: PlateProcess) -> None:
    """Write the sources to ``<basefn>_sources.csv`` in
    ``[Files] write_source_dir``."""
    log = proc.ctx.log
    write_source_dir = proc.ctx.cfg.files.write_source_dir
    log.to_db(3, 'Writing sources to a file', event=81)
    _make_dir(proc, write_source_dir, 81)
    fn_world = os.path.join(write_source_dir,
                            '{}_sources.csv'.format(proc.ctx.basefn))

    # Output CSV file with extracted sources
    log.write('Writing output file {}'.format(fn_world), level=4, event=81)
    proc.sources.output_csv(fn_world)


def write_wcs_header(proc: PlateProcess) -> None:
    """Write the WCS header to ``<basefn>.wcs`` in ``[Files]
    write_wcs_dir``."""
    if not proc.plate_solved:
        return

    log = proc.ctx.log
    write_wcs_dir = proc.ctx.cfg.files.write_wcs_dir
    log.write('Writing WCS header to a file', level=3, event=36)

    # Create output directory, if missing
    if write_wcs_dir and not os.path.isdir(write_wcs_dir):
        log.write('Creating WCS output directory {}'
                  ''.format(write_wcs_dir), level=4, event=36)
        os.makedirs(write_wcs_dir)

    fn_header = os.path.join(write_wcs_dir,
                             '{}.wcs'.format(proc.ctx.basefn))
    log.write('Writing WCS output file {}'.format(fn_header),
              level=4, event=36)
    proc.solution.header_wcs.tofile(fn_header, overwrite=True)
