"""Run the redesigned pipeline on the reference plates and save its results.

The counterpart of tests/ver4/run_reference_plates.py (which runs the old
code) for the redesigned code: the same plates, wrappers and output layout,
so that tests/ver4/compare_reference_plates.py --dr5 compares the two runs.
See tests/ver4/README.md, section "Reference plates".

The configuration is converted from tests/ver4/plates/plates.conf and
tests/ver4/plates/paths.local with pyplate.config.ini2toml (without the
[Reference] section of the driver) and written to <plate>/config.toml in
the run directory, with the output directories inside the run directory.
The SQL goes to a recording database (no database server is needed).

Run from the repository root, for example:

    PYTHONPATH=. python tests/run_reference_plates.py new1 --replay-gaia run1
    python tests/ver4/compare_reference_plates.py run1 new1 --dr5
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tomllib
import traceback
from types import SimpleNamespace

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

import pyplate.database.database as database
import pyplate.metadata.header as header
from pyplate.config import config_from_dict
from pyplate.config.ini2toml import convert, read_ini
from pyplate.database.db_pgsql import PgsqlBackend
from pyplate.process.catalog import StarCatalog
from pyplate.process.pipeline import PlatePipeline
from pyplate.process.process import Process
from pyplate.process.sources import SourceTable

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VER4 = os.path.join(HERE, 'ver4')
PLATES_DIR = os.path.join(VER4, 'plates')
PLATES_CONF = os.path.join(PLATES_DIR, 'plates.conf')
PATHS_LOCAL = os.path.join(PLATES_DIR, 'paths.local')
sys.path.insert(0, VER4)

from sqlrender import (FakeConnection, RecordingCursor,  # noqa: E402
                       format_queries)

PLATES = ['LA02333_x.fits', 'LA02334_x.fits']
SCHEMA = 'applause_dr5'

# Fixed values written into headers and the process table
UTCNOW = datetime.datetime(2001, 2, 3, 4, 5, 6)
VERSION = 'reference-version'
HOSTNAME = 'reference-host'

# Mock IDs returned by the recording database for the scan lookup
SCAN_ID, PLATE_ID = 9001, 9002

# Source-table columns written by crossmatch_gaia()
MATCH_COLUMNS = ['source_num', 'solution_num', 'match_radius', 'gaiadr4_id',
                 'gaiadr4_gmag', 'gaiadr4_bpmag', 'gaiadr4_rpmag',
                 'gaiadr4_bp_rp', 'gaiadr4_dist', 'gaiadr4_neighbors']

# Source-table columns summarised in summary.json
STAT_COLUMNS = ['x_source', 'y_source', 'mag_auto', 'flux_auto', 'fwhm_image',
                'ra_icrs', 'dec_icrs', 'annular_bin', 'flag_clean',
                'model_prediction', 'gaiadr4_dist', 'natmag',
                'natmag_error', 'bpmag', 'rpmag']

# External programs run by the pipeline (versions go into the manifest)
PROGRAMS = ['sextractor_path', 'psfex_path', 'scamp_path', 'solve_field_path',
            'wcs_to_tan_path']

# Result tables whose SQL is too large for results_small.sql
LARGE_TABLES = ['solution_healpix', 'source', 'source_calib', 'source_xmatch']


class ReferenceCursor(RecordingCursor):
    """
    Recording cursor whose SELECTs answer like a database that knows the scan
    (get_scan_id) and nothing else, e.g. no plate epoch (get_plate_epoch).
    """

    def execute(self, sql, params=None):
        super().execute(sql, params)

        if sql.startswith('SELECT'):
            if sql.startswith('SELECT scan_id, plate_id'):
                self._last = (SCAN_ID, PLATE_ID)
            else:
                self._last = None


# State of the current plate, used by the wrappers
STATE = SimpleNamespace(plate_dir=None, cursor=None, xmatch_count=0,
                        gaia_index=[], replay=None, replay_dir=None,
                        errors=[])


class ReferenceBackend(PgsqlBackend):
    """PostgreSQL backend that writes to the recording cursor of the plate."""

    def connect(self):
        self.cursor = STATE.cursor
        self.connection = FakeConnection()

    def close(self):
        self.connection = None
        self.cursor = None


# --- Helpers ---------------------------------------------------------------

def encode(val):
    """Convert a value to JSON; mark values that JSON cannot hold."""
    if isinstance(val, fits.Header):
        return {'header': val.tostring(sep='\n')}
    if isinstance(val, u.Quantity):
        v = val.value
        return {'quantity': encode(v.tolist() if np.ndim(v) else float(v)),
                'unit': val.unit.to_string()}
    if isinstance(val, SkyCoord):
        return {'skycoord': {'ra': encode(val.ra.deg.tolist()),
                             'dec': encode(val.dec.deg.tolist())}}
    if isinstance(val, Table):
        return {'table_rows': len(val)}
    if isinstance(val, np.ndarray):
        return encode(val.tolist())
    if isinstance(val, np.generic):
        val = val.item()
    if isinstance(val, float):
        if math.isnan(val):
            return {'nan': True}
        if math.isinf(val):
            return {'inf': 1 if val > 0 else -1}
        return val
    if isinstance(val, (list, tuple)):
        return [encode(v) for v in val]
    if isinstance(val, dict):
        return {str(k): encode(v) for k, v in val.items()}
    if val is None or isinstance(val, (str, int, bool)):
        return val
    return {'repr': repr(val)}


def write_table(tab, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Table(tab).write(path, format='fits', overwrite=True)


def sha256(path):
    h = hashlib.sha256()

    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)

    return h.hexdigest()


def file_entry(path, base):
    return {'name': os.path.relpath(path, base),
            'size': os.path.getsize(path), 'sha256': sha256(path)}


def column_stats(sources):
    stats = {}

    for col in STAT_COLUMNS:
        if col not in sources.colnames:
            continue

        data = np.ma.filled(np.ma.asarray(sources[col], dtype=float), np.nan)
        good = np.isfinite(data)
        stats[col] = {'valid': int(good.sum())}

        if good.any():
            d = data[good]
            stats[col].update({'min': float(d.min()), 'max': float(d.max()),
                               'mean': float(d.mean()),
                               'median': float(np.median(d))})

    return stats


# --- Wrappers --------------------------------------------------------------

def install_wrappers():
    # Fixed version and time in headers (HISTORY, DATE)
    header.__version__ = VERSION
    header.utc_now = lambda: UTCNOW

    # Recording database instead of a database server, with fixed host
    # name and version in the process table
    database.BACKENDS['pgsql'] = ReferenceBackend
    orig_init = database.PlateDB.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault('hostname', HOSTNAME)
        kwargs.setdefault('version', VERSION)
        orig_init(self, *args, **kwargs)

    database.PlateDB.__init__ = init

    # Gaia TAP queries: save the results; replay them with --replay-gaia
    orig_tap = StarCatalog.query_gaia_tap

    def query_gaia_tap(self, query, skycoord, radius):
        key = {'query': query,
               'ra': '{:f}'.format(skycoord.ra.to(u.deg).value),
               'dec': '{:f}'.format(skycoord.dec.to(u.deg).value),
               'radius': '{:f}'.format(radius.to(u.deg).value)}
        n = len(STATE.gaia_index) + 1
        fn = 'gaia_{:02d}.vot'.format(n)

        if STATE.replay is not None:
            if n > len(STATE.replay) or STATE.replay[n - 1]['key'] != key:
                raise RuntimeError('Gaia query {:d} differs from the replayed '
                                   'run: {}'.format(n, key))

            if self.log is not None:
                self.log.write('Gaia TAP query (replayed): {}'
                               .format(key), level=4, event=41)

            tab = Table.read(os.path.join(STATE.replay_dir,
                                          STATE.replay[n - 1]['file']),
                             format='votable')
        else:
            tab = orig_tap(self, query, skycoord, radius)

        gaia_dir = os.path.join(STATE.plate_dir, 'gaia')
        os.makedirs(gaia_dir, exist_ok=True)
        tab.write(os.path.join(gaia_dir, fn), format='votable',
                  tabledata_format='binary2', overwrite=True)
        STATE.gaia_index.append({'key': key, 'file': fn, 'rows': len(tab)})
        return tab

    StarCatalog.query_gaia_tap = query_gaia_tap

    def query_gaia_sql(self, *args, **kwargs):
        raise RuntimeError('The reference run supports Gaia TAP queries only')

    StarCatalog.query_gaia_sql = query_gaia_sql

    # Crossmatch: inputs and outputs of every call
    orig_crossmatch = SourceTable.crossmatch_gaia

    def crossmatch_gaia(self, plate_solution=None, star_catalog=None):
        STATE.xmatch_count += 1
        n = STATE.xmatch_count
        d = os.path.join(STATE.plate_dir, 'crossmatch')
        prefix = os.path.join(d, 'xmatch_{:02d}_'.format(n))

        if n == 1:
            write_table(self, prefix + 'sources_in.fits')

        if star_catalog is not None:
            write_table(star_catalog, prefix + 'catalog_in.fits')

        if plate_solution is not None:
            sol_info = {
                'source_table_plate_epoch': self.plate_epoch,
                'num_solutions': plate_solution.num_solutions,
                'mean_pixel_scale': plate_solution.mean_pixel_scale,
                'imwidth': plate_solution.imwidth,
                'imheight': plate_solution.imheight,
                'solutions': [{k: s[k] for k in
                               ['solution_num', 'header_wcs', 'scamp_sigma_1',
                                'scamp_sigma_2', 'scamp_ndeg']}
                              for s in plate_solution.solutions]}

            with open(prefix + 'solution.json', 'w') as f:
                json.dump(encode(sol_info), f, indent=1)

        result = orig_crossmatch(self, plate_solution, star_catalog)
        cols = [c for c in MATCH_COLUMNS if c in self.colnames]
        write_table(self[cols], prefix + 'matches.fits')

        if getattr(self, 'neighbors_gaia', None) is not None:
            write_table(self.neighbors_gaia, prefix + 'neighbors.fits')

        return result

    SourceTable.crossmatch_gaia = crossmatch_gaia

    # Process.finish() removes the scratch directory: save the results first
    orig_finish = Process.finish

    def finish(self, completed=1):
        try:
            save_results(self, completed)
        except Exception:
            STATE.errors.append(traceback.format_exc())
            traceback.print_exc()

        return orig_finish(self, completed)

    Process.finish = finish


def save_results(proc, completed):
    d = STATE.plate_dir
    results = {'process': {k: getattr(proc, k, None) for k in
                           ['filename', 'completed', 'plate_epoch', 'plate_year',
                            'imwidth', 'imheight', 'num_sources',
                            'plate_solved', 'num_iterations', 'archive_id',
                            'scan_id', 'plate_id', 'process_id']}}
    results['process']['completed'] = completed

    if proc.sources is not None:
        write_table(proc.sources, os.path.join(d, 'sources.fits'))
        results['sources'] = {'rows': len(proc.sources),
                              'columns': proc.sources.colnames,
                              'stats': column_stats(proc.sources)}

    psol = proc.plate_solution

    if psol is not None:
        attrs = ['num_solutions', 'num_duplicate_solutions', 'pattern_ratio',
                 'mean_pixel_scale', 'min_pixel_scale', 'max_pixel_scale',
                 'mean_fov1', 'mean_fov2', 'source_density', 'max_separation',
                 'exp_numbers', 'centroid', 'radius']
        results['plate_solution'] = {a: getattr(psol, a, None) for a in attrs}

        for kind in ['solutions', 'duplicate_solutions']:
            sols = getattr(psol, kind, None) or []
            results['plate_solution'][kind] = []

            for i, sol in enumerate(sols):
                if sol.get('healpix_table') is not None:
                    write_table(sol['healpix_table'],
                                os.path.join(d, '{}_{:02d}_healpix.fits'
                                             .format(kind, i + 1)))

                results['plate_solution'][kind].append(dict(sol))

        if getattr(psol, 'pattern_table', None) is not None:
            write_table(psol.pattern_table,
                        os.path.join(d, 'scanner_pattern.fits'))

    results['phot_calib'] = list(proc.phot_calib_list or [])

    # Calibration curves are functions: evaluate them as
    # Process.output_calibration_db() does
    curves = []

    for i, calib in enumerate(results['phot_calib']):
        try:
            curve = proc.phot_calib_curves[i]
            mags = np.arange(np.floor(calib['plate_mag_brightest'] * 10) / 10.,
                             np.ceil(calib['plate_mag_lim'] * 10) / 10. + 0.01,
                             0.1)
            curves.append({'solution_num': calib['solution_num'],
                           'iteration': calib['iteration'],
                           'plate_mag': mags,
                           'natmag': [curve(m).item() for m in mags]})
        except Exception as e:
            curves.append({'error': repr(e)})

    results['phot_calib_curves'] = curves

    with open(os.path.join(d, 'summary.json'), 'w') as f:
        json.dump(encode(results), f, indent=1)


# --- Configuration ---------------------------------------------------------

def base_config():
    """TOML data of plates.conf and paths.local, and the [Reference]
    output directory."""
    ini = read_ini([PLATES_CONF, PATHS_LOCAL])
    output_dir = ini.get('Reference', 'output_dir')
    ini.remove_section('Reference')
    text, _ = convert(ini)
    return tomllib.loads(text), output_dir


def plate_config(data, plate_dir):
    """Configuration with the output directories inside plate_dir, and the
    recording database; also written to plate_dir/config.toml."""
    files = data['Files']

    for key, sub in [('work_dir', 'work'), ('write_log_dir', 'logs'),
                     ('write_header_dir', 'headers'),
                     ('write_wcs_dir', 'wcs'), ('write_source_dir', 'sources'),
                     ('write_fits_dir', 'fits')]:
        path = os.path.join(plate_dir, sub)
        os.makedirs(path, exist_ok=True)
        files[key] = path

    # Any connection enables the database output; the recording backend is
    # used for pgsql
    data.setdefault('Database', {})['connection'] = {'rdbms': 'pgsql',
                                                     'schema': SCHEMA}
    cfg = config_from_dict(data)

    from pyplate.config.ini2toml import to_toml

    with open(os.path.join(plate_dir, 'config.toml'), 'w') as f:
        f.write(to_toml(data))

    return cfg


# --- Run -------------------------------------------------------------------

def run_plate(filename, run_dir, replay_run_dir, data):
    basefn = os.path.splitext(filename)[0]
    plate_dir = os.path.join(run_dir, basefn)
    os.makedirs(plate_dir)

    STATE.plate_dir = plate_dir
    STATE.cursor = ReferenceCursor()
    STATE.xmatch_count = 0
    STATE.gaia_index = []
    STATE.errors = []
    STATE.replay = None
    STATE.replay_dir = None

    if replay_run_dir:
        STATE.replay_dir = os.path.join(replay_run_dir, basefn, 'gaia')

        with open(os.path.join(STATE.replay_dir, 'queries.json')) as f:
            STATE.replay = json.load(f)

    cfg = plate_config(json.loads(json.dumps(data)), plate_dir)
    pipe = PlatePipeline()
    pipe.assign_conf(cfg)

    try:
        pipe.single_image(filename)
    except Exception:
        STATE.errors.append(traceback.format_exc())
        traceback.print_exc()

    # Gaia query index
    if STATE.gaia_index:
        with open(os.path.join(plate_dir, 'gaia', 'queries.json'), 'w') as f:
            json.dump(STATE.gaia_index, f, indent=1)

    # Recorded SQL: all statements, and the small result tables
    queries = STATE.cursor.queries

    with open(os.path.join(plate_dir, 'results.sql'), 'w') as f:
        f.write(format_queries('pgsql', [(basefn, queries)]))

    small = [(sql, params) for sql, params in queries
             if not sql.startswith('--')
             and not any(' {}.{} '.format(SCHEMA, t) in sql
                         for t in LARGE_TABLES)]

    with open(os.path.join(plate_dir, 'results_small.sql'), 'w') as f:
        f.write(format_queries('pgsql', [(basefn, small)]))

    # Final header as text
    fn_hdr = os.path.join(plate_dir, 'headers', basefn + '.hdr')

    if os.path.exists(fn_hdr):
        h = fits.Header.fromfile(fn_hdr, sep='', endcard=True, padding=True)

        with open(os.path.join(plate_dir, 'header.txt'), 'w') as f:
            f.write(h.tostring(sep='\n') + '\n')

    return {'filename': filename, 'errors': STATE.errors,
            'gaia_queries': len(STATE.gaia_index),
            'crossmatch_calls': STATE.xmatch_count}


def program_version(path):
    if not path:
        return None

    for flag in ['--version', '-v']:
        try:
            out = subprocess.run([path, flag], capture_output=True, text=True,
                                 timeout=60)
        except Exception:
            continue

        lines = [l.strip() for l in (out.stdout + out.stderr).splitlines()
                 if l.strip()]

        if lines:
            return lines[0]

    return None


def package_versions():
    out = {}

    for p in ['numpy', 'scipy', 'astropy', 'statsmodels', 'healpy', 'ephem',
              'pyvo', 'keras', 'tensorflow', 'scikit-learn', 'Pillow',
              'pytz', 'unidecode', 'pytimeparse']:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None

    return out


def git_info():
    def git(*args):
        return subprocess.run(['git', '-C', ROOT] + list(args),
                              capture_output=True, text=True).stdout.strip()

    return {'commit': git('rev-parse', 'HEAD'),
            'pyplate_dir_modified': bool(git('status', '--porcelain',
                                             '--', 'pyplate'))}


def write_manifest(run_dir, run, replay, plate_results, cfg):
    files = cfg.files
    inputs = {'plates_conf': file_entry(PLATES_CONF, PLATES_DIR),
              'scans': [], 'metadata': [], 'index_files': [],
              'gaia_files': []}

    for fn in PLATES:
        path = os.path.join(files.fits_dir, fn)

        if os.path.exists(path):
            inputs['scans'].append(file_entry(path, files.fits_dir))

    for key in ['plate_csv', 'scan_csv', 'logbook_csv', 'logpage_csv',
                'preview_csv']:
        for fn in getattr(files, key):
            inputs['metadata'].append(
                file_entry(os.path.join(files.csv_dir, fn), files.csv_dir))

    # Index files of the plate years used, and the Gaia file for solving
    years = set()

    for p in plate_results:
        fn = os.path.join(run_dir, os.path.splitext(p['filename'])[0],
                          'summary.json')

        if os.path.exists(fn):
            with open(fn) as f:
                year = json.load(f)['process'].get('plate_year')

            if year:
                years.add(int(year))

    for year in sorted(years):
        index_dir = os.path.join(files.index_dir, 'index_{:d}'.format(year))

        if os.path.isdir(index_dir):
            for fn in sorted(os.listdir(index_dir)):
                if fn.startswith('.'):
                    continue

                inputs['index_files'].append(
                    file_entry(os.path.join(index_dir, fn), files.index_dir))

    fn_gaia = os.path.join(files.gaia_dir, 'gaiaedr3_pyplate.fits')

    if os.path.exists(fn_gaia):
        inputs['gaia_files'].append(file_entry(fn_gaia, files.gaia_dir))

    outputs = []

    for dirpath, _, filenames in os.walk(run_dir):
        for fn in sorted(filenames):
            if fn != 'manifest.json':
                outputs.append(file_entry(os.path.join(dirpath, fn), run_dir))

    programs = {k: program_version(getattr(cfg.programs, k)) for k in PROGRAMS}

    manifest = {
        'run': run,
        'replay_gaia': replay,
        'created_utc': datetime.datetime.now(datetime.timezone.utc)
                       .strftime('%Y-%m-%dT%H:%M:%S'),
        'pyplate': git_info(),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': package_versions(),
        'programs': programs,
        'fixed_values': {'utcnow': UTCNOW.strftime('%Y-%m-%dT%H:%M:%S'),
                         'version': VERSION, 'hostname': HOSTNAME,
                         'scan_id': SCAN_ID, 'plate_id': PLATE_ID},
        'plates': plate_results,
        'inputs': inputs,
        'outputs': sorted(outputs, key=lambda e: e['name']),
    }

    with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', help='name of the run directory')
    parser.add_argument('--replay-gaia', metavar='RUN',
                        help='reuse the Gaia query results of another run')
    parser.add_argument('--plates', nargs='+', default=PLATES,
                        help='scan filenames (default: %(default)s)')
    args = parser.parse_args()

    if not os.path.exists(PATHS_LOCAL):
        sys.exit('Missing {}; see paths.local.example'
                 .format(os.path.relpath(PATHS_LOCAL, ROOT)))

    data, ref_dir = base_config()
    run_dir = os.path.join(ref_dir, args.run)

    if os.path.exists(run_dir):
        sys.exit('Run directory exists: {}'.format(run_dir))

    replay_dir = (os.path.join(ref_dir, args.replay_gaia)
                  if args.replay_gaia else None)

    os.makedirs(run_dir)
    install_wrappers()
    results = [run_plate(fn, run_dir, replay_dir, data) for fn in args.plates]
    write_manifest(run_dir, args.run, args.replay_gaia, results,
                   config_from_dict(data))

    for r in results:
        print('{}: {} crossmatch calls, {} Gaia queries, {} errors'
              .format(r['filename'], r['crossmatch_calls'],
                      r['gaia_queries'], len(r['errors'])))

    print('Results in', run_dir)


if __name__ == '__main__':
    main()
