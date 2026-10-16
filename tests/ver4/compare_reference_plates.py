"""Compare two reference-plate runs made with run_reference_plates.py.

Compares, for each plate, the run NEW against the run REF:

- inputs: the checksums of the scans, metadata, index and Gaia files in the
  manifests (a difference means the runs did not use the same inputs);
- summary.json: every value (numbers by absolute and relative difference;
  FITS headers keyword by keyword, without COMMENT, HISTORY and DATE cards,
  which contain local paths and times);
- sources.fits: rows matched by source_num, column by column;
- crossmatch: the match columns of every crossmatch call, and the
  source–star pairs of the neighbour tables;
- Gaia queries: query parameters and number of rows;
- results.sql: the recorded SQL, line by line (the process duration and
  the COMMENT, HISTORY and DATE cards of stored FITS headers are ignored).

The report lists what differs and by how much; it does not decide what is
acceptable. Runs are given by name (a directory under [Reference]
output_dir in tests/ver4/plates/paths.local) or by path. With --dr5, the
DR5 names of the new run (gaia_* instead of gaiaedr3_*) are mapped back to
the DR4 names before comparing.

Usage (from the repository root):

    python tests/ver4/compare_reference_plates.py run1 run2
"""
import argparse
import configparser
import json
import math
import os
import re
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table

from sqlrender import strip_header_comments

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS_LOCAL = os.path.join(HERE, 'plates', 'paths.local')

# FITS header cards that are not compared: COMMENT and HISTORY hold local
# paths, times and CPU times; DATE is the time the header was written
IGNORED_CARDS = ('COMMENT', 'HISTORY', '', 'DATE')

# DR5 name -> DR4 name, for comparing the redesigned code with the reference
DR5_TO_DR4 = [('num_gaia', 'num_gaia_edr3'), ('gaia_', 'gaiaedr3_')]


def run_dir(name):
    if os.path.isdir(name):
        return name

    conf = configparser.ConfigParser()
    conf.read(PATHS_LOCAL)
    path = os.path.join(conf.get('Reference', 'output_dir'), name)

    if not os.path.isdir(path):
        sys.exit('Run not found: {}'.format(name))

    return path


def dr4_name(name, dr5):
    if not dr5:
        return name

    for new, old in DR5_TO_DR4:
        if name.startswith(new) and not name.startswith(old):
            return old + name[len(new):]

    return name


def decode_marker(val):
    if isinstance(val, dict):
        if 'nan' in val:
            return float('nan')
        if 'inf' in val:
            return float('inf') * val['inf']
    return val


def flatten(val, path='', out=None):
    """Flatten JSON into {path: value}; FITS headers keyword by keyword."""
    if out is None:
        out = {}

    val = decode_marker(val)

    if isinstance(val, dict) and set(val) == {'header'}:
        h = fits.Header.fromstring(val['header'], sep='\n')

        for card in h.cards:
            if card.keyword not in IGNORED_CARDS:
                out['{}.{}'.format(path, card.keyword)] = card.value
    elif isinstance(val, dict):
        for k, v in val.items():
            flatten(v, '{}.{}'.format(path, k) if path else k, out)
    elif isinstance(val, list):
        for i, v in enumerate(val):
            flatten(v, '{}[{}]'.format(path, i), out)
    else:
        out[path] = val

    return out


def numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare_values(ref, new):
    """Return a description of the difference, or None if equal."""
    if numeric(ref) and numeric(new):
        if math.isnan(ref) and math.isnan(new):
            return None
        if ref == new:
            return None
        diff = abs(new - ref)
        rel = diff / abs(ref) if ref else float('inf')
        return {'ref': ref, 'new': new, 'abs': diff, 'rel': rel}

    if ref == new or (str(ref) == str(new)):
        return None

    return {'ref': ref, 'new': new}


def compare_summary(ref_file, new_file, dr5, max_lines):
    with open(ref_file) as f:
        ref = flatten(json.load(f))

    with open(new_file) as f:
        new = {dr4_name_path(k, dr5): v
               for k, v in flatten(json.load(f)).items()}

    missing = sorted(set(ref) - set(new))
    extra = sorted(set(new) - set(ref))
    diffs = {}

    for k in sorted(set(ref) & set(new)):
        d = compare_values(ref[k], new[k])

        if d is not None:
            diffs[k] = d

    print('  summary.json: {} values, {} differ, {} missing, {} extra'
          .format(len(ref), len(diffs), len(missing), len(extra)))
    ranked = sorted(diffs.items(),
                    key=lambda kv: -kv[1].get('rel', float('inf')))

    for k, d in ranked[:max_lines]:
        if 'abs' in d:
            print('    {}: {} -> {} (abs {:.3g}, rel {:.3g})'
                  .format(k, d['ref'], d['new'], d['abs'], d['rel']))
        else:
            print('    {}: {!r:.60} -> {!r:.60}'.format(k, d['ref'], d['new']))

    for k in missing[:max_lines]:
        print('    missing: {}'.format(k))

    for k in extra[:max_lines]:
        print('    extra: {}'.format(k))


def dr4_name_path(path, dr5):
    if not dr5:
        return path

    return '.'.join(dr4_name(p, dr5) for p in path.split('.'))


def read_table(path, dr5=False):
    tab = Table.read(path)

    for col in list(tab.colnames):
        old = dr4_name(col, dr5)

        if old != col:
            tab.rename_column(col, old)

    return tab


def column_data(col):
    data = np.ma.asarray(col)
    mask = np.ma.getmaskarray(data)

    if data.dtype.kind == 'f':
        mask = mask | ~np.isfinite(np.ma.filled(data, np.nan))

    return np.ma.getdata(data), mask


def compare_tables(name, ref_file, new_file, key, dr5, max_lines):
    ref = read_table(ref_file)
    new = read_table(new_file, dr5)
    print('  {}: {} rows -> {} rows'.format(name, len(ref), len(new)))

    if key not in ref.colnames or key not in new.colnames:
        print('    no {} column, not compared row by row'.format(key))
        return

    ref.sort(key)
    new.sort(key)
    common = np.intersect1d(ref[key], new[key])

    if len(common) < max(len(ref), len(new)):
        print('    {} {} values only in REF, {} only in NEW'
              .format(len(ref) - len(common), key, len(new) - len(common)))

    ref = ref[np.isin(ref[key], common)]
    new = new[np.isin(new[key], common)]
    missing = [c for c in ref.colnames if c not in new.colnames]
    extra = [c for c in new.colnames if c not in ref.colnames]
    lines = []

    for col in ref.colnames:
        if col not in new.colnames:
            continue

        a, ma = column_data(ref[col])
        b, mb = column_data(new[col])

        if a.shape != b.shape:
            lines.append('    {}: different shape {} -> {}'
                         .format(col, a.shape, b.shape))
            continue

        valid_diff = int((ma != mb).sum())
        both = ~ma & ~mb

        if a.dtype.kind in 'fc' or b.dtype.kind in 'fc':
            d = np.abs(a[both].astype(float) - b[both].astype(float))
            n_diff = int((d > 0).sum())

            if n_diff or valid_diff:
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel = d / np.abs(a[both].astype(float))
                lines.append('    {}: {} values differ (max abs {:.3g}, max '
                             'rel {:.3g}), {} differ in validity'
                             .format(col, n_diff, d.max() if len(d) else 0,
                                     np.nanmax(rel) if len(d) else 0,
                                     valid_diff))
        else:
            n_diff = int((a[both] != b[both]).sum())

            if n_diff or valid_diff:
                lines.append('    {}: {} values differ, {} differ in validity'
                             .format(col, n_diff, valid_diff))

    print('    {} of {} columns differ'.format(len(lines), len(ref.colnames)))

    for line in lines[:max_lines]:
        print(line)

    if missing:
        print('    columns missing: {}'.format(', '.join(missing)))

    if extra:
        print('    columns extra: {}'.format(', '.join(extra)))


def compare_pairs(name, ref_file, new_file, dr5):
    ref = read_table(ref_file)
    new = read_table(new_file, dr5)
    cols = ['source_num', 'gaiaedr3_id', 'solution_num']
    a = set(map(tuple, np.array([np.asarray(ref[c]) for c in cols]).T))
    b = set(map(tuple, np.array([np.asarray(new[c]) for c in cols]).T))
    print('  {}: {} pairs -> {} pairs, {} only in REF, {} only in NEW'
          .format(name, len(a), len(b), len(a - b), len(b - a)))


def compare_sql(ref_file, new_file, dr5, max_lines):
    def lines(path, rename):
        with open(path) as f:
            text = f.read()

        if rename:
            for new, old in DR5_TO_DR4:
                text = re.sub(r'\b' + new, old, text)

        text = re.sub(r'duration=\d+', 'duration=N', text)
        text = strip_header_comments(text)
        text = re.sub(r"DATE    = '+[0-9T:-]+'+", 'DATE = X', text)
        return text.splitlines()

    a = lines(ref_file, False)
    b = lines(new_file, dr5)
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print('  results.sql: {} lines -> {} lines, {} differ'
          .format(len(a), len(b), len(diffs)))

    for i, x, y in diffs[:max_lines]:
        print('    line {}:\n      REF {:.150}\n      NEW {:.150}'
              .format(i + 1, x, y))


def compare_inputs(ref_manifest, new_manifest):
    def checksums(m):
        out = {}

        for kind, entries in m['inputs'].items():
            for e in entries if isinstance(entries, list) else [entries]:
                out['{}/{}'.format(kind, e['name'])] = e['sha256']

        return out

    a, b = checksums(ref_manifest), checksums(new_manifest)
    differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    print('Inputs: {} files, {} differ'.format(len(a), len(differ)))

    for k in differ:
        print('  {}: {} -> {}'.format(k, a.get(k, 'missing')[:12],
                                      b.get(k, 'missing')[:12]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('ref', help='reference run (name or path)')
    parser.add_argument('new', help='run to compare (name or path)')
    parser.add_argument('--dr5', action='store_true',
                        help='the new run uses DR5 column names')
    parser.add_argument('--max-lines', type=int, default=20,
                        help='maximum number of differences listed per item')
    args = parser.parse_args()

    ref_dir, new_dir = run_dir(args.ref), run_dir(args.new)

    with open(os.path.join(ref_dir, 'manifest.json')) as f:
        ref_manifest = json.load(f)

    with open(os.path.join(new_dir, 'manifest.json')) as f:
        new_manifest = json.load(f)

    print('REF {} (pyplate {}), NEW {} (pyplate {})'
          .format(ref_manifest['run'], ref_manifest['pyplate']['commit'][:10],
                  new_manifest['run'], new_manifest['pyplate']['commit'][:10]))
    compare_inputs(ref_manifest, new_manifest)

    for plate in ref_manifest['plates']:
        basefn = os.path.splitext(plate['filename'])[0]
        r = os.path.join(ref_dir, basefn)
        n = os.path.join(new_dir, basefn)
        print('\n{}'.format(plate['filename']))

        if not os.path.isdir(n):
            print('  not in NEW')
            continue

        for fn in ['summary.json', 'results.sql', 'sources.fits']:
            if not os.path.exists(os.path.join(n, fn)):
                print('  {} not in NEW'.format(fn))

        if os.path.exists(os.path.join(n, 'summary.json')):
            compare_summary(os.path.join(r, 'summary.json'),
                            os.path.join(n, 'summary.json'), args.dr5,
                            args.max_lines)

        if os.path.exists(os.path.join(n, 'sources.fits')):
            compare_tables('sources.fits', os.path.join(r, 'sources.fits'),
                           os.path.join(n, 'sources.fits'), 'source_num',
                           args.dr5, args.max_lines)

        xdir = os.path.join(r, 'crossmatch')

        for fn in sorted(os.listdir(xdir)) if os.path.isdir(xdir) else []:
            nf = os.path.join(n, 'crossmatch', fn)

            if not os.path.exists(nf):
                print('  crossmatch/{} not in NEW'.format(fn))
            elif fn.endswith('_matches.fits'):
                compare_tables('crossmatch/' + fn, os.path.join(xdir, fn), nf,
                               'source_num', args.dr5, args.max_lines)
            elif fn.endswith('_neighbors.fits'):
                compare_pairs('crossmatch/' + fn, os.path.join(xdir, fn), nf,
                              args.dr5)

        gref = os.path.join(r, 'gaia', 'queries.json')
        gnew = os.path.join(n, 'gaia', 'queries.json')

        if os.path.exists(gref) and os.path.exists(gnew):
            with open(gref) as f:
                qa = json.load(f)

            with open(gnew) as f:
                qb = json.load(f)

            same = sum(1 for x, y in zip(qa, qb)
                       if x['key'] == y['key'] and x['rows'] == y['rows'])
            print('  Gaia queries: {} -> {}, {} with the same parameters and '
                  'rows'.format(len(qa), len(qb), same))

        if os.path.exists(os.path.join(n, 'results.sql')):
            compare_sql(os.path.join(r, 'results.sql'),
                        os.path.join(n, 'results.sql'), args.dr5,
                        args.max_lines)


if __name__ == '__main__':
    main()
