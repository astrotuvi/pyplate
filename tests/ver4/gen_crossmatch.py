"""Cut the crossmatch inputs and outputs of the old code for a small part of
each reference plate into tests/ver4/crossmatch/.

Reads the crossmatch files of the reference run (default run1) that
run_reference_plates.py saved for every crossmatch_gaia() call
(crossmatch/xmatch_NN_*), see README.md, section "Reference plates". For each
plate it keeps the sources in a box of the image and the Gaia stars that
fall near the box (at the epoch of the old crossmatch, 1950, with any of the
solutions), so that the matches and neighbours of the kept sources are the
same as on the whole plate:

- sources.fits: source_num, x_source and y_source of the kept sources (the
  input of the first call; the coordinates do not change between calls);
- solution.json: the solutions (the same in every call);
- catalog_NN.fits: the kept stars of the catalogue of call NN;
- matches_NN.fits: the match columns of the kept sources after call NN
  (DR4 column names, as the old code wrote them);
- neighbors_NN.fits: the neighbour pairs of the kept sources of call NN.

The boxes contain sources that keep a match of an earlier call (see
DESIGN.md, section 9). Needs only numpy and astropy. Run from the
repository root on the machine with the reference run:

    python tests/ver4/gen_crossmatch.py [run]
"""
import configparser
import json
import os
import shutil
import sys

import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'crossmatch')
PATHS_LOCAL = os.path.join(HERE, 'plates', 'paths.local')

# Kept part of each plate: x and y range of the sources (pixels)
BOXES = {'LA02333_x': ((4600, 6100), (2100, 3600)),
         'LA02334_x': ((4500, 6000), (2100, 3600))}

# Stars are kept within this distance of the box (pixels), more than the
# search radius of the crossmatch
MARGIN = 100

# Epoch of the old crossmatch
EPOCH = 1950

NUM_CALLS = 3


def solution_headers(info):
    return [fits.Header.fromstring(s['header_wcs']['header'], sep='\n')
            for s in info['solutions']]


def star_mask(catalog, info, box):
    """Stars that fall within MARGIN of the box with any solution."""
    (x0, x1), (y0, y1) = box
    ra = (catalog['ra'] + (EPOCH - catalog['ref_epoch']) * catalog['pmra']
          / np.cos(np.radians(catalog['dec'])) / 3600000.)
    dec = (catalog['dec'] + (EPOCH - catalog['ref_epoch']) * catalog['pmdec']
           / 3600000.)
    keep = np.zeros(len(catalog), dtype=bool)

    for header in solution_headers(info):
        x, y = wcs.WCS(header).all_world2pix(ra, dec, 1, quiet=True)
        keep |= ((x > x0 - MARGIN) & (x < x1 + MARGIN) &
                 (y > y0 - MARGIN) & (y < y1 + MARGIN))

    return keep


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'run1'
    paths = configparser.ConfigParser()
    paths.read(PATHS_LOCAL)
    run_dir = os.path.join(paths['Reference']['output_dir'], run)

    for plate, box in BOXES.items():
        src_dir = os.path.join(run_dir, plate, 'crossmatch')
        out = os.path.join(OUT_DIR, plate)
        os.makedirs(out, exist_ok=True)
        (x0, x1), (y0, y1) = box

        sources = Table.read(os.path.join(src_dir, 'xmatch_01_sources_in.fits'))
        mask = ((sources['x_source'] >= x0) & (sources['x_source'] < x1) &
                (sources['y_source'] >= y0) & (sources['y_source'] < y1))
        sources = sources[mask]['source_num', 'x_source', 'y_source']
        sources.write(os.path.join(out, 'sources.fits'), overwrite=True)
        source_nums = set(sources['source_num'])

        with open(os.path.join(src_dir, 'xmatch_01_solution.json')) as f:
            info = json.load(f)

        shutil.copy(os.path.join(src_dir, 'xmatch_01_solution.json'),
                    os.path.join(out, 'solution.json'))

        for n in range(1, NUM_CALLS + 1):
            prefix = os.path.join(src_dir, 'xmatch_{:02d}_'.format(n))

            with open(prefix + 'solution.json') as f:
                assert json.load(f) == info

            catalog = Table.read(prefix + 'catalog_in.fits')
            catalog = catalog[star_mask(catalog, info, box)]
            catalog.write(os.path.join(out, 'catalog_{:02d}.fits'.format(n)),
                          overwrite=True)

            for kind in ['matches', 'neighbors']:
                tab = Table.read(prefix + kind + '.fits')
                tab = tab[[s in source_nums for s in tab['source_num']]]
                tab.write(os.path.join(out, '{}_{:02d}.fits'.format(kind, n)),
                          overwrite=True)

        print('{}: {} sources'.format(plate, len(sources)))


if __name__ == '__main__':
    main()
