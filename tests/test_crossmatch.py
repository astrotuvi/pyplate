"""The crossmatch package gives the results of the old crossmatch_gaia().

The inputs and outputs of the three crossmatch calls of the old code on a
part of each reference plate are in tests/ver4/crossmatch (see
tests/ver4/README.md, section "Crossmatch"). The calls are repeated on the
same source table, as the pipeline does, with the epoch of the old code
(1950); the source columns after each call and the neighbour tables must be
the same.
"""
import json
import os

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.table import MaskedColumn, Table

from pyplate.process.crossmatch import (NearestMatcher, ReferenceStars,
                                        find_pairs, tolerance)
from pyplate.process.solve import Solution, SolutionSet
from pyplate.process.sources import SourceTable

HERE = os.path.dirname(os.path.abspath(__file__))
XMATCH_DIR = os.path.join(HERE, 'ver4', 'crossmatch')
PLATES = ['LA02333_x', 'LA02334_x']
OLD_EPOCH = 1950

# Source columns written by the crossmatch (DR5 name, DR4 name)
MATCH_COLUMNS = [('solution_num', 'solution_num'),
                 ('match_radius', 'match_radius'),
                 ('gaiadr4_id', 'gaiaedr3_id'),
                 ('gaiadr4_gmag', 'gaiaedr3_gmag'),
                 ('gaiadr4_bpmag', 'gaiaedr3_bpmag'),
                 ('gaiadr4_rpmag', 'gaiaedr3_rpmag'),
                 ('gaiadr4_bp_rp', 'gaiaedr3_bp_rp'),
                 ('gaiadr4_dist', 'gaiaedr3_dist'),
                 ('gaiadr4_neighbors', 'gaiaedr3_neighbors')]


class Log:
    def write(self, *args, **kwargs):
        pass


def quantity(value):
    return u.Quantity(value['quantity'], value['unit'])


def solution_set(plate):
    with open(os.path.join(XMATCH_DIR, plate, 'solution.json')) as f:
        info = json.load(f)

    solutions = [Solution(solution_num=s['solution_num'],
                          header_wcs=fits.Header.fromstring(
                              s['header_wcs']['header'], sep='\n'),
                          scamp_sigma_1=quantity(s['scamp_sigma_1']),
                          scamp_sigma_2=quantity(s['scamp_sigma_2']),
                          scamp_ndeg=s['scamp_ndeg'])
                 for s in info['solutions']]
    return SolutionSet(num_solutions=len(solutions), solutions=solutions,
                       mean_pixel_scale=quantity(info['mean_pixel_scale']),
                       imwidth=info['imwidth'], imheight=info['imheight'])


def source_table(plate):
    tab = Table.read(os.path.join(XMATCH_DIR, plate, 'sources.fits'))
    sources = SourceTable()
    sources.populate(num_sources=len(tab))

    for col in tab.colnames:
        sources[col] = tab[col]

    return sources


def crossmatch(sources, catalog, solutions):
    refs = ReferenceStars.from_catalog(catalog, solutions, OLD_EPOCH, Log())
    radii = tolerance.radii(solutions)
    pairs = find_pairs(sources, refs, radii.search)
    association = NearestMatcher(radii).associate(sources, refs, pairs)
    sources.add_matches(association, refs)
    return association


def filled(col):
    """Values of a column with masked values as NaN or 0."""
    col = MaskedColumn(col)
    return col.filled(np.nan if col.dtype.kind == 'f' else 0)


def read(plate, kind, n):
    return Table.read(os.path.join(XMATCH_DIR, plate,
                                   '{}_{:02d}.fits'.format(kind, n)))


@pytest.mark.parametrize('plate', PLATES)
def test_crossmatch_calls(plate):
    sources = source_table(plate)
    solutions = solution_set(plate)

    for n in (1, 2, 3):
        association = crossmatch(sources, read(plate, 'catalog', n),
                                 solutions)
        expected = read(plate, 'matches', n)
        assert list(sources['source_num']) == list(expected['source_num'])

        for new, old in MATCH_COLUMNS:
            np.testing.assert_array_equal(filled(sources[new]),
                                          filled(expected[old]),
                                          err_msg='{} call {}'.format(new, n))

        nbs = association.neighbors
        expected = read(plate, 'neighbors', n)
        assert len(nbs) == len(expected)

        for col in ['source_num', 'solution_num', 'dist', 'x_gaia', 'y_gaia',
                    'flag_xmatch']:
            np.testing.assert_array_equal(nbs[col], expected[col],
                                          err_msg='{} call {}'.format(col, n))

        np.testing.assert_array_equal(nbs['gaiadr4_id'],
                                      expected['gaiaedr3_id'])


@pytest.mark.parametrize('plate', PLATES)
def test_matches_of_earlier_calls_are_kept(plate):
    """Very red stars of the first catalogue (G < 12) are missing from the
    deeper catalogues, which are limited in the plate passband; their
    sources keep the first match (DESIGN.md 9)."""
    sources = source_table(plate)
    solutions = solution_set(plate)

    for n in (1, 2, 3):
        association = crossmatch(sources, read(plate, 'catalog', n),
                                 solutions)

    matched = set(sources['source_num'][filled(sources['gaiadr4_id']) > 0])
    kept = matched - set(association.matches['source_num'])
    assert len(kept) > 0
    assert kept.isdisjoint(association.neighbors['source_num'])


def test_nearest_matcher():
    """The nearest star within the match radius is the match; all stars
    within the search radius are neighbours."""
    sources = SourceTable()
    sources.populate(num_sources=3)
    sources['source_num'] = [1, 2, 3]
    sources['x_source'] = [10., 50., 90.]
    sources['y_source'] = [10., 50., 90.]
    catalog = Table({'source_id': [101, 102, 103, 104],
                     'mag': [10., 11., 12., 13.], 'mag1': [10.5] * 4,
                     'mag2': [9.5] * 4, 'color_index': [1.] * 4})
    refs = ReferenceStars(catalog=catalog, index=np.arange(4),
                          solution_num=np.ones(4, dtype=int),
                          xy=np.array([[11., 10.], [10., 13.], [50., 54.],
                                       [200., 200.]]))
    radii = tolerance.Radii(match=2., match_arcsec=2., search=5.,
                            search_arcsec=5., pixel_scale=1. * u.arcsec / u.pixel)
    pairs = find_pairs(sources, refs, radii.search)
    association = NearestMatcher(radii).associate(sources, refs, pairs)

    matches = association.matches
    assert list(matches['source_num']) == [1]
    assert list(matches['gaiadr4_id']) == [101]
    assert list(matches['gaiadr4_dist']) == [1.]
    assert list(matches['gaiadr4_neighbors']) == [2]

    nbs = association.neighbors
    assert sorted(zip(nbs['source_num'], nbs['gaiadr4_id'],
                      nbs['flag_xmatch'])) == [(1, 101, 1), (1, 102, 0),
                                               (2, 103, 0)]

    sources.add_matches(association, refs)
    assert list(filled(sources['gaiadr4_id'])) == [101, 0, 0]
    assert list(sources['gaiadr4_neighbors']) == [2, 1, 0]
    assert sources['gaiadr4_gmag'][0] == 10.
