import os
import numpy as np
import warnings
from typing import TYPE_CHECKING
from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn, vstack
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.coordinates import Galactic, GeocentricMeanEcliptic
from astropy.coordinates import match_coordinates_sky
from astropy import units as u
from astropy.time import Time
from collections import OrderedDict

try:
    from scipy.spatial import cKDTree as KDT
except ImportError:
    from scipy.spatial import KDTree as KDT

try:
    import healpy
    have_healpy = True
except ImportError:
    have_healpy = False

from .log import ProcessLog

if TYPE_CHECKING:
    from .crossmatch import Association, ReferenceStars


_source_meta = OrderedDict([
    ('source_num',          ('i4', '%7d', 'NUMBER')),
    ('x_source',            ('f8', '%11.4f', 'X_IMAGE')),
    ('y_source',            ('f8', '%11.4f', 'Y_IMAGE')),
    ('erra_source',         ('f4', '%9.5f', 'ERRA_IMAGE')),
    ('errb_source',         ('f4', '%9.5f', 'ERRB_IMAGE')),
    ('errtheta_source',     ('f4', '%6.2f', 'ERRTHETA_IMAGE')),
    ('a_source',            ('f4', '%9.3f', 'A_IMAGE')),
    ('b_source',            ('f4', '%9.3f', 'B_IMAGE')),
    ('theta_source',        ('f4', '%6.2f', 'THETA_IMAGE')),
    ('elongation',          ('f4', '%8.3f', 'ELONGATION')),
    ('x_peak',              ('i4', '%6d', 'XPEAK_IMAGE')),
    ('y_peak',              ('i4', '%6d', 'YPEAK_IMAGE')),
    ('flag_usepsf',         ('i1', '%1d', '')),
    ('x_image',             ('f8', '%11.4f', 'X_IMAGE')),
    ('y_image',             ('f8', '%11.4f', 'Y_IMAGE')),
    ('erra_image',          ('f4', '%9.5f', 'ERRA_IMAGE')),
    ('errb_image',          ('f4', '%9.5f', 'ERRB_IMAGE')),
    ('errtheta_image',      ('f4', '%6.2f', 'ERRTHETA_IMAGE')),
    ('x_psf',               ('f8', '%11.4f', '')),
    ('y_psf',               ('f8', '%11.4f', '')),
    ('erra_psf',            ('f4', '%9.5f', '')),
    ('errb_psf',            ('f4', '%9.5f', '')),
    ('errtheta_psf',        ('f4', '%6.2f', '')),
    ('mag_auto',            ('f4', '%7.4f', 'MAG_AUTO')),
    ('magerr_auto',         ('f4', '%7.4f', 'MAGERR_AUTO')),
    ('flux_auto',           ('f4', '%12.5e', 'FLUX_AUTO')),
    ('fluxerr_auto',        ('f4', '%12.5e', 'FLUXERR_AUTO')),
    ('mag_iso',             ('f4', '%7.4f', 'MAG_ISO')),
    ('magerr_iso',          ('f4', '%7.4f', 'MAGERR_ISO')),
    ('flux_iso',            ('f4', '%12.5e', 'FLUX_ISO')),
    ('fluxerr_iso',         ('f4', '%12.5e', 'FLUXERR_ISO')),
    ('flux_max',            ('f4', '%12.5e', 'FLUX_MAX')),
    ('flux_radius',         ('f4', '%12.5e', 'FLUX_RADIUS')),
    ('fwhm_image',          ('f4', '%12.5e', 'FWHM_IMAGE')),
    ('isoarea',             ('i4', '%6d', 'ISOAREA_IMAGE')),
    ('sqrt_isoarea',        ('f4', '%12.5e', '')),
    ('background',          ('f4', '%12.5e', 'BACKGROUND')),
    ('sextractor_flags',    ('i2', '%3d', 'FLAGS')),
    ('dist_center',         ('f4', '%9.3f', '')),
    ('dist_edge',           ('f4', '%9.3f', '')),
    ('annular_bin',         ('i2', '%1d', '')),
    ('flag_negradius',      ('i1', '%1d', '')),
    ('flag_rim',            ('i1', '%1d', '')),
    ('flag_clean',          ('i1', '%1d', '')),
    ('model_prediction',    ('f4', '%7.5f', '')),
    ('solution_num',        ('i2', '%1d', '')),
    ('ra_icrs',             ('f8', '%11.7f', '')),
    ('dec_icrs',            ('f8', '%11.7f', '')),
    ('ra_error',            ('f4', '%7.4f', '')),
    ('dec_error',           ('f4', '%7.4f', '')),
    ('gal_lat',             ('f8', '%11.7f', '')),
    ('gal_lon',             ('f8', '%11.7f', '')),
    ('ecl_lat',             ('f8', '%11.7f', '')),
    ('ecl_lon',             ('f8', '%11.7f', '')),
    ('x_sphere',            ('f8', '%10.7f', '')),
    ('y_sphere',            ('f8', '%10.7f', '')),
    ('z_sphere',            ('f8', '%10.7f', '')),
    ('healpix256',          ('i4', '%6d', '')),
    ('healpix1024',         ('i4', '%8d', '')),
    ('nn_dist',             ('f4', '%6.3f', '')),
    ('zenith_angle',        ('f4', '%7.4f', '')),
    ('airmass',             ('f4', '%7.4f', '')),
    ('natmag',              ('f4', '%7.4f', '')),
    ('natmag_error',        ('f4', '%7.4f', '')),
    ('bpmag',               ('f4', '%7.4f', '')),
    ('bpmag_error',         ('f4', '%7.4f', '')),
    ('rpmag',               ('f4', '%7.4f', '')),
    ('rpmag_error',         ('f4', '%7.4f', '')),
    ('natmag_plate',        ('f4', '%7.4f', '')),
    ('natmag_correction',   ('f4', '%7.4f', '')),
    ('natmag_residual',     ('f4', '%7.4f', '')),
    ('phot_range_flags',    ('i2', '%1d', '')),
    ('phot_calib_flags',    ('i2', '%1d', '')),
    ('color_term',          ('f4', '%7.4f', '')),
    ('cat_natmag',          ('f4', '%7.4f', '')),
    ('match_radius',        ('f4', '%7.3f', '')),
    ('gaiadr4_id',         ('i8', '%d', '')),
    ('gaiadr4_gmag',       ('f4', '%7.4f', '')),
    ('gaiadr4_bpmag',      ('f4', '%7.4f', '')),
    ('gaiadr4_rpmag',      ('f4', '%7.4f', '')),
    ('gaiadr4_bp_rp',      ('f4', '%7.4f', '')),
    ('gaiadr4_dist',       ('f4', '%6.3f', '')),
    ('gaiadr4_neighbors',  ('i4', '%3d', ''))
])


class SourceTable(Table):
    """
    Source table class

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log = None
        self.plate = None
        self.num_sources_sixbins = None
        self.rel_area_sixbins = None

    def populate(self, num_sources=0):
        """
        Populate table with columns.

        """

        for k in _source_meta:
            zerodata = np.zeros(num_sources, dtype=_source_meta[k][0])
            self.add_column(Column(name=k, dtype=_source_meta[k][0], 
                                   data=zerodata))

        self['flag_usepsf'] = 0
        self['x_psf'] = np.nan
        self['y_psf'] = np.nan
        self['erra_psf'] = np.nan
        self['errb_psf'] = np.nan
        self['errtheta_psf'] = np.nan
        self['ra_icrs'] = np.nan
        self['dec_icrs'] = np.nan
        self['ra_error'] = np.nan
        self['dec_error'] = np.nan
        self['gal_lat'] = np.nan
        self['gal_lon'] = np.nan
        self['ecl_lat'] = np.nan
        self['ecl_lon'] = np.nan
        self['x_sphere'] = np.nan
        self['y_sphere'] = np.nan
        self['z_sphere'] = np.nan
        self['healpix256'] = -1
        self['healpix1024'] = -1
        self['nn_dist'] = np.nan
        self['zenith_angle'] = np.nan
        self['airmass'] = np.nan
        self['gaiadr4_gmag'] = np.nan
        self['gaiadr4_bpmag'] = np.nan
        self['gaiadr4_rpmag'] = np.nan
        self['gaiadr4_bp_rp'] = np.nan
        self['gaiadr4_dist'] = np.nan
        self['gaiadr4_neighbors'] = 0
        self['natmag'] = np.nan
        self['natmag_error'] = np.nan
        self['bpmag'] = np.nan
        self['bpmag_error'] = np.nan
        self['rpmag'] = np.nan
        self['rpmag_error'] = np.nan
        self['natmag_residual'] = np.nan
        self['natmag_correction'] = np.nan
        self['color_term'] = np.nan
        self['cat_natmag'] = np.nan
        self['phot_range_flags'] = 0
        self['phot_calib_flags'] = 0

    def copy_from_sextractor(self, xycat):
        """
        Copy source data from SExtractor output file.

        Parameters
        ----------
        xycat: astropy.io.fits.HDUList object

        """

        for k,v in [(n,_source_meta[n][2]) for n in _source_meta
                    if _source_meta[n][2]]:
            self[k] = xycat[1].data.field(v)

    def apply_scanner_pattern(self, solution_set=None):
        """
        Correct source coordinates for scanner pattern.

        Parameters
        ----------
        solution_set: SolutionSet instance

        """

        from .solve import SolutionSet
        assert isinstance(solution_set, SolutionSet)
        assert solution_set.pattern_ratio is not None

        if solution_set.pattern_ratio > 1.5:
            y_source = self['y_source']
            self['y_source'] = (y_source - solution_set.pattern_y(y_source))
        elif solution_set.pattern_ratio < 2./3.:
            x_source = self['x_source']
            self['x_source'] = (x_source - solution_set.pattern_x(x_source))

    def compute_annular_bins(self, imwidth: int, imheight: int,
                             log: ProcessLog) -> None:
        """
        Compute the distances of the sources from the image centre and the
        nearest edge, and divide the image into 8 concentric annular bins of
        equal area, plus bin 9 at the edges and corners. Sets
        ``dist_center``, ``dist_edge``, ``annular_bin``,
        ``num_sources_sixbins`` and ``rel_area_sixbins``.

        """

        # For brevity, define following variables
        xim = self['x_image']
        yim = self['y_image']

        self['dist_center'] = np.sqrt((xim - imwidth / 2.)**2 +
                                      (yim - imheight / 2.)**2)
        min_halfwidth = min([imwidth, imheight]) / 2.

        # Calculate distance from edge
        distarr = np.column_stack((xim, imwidth-xim,
                                   yim, imheight-yim))
        self['dist_edge'] = np.amin(distarr, 1)

        # Define 8 concentric annular bins + bin9 for edges
        sampling = 100
        imwidth_s = int(imwidth / sampling)
        imheight_s = int(imheight / sampling)
        dist_s = np.zeros((imheight_s, imwidth_s))
        area_all = dist_s.size

        for x in np.arange(imwidth_s):
            for y in np.arange(imheight_s):
                dist_s[y,x] = np.sqrt((x*sampling - imwidth/2.)**2 +
                                 (y*sampling - imheight/2.)**2)

        # Define bin9 as 1/10 of min_halfwidth at all edges
        bin9_width_s = int(0.1 * min_halfwidth / sampling)
        dist_s[:bin9_width_s,:] = -100.
        dist_s[imheight_s-bin9_width_s:,:] = -100.
        dist_s[:,:bin9_width_s] = -100.
        dist_s[:,imwidth_s-bin9_width_s:] = -100.

        # Additionally, include in bin9 corner pixels that are closer
        # to corners than 1/6 of half-diagonal
        bin9_corner_dist = (np.sqrt(imwidth**2 + imheight**2) *
                            5. / 12.)
        dist_s[np.where(dist_s > bin9_corner_dist)] = -100.

        # Exclude bin9 from dist_s
        dist_s = dist_s[np.where(dist_s >= 0)]
        self.rel_area_sixbins = 0.75 * dist_s.size / area_all

        # Divide the rest of pixels between 8 equal-area bins 1-8
        bin_dist = np.array([np.percentile(dist_s, perc)
                             for perc in (np.arange(8)+1.)*100./8.])
        bin_dist[7] = bin9_corner_dist
        bin_dist = np.insert(bin_dist, 0, 0.)

        for b in np.arange(8)+1:
            bbin = ((self['dist_center'] >= bin_dist[b-1]) &
                    (self['dist_center'] < bin_dist[b]))
            nbin = bbin.sum()
            log.write('Annular bin {:d} (radius {:8.2f} pixels): '
                      '{:6d} sources'.format(b, bin_dist[b], nbin),
                      double_newline=False, level=4, event=27)

            if nbin > 0:
                indbin = np.where(bbin)
                self['annular_bin'][indbin] = b

        bbin = ((self['dist_edge'] < 0.1*min_halfwidth) |
                (self['dist_center'] >= bin9_corner_dist))
        nbin = bbin.sum()
        log.write('Annular bin 9 (radius {:8.2f} pixels): '
                  '{:6d} sources'.format(bin9_corner_dist, nbin),
                  level=4, event=27)

        if nbin > 0:
            indbin = np.where(bbin)
            self['annular_bin'][indbin] = 9

        indbin6 = (self['annular_bin'] <= 6)
        self.num_sources_sixbins = indbin6.sum()

    def flag_edge_sources(self, imwidth: int, imheight: int,
                          circular_film: bool, log: ProcessLog) -> None:
        """
        Flag clean sources, sources at the plate rim and sources with
        negative ``flux_radius``. A source is at the rim if its background
        deviates from the mean background near the centre by much more
        than its distance from the edge allows (from the circular rim if
        circular_film). Needs ``dist_center`` (see
        :meth:`compute_annular_bins`).

        """

        # For brevity, define following variables
        xim = self['x_image']
        yim = self['y_image']
        bg = self['background']

        # Calculate mean background in an annulus around the centre of the plate
        min_halfwidth = min([imwidth, imheight]) / 2.
        max_halfwidth = max([imwidth, imheight]) / 2.
        bannulus = ((self['dist_center'] > 0.1*min_halfwidth) &
                    (self['dist_center'] < 0.9*min_halfwidth))

        if bannulus.sum() > 3:
            mean_bg = bg[np.where(bannulus)].mean()
            std_bg = bg[np.where(bannulus)].std()
        else:
            mean_bg = bg.mean()
            std_bg = bg.std()

        # Find and flag dubious stars at the edges
        if circular_film:
            rim_dist = max_halfwidth - self['dist_center']
            self['dist_edge'] = rim_dist

            # Consider sources that are further away from centre than
            # half-width, to be at the rim.
            bneg = (rim_dist <= 0)

            if bneg.sum() > 0:
                rim_dist[np.where(bneg)] = 0.1

            borderbg = max_halfwidth / rim_dist * (np.abs(bg - mean_bg) /
                                                   std_bg)
        else:
            min_xedgedist = np.minimum(xim, imwidth-xim)
            min_yedgedist = np.minimum(yim, imheight-yim)
            min_reldist = np.minimum(min_xedgedist/imwidth,
                                     min_yedgedist/imheight)

            # Avoid dividing by zero
            min_reldist[np.where(min_reldist < 1e-5)] = 1e-5

            # Combine distance from edge with deviation from mean background
            borderbg = (1. / min_reldist) * (np.abs(bg - mean_bg) / std_bg)

        bclean = ((self['flux_radius'] > 0) &
                  (self['elongation'] < 5) &
                  (self['magerr_auto'] < 5) &
                  (borderbg < 100))
        indclean = np.where(bclean)[0]
        self['flag_clean'][indclean] = 1
        log.write('Flagged {:d} clean sources'.format(bclean.sum()),
                  double_newline=False, level=4, event=27)

        indrim = np.where(borderbg >= 100)[0]
        self['flag_rim'][indrim] = 1
        log.write('Flagged {:d} sources at the plate rim'
                  ''.format(len(indrim)), double_newline=False,
                  level=4, event=27)

        indnegrad = np.where(self['flux_radius'] <= 0)[0]
        self['flag_negradius'][indnegrad] = 1
        log.write('Flagged {:d} sources with negative FLUX_RADIUS'
                  ''.format(len(indnegrad)), level=4, event=27)

    def apply_psf_coordinates(self, fn_psfcat: str,
                              log: ProcessLog) -> int | None:
        """
        Replace the coordinates of the sources found in the SExtractor
        catalogue made with the PSF model (peaks within 1 pixel) by the
        PSF coordinates, and set ``flag_usepsf``.

        Returns
        -------
        int or None
            Number of sources with PSF coordinates (0 if the catalogue does
            not exist or is empty), or None if it is corrupt.

        """

        if not os.path.exists(fn_psfcat):
            log.write('Could not read PSF coordinates, '
                      'file {} does not exist!'.format(fn_psfcat),
                      level=2, event=28)
            return 0

        try:
            psfcat = fits.open(fn_psfcat)
        except IOError:
            log.write('Could not read PSF coordinates, file {} '
                      'is corrupt'.format(fn_psfcat),
                      level=2, event=28)
            return None

        if psfcat[1].header['NAXIS2'] == 0:
            log.write('There are no sources with PSF coordinates!',
                      level=2, event=28)
            return 0

        num_sources = len(self)
        xpeakpsf = psfcat[1].data.field('XPEAK_IMAGE')
        ypeakpsf = psfcat[1].data.field('YPEAK_IMAGE')

        # Match sources in two lists (distance < 1 px)
        coords1 = np.empty((num_sources, 2))
        coords1[:,0] = self['x_peak']
        coords1[:,1] = self['y_peak']
        coords2 = np.empty((xpeakpsf.size, 2))
        coords2[:,0] = xpeakpsf
        coords2[:,1] = ypeakpsf
        kdt = KDT(coords2)
        ds,ind2 = kdt.query(coords1)
        ind1 = np.arange(num_sources)
        indmask = ds < 1.
        ind1 = ind1[indmask]
        ind2 = ind2[indmask]

        num_psf_sources = len(ind1)
        log.write('Replacing x,y values from PSF photometry '
                  'for {:d} sources'.format(num_psf_sources),
                  level=4, event=28)
        self['x_psf'][ind1] = psfcat[1].data.field('XPSF_IMAGE')[ind2]
        self['y_psf'][ind1] = psfcat[1].data.field('YPSF_IMAGE')[ind2]
        self['erra_psf'][ind1] = psfcat[1].data.field('ERRAPSF_IMAGE')[ind2]
        self['errb_psf'][ind1] = psfcat[1].data.field('ERRBPSF_IMAGE')[ind2]
        self['errtheta_psf'][ind1] = \
                psfcat[1].data.field('ERRTHETAPSF_IMAGE')[ind2]
        self['x_source'][ind1] = psfcat[1].data.field('XPSF_IMAGE')[ind2]
        self['y_source'][ind1] = psfcat[1].data.field('YPSF_IMAGE')[ind2]
        self['erra_source'][ind1] = \
                psfcat[1].data.field('ERRAPSF_IMAGE')[ind2]
        self['errb_source'][ind1] = \
                psfcat[1].data.field('ERRBPSF_IMAGE')[ind2]
        self['errtheta_source'][ind1] = \
                psfcat[1].data.field('ERRTHETAPSF_IMAGE')[ind2]
        self['flag_usepsf'][ind1] = 1
        return num_psf_sources

    def add_matches(self, association: 'Association',
                    refs: 'ReferenceStars') -> None:
        """
        Copy the matches of a crossmatch and the catalogue values of the
        matched stars (G, BP and RP magnitudes and BP-RP) to the source
        columns, and the number of neighbours of each source. Values of
        sources that are not matched are left as they are.

        """

        matches = association.matches
        ind_plate = matches['source_index']
        ind_gaia = refs.index[matches['ref_index']]
        catalog = refs.catalog
        self['solution_num'][ind_plate] = matches['solution_num']
        self['match_radius'][ind_plate] = matches['match_radius']
        self['gaiadr4_id'][ind_plate] = matches['gaiadr4_id']
        self['gaiadr4_gmag'][ind_plate] = catalog['mag'][ind_gaia]
        self['gaiadr4_bpmag'][ind_plate] = catalog['mag1'][ind_gaia]
        self['gaiadr4_rpmag'][ind_plate] = catalog['mag2'][ind_gaia]
        self['gaiadr4_bp_rp'][ind_plate] = catalog['color_index'][ind_gaia]
        self['gaiadr4_dist'][ind_plate] = matches['gaiadr4_dist']

        # Mask nan values in listed columns
        for col in ['gaiadr4_gmag', 'gaiadr4_bpmag', 'gaiadr4_rpmag',
                    'gaiadr4_bp_rp', 'gaiadr4_dist']:
            self[col] = MaskedColumn(self[col], mask=np.isnan(self[col]))

        # Mask zeros in the ID column
        col = 'gaiadr4_id'
        self[col] = MaskedColumn(self[col], mask=(self[col] == 0))

        # Calculate neighbor counts
        neighbors = association.neighbors

        if len(neighbors) > 0:
            source_num, cnt = np.unique(neighbors['source_num'].data,
                                        return_counts=True)
            mask = np.isin(self['source_num'], source_num)
            ind_mask = np.where(mask)[0]
            self['gaiadr4_neighbors'][ind_mask] = cnt

    def process_coordinates(self, solution_set=None):
        """
        Calculate HEALPix numbers, (X, Y, Z) on the unit sphere, nearest
        neighbor distance, zenith angle, air mass.

        Parameters:
        -----------
        solution_set : :class:`solve.SolutionSet`
            Plate solution with one or more astrometric solutions

        """

        self.log.write('Processing source coordinates', level=3, event=60)

        if solution_set is None or solution_set.num_solutions == 0:
            self.log.write('Cannot process source coordinates '
                           'due to missing astrometric solutions!',
                           level=2, event=60)
            return

        # Loop over solutions and transform image coordinates to RA and Dec
        for i,solution in enumerate(solution_set.solutions):
            w = wcs.WCS(solution.header_wcs)

            # If there is only one solution, then transform coordinates of
            # all sources
            if solution_set.num_solutions == 1:
                m = np.isfinite(self['x_source'])
            else:
                m = self['solution_num'] == i + 1

            if m.sum() > 0:
                ra, dec = w.all_pix2world(self['x_source'][m],
                                          self['y_source'][m], 1)
                self['ra_icrs'][m] = ra
                self['dec_icrs'][m] = dec

                # Assign astrometric errors
                if (solution.scamp_sigma_1 is not None
                    and solution.scamp_sigma_2 is not None):
                    self['ra_error'][m] = solution.scamp_sigma_1
                    self['dec_error'][m] = solution.scamp_sigma_2

        # Check if we have any usable coordinates
        bool_finite = (np.isfinite(self['ra_icrs']) &
                       np.isfinite(self['dec_icrs']))
        num_finite = bool_finite.sum()

        if num_finite == 0:
            self.log.write('No sources with usable coordinates!',
                           level=2, event=60)
            return

        ind_finite = np.where(bool_finite)[0]
        ra_finite = self['ra_icrs'][ind_finite]
        dec_finite = self['dec_icrs'][ind_finite]

        # Calculate X, Y, and Z on the unit sphere
        # http://www.sdss3.org/svn/repo/idlutils/tags/v5_5_5/pro/coord/angles_to_xyz.pro
        phi_rad = np.radians(self['ra_icrs'])
        theta_rad = np.radians(90. - self['dec_icrs'])
        self['x_sphere'] = np.cos(phi_rad) * np.sin(theta_rad)
        self['y_sphere'] = np.sin(phi_rad) * np.sin(theta_rad)
        self['z_sphere'] = np.cos(theta_rad)

        if have_healpy:
            phi_rad = np.radians(self['ra_icrs'][ind_finite])
            theta_rad = np.radians(90. - self['dec_icrs'][ind_finite])
            hp256 = healpy.ang2pix(256, theta_rad, phi_rad, nest=True)
            self['healpix256'][ind_finite] = hp256.astype(np.int32)
            hp1024 = healpy.ang2pix(1024, theta_rad, phi_rad, nest=True)
            self['healpix1024'][ind_finite] = hp1024.astype(np.int32)

            # Loop over solutions and calculate healpix statistics for each
            # solution
            for i,solution in enumerate(solution_set.solutions):
                # Find all healpixes inside solution corners
                lat = solution.skycoord_corners.dec.deg
                lon = solution.skycoord_corners.ra.deg
                vertices = healpy.pixelfunc.ang2vec(lon, lat, lonlat=True)
                pix_inside = healpy.query_polygon(1024, vertices,
                                                  inclusive=False, nest=True)
                tab_inside = Table()
                tab_inside['healpix1024'] = pix_inside
                tab_inside['num_sources'] = 0

                # Select sources that belong to the solution
                m = self['solution_num'] == i + 1

                # Find all healpixes that have sources
                if m.sum() > 0:
                    grp = self[m].group_by('healpix1024').groups
                    tab_grp = Table(grp.aggregate(len)['healpix1024',
                                                       'source_num'])
                    tab_grp.rename_column('source_num', 'num_sources')

                    tab_stack = vstack([tab_grp, tab_inside])
                    grp_stack = tab_stack.group_by('healpix1024').groups
                    tab_sum = grp_stack.aggregate(np.sum)
                else:
                    tab_sum = tab_inside

                solution.healpix_table = tab_sum

        # Create SkyCoord from RA and Dec
        coords = SkyCoord(ra_finite, dec_finite, unit=(u.deg, u.deg))

        # Find nearest neighbours
        if num_finite > 1:
            _, ds2d, _ = match_coordinates_sky(coords, coords, nthneighbor=2)
            matchdist = ds2d.to(u.arcsec).value
            self['nn_dist'][ind_finite] = matchdist.astype(np.float32)

        # Calculate Galactic and ecliptic coordinates
        coords_gal = coords.transform_to(Galactic)
        self['gal_lon'][ind_finite] = coords_gal.l.deg
        self['gal_lat'][ind_finite] = coords_gal.b.deg
        coords_ecl = coords.transform_to(GeocentricMeanEcliptic)
        self['ecl_lon'][ind_finite] = coords_ecl.lon.deg
        self['ecl_lat'][ind_finite] = coords_ecl.lat.deg

        # Suppress ERFA warnings
        warnings.filterwarnings('ignore', message='Tried to get polar motions')
        warnings.filterwarnings('ignore', message='ERFA function')

        # Calculate zenith angle and air mass for each source
        # Check for location and single exposure
        if (self.plate and 
                self.plate.site_latitude and 
                self.plate.site_longitude and 
                (self.plate.numexp == 1) and
                self.plate.exposures and
                self.plate.exposures[0].ut_mid):
            self.log.write('Calculating zenith angle and air mass for sources', 
                           level=3, event=61)
            lon = self.plate.site_longitude
            lat = self.plate.site_latitude
            height = 0.

            if self.plate.site_elevation:
                height = self.plate.site_elevation

            loc = EarthLocation.from_geodetic(lon, lat, height)
            date_avg = Time(self.plate.exposures[0].ut_mid, 
                            format='isot', scale='ut1')
            c_altaz = coords.transform_to(AltAz(obstime=date_avg, location=loc))
            self['zenith_angle'][ind_finite] = c_altaz.zen.deg
            coszt = np.cos(c_altaz.zen)
            airmass = ((1.002432 * coszt**2 + 0.148386 * coszt + 0.0096467) 
                       / (coszt**3 + 0.149864 * coszt**2 + 0.0102963 * coszt 
                          + 0.000303978))
            self['airmass'][ind_finite] = airmass

        # Restore ERFA warnings
        warnings.filterwarnings('default', message='Tried to get polar motions')
        warnings.filterwarnings('default', message='ERFA function')

    def output_csv(self, filename):
        """
        Write extracted sources to a CSV file.

        """

        outfields = list(_source_meta)
        outfmt = [_source_meta[f][1] for f in outfields]
        outhdr = ','.join(outfields)
        delimiter = ','

        np.savetxt(filename, self[outfields], fmt=outfmt,
                   delimiter=delimiter, header=outhdr, comments='')
