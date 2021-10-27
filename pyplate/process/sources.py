import os
import glob
import shutil
import sys
import math
import datetime as dt
import subprocess as sp
import numpy as np
import warnings
import xml.etree.ElementTree as ET
from astropy import __version__ as astropy_version
from astropy import wcs
from astropy.io import fits, votable
from astropy.table import Table, Column, MaskedColumn, vstack, join
from astropy.coordinates import Angle, EarthLocation, SkyCoord, ICRS, AltAz
from astropy.coordinates import Galactic, GeocentricMeanEcliptic
from astropy.coordinates import match_coordinates_sky
from astropy import units as u
from astropy.time import Time
from astropy.stats import sigma_clip
from scipy.interpolate import InterpolatedUnivariateSpline, SmoothBivariateSpline
from scipy.ndimage.filters import generic_filter
from scipy.linalg import lstsq
from collections import OrderedDict
from ..database.database import PlateDB
from ..conf import read_conf
from .._version import __version__

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

try:
    from scipy.spatial import cKDTree as KDT
except ImportError:
    from scipy.spatial import KDTree as KDT

try:
    from sklearn.cluster import DBSCAN
    have_sklearn = True
except ImportError:
    have_sklearn = False

try:
    import MySQLdb
except ImportError:
    pass

try:
    import healpy
    have_healpy = True
except ImportError:
    have_healpy = False

try:
    import statsmodels.api as sm
    have_statsmodels = True
except ImportError:
    have_statsmodels = False


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
    ('gaiaedr3_id',         ('i8', '%d', '')),
    ('gaiaedr3_gmag',       ('f4', '%7.4f', '')),
    ('gaiaedr3_bpmag',      ('f4', '%7.4f', '')),
    ('gaiaedr3_rpmag',      ('f4', '%7.4f', '')),
    ('gaiaedr3_bp_rp',      ('f4', '%7.4f', '')),
    ('gaiaedr3_dist',       ('f4', '%6.3f', '')),
    ('gaiaedr3_neighbors',  ('i4', '%3d', ''))
])


def crossmatch_cartesian(coords_image, coords_ref, tolerance=None):
    """
    Crossmatch source coordinates with reference-star coordinates.

    Parameters
    ----------
    coords_image : array-like
        The coordinates of points to match
    coords_ref : array-like
        The coordinates of reference points to match
    tolerance : float
        Crossmatch distance in pixels (default: 5)

    """

    if tolerance is None:
        tolerance = 5.

    try:
        kdt = KDT(coords_ref, balanced_tree=False, compact_nodes=False)
    except TypeError:
        kdt = KDT(coords_ref)

    ds,ind_ref = kdt.query(coords_image, k=1, distance_upper_bound=tolerance)
    mask_xmatch = ds < tolerance
    ind_image = np.arange(len(coords_image))

    return ind_image[mask_xmatch], ind_ref[mask_xmatch], ds[mask_xmatch]


class SourceTable(Table):
    """
    Source table class

    """

    def __init__(self, *args, **kwargs):
        num_sources = kwargs.pop('num_sources', None)
        super().__init__(*args, **kwargs)

        #self.filename = os.path.basename(filename)
        #self.archive_id = archive_id
        self.basefn = ''
        self.fn_fits = ''
        self.process_id = None
        self.scan_id = None
        self.plate_id = None
        self.log = None

        self.work_dir = ''
        self.scratch_dir = None
        self.write_source_dir = ''
        self.write_db_source_dir = ''
        self.write_db_source_calib_dir = ''

        self.plate_epoch = 1950
        self.plate_year = int(self.plate_epoch)
        self.threshold_sigma = 4.
        self.use_filter = False
        self.filter_path = None
        self.use_psf = False
        self.psf_threshold_sigma = 20.
        self.psf_model_sigma = 20.
        self.min_model_sources = 100
        self.max_model_sources = 10000
        self.sip = 3
        self.skip_bright = 10
        self.distort = 3
        self.subfield_distort = 1
        self.max_recursion_depth = 5
        self.force_recursion_depth = 0
        self.circular_film = False
        self.crossmatch_radius = None
        self.crossmatch_nsigma = 10.
        self.crossmatch_nlogarea = 2.
        self.crossmatch_maxradius = 20.

        self.plate_header = None
        self.platemeta = None
        self.imwidth = None
        self.imheight = None
        self.plate_solved = False
        self.mean_pixscale = None
        self.num_sources = num_sources
        self.num_sources_sixbins = None
        self.rel_area_sixbins = None
        self.min_ra = None
        self.max_ra = None
        self.min_dec = None
        self.max_dec = None
        self.ncp_close = None
        self.scp_close = None
        self.ncp_on_plate = None
        self.scp_on_plate = None
        self.pattern_x = None
        self.pattern_y = None
        self.pattern_ratio = None
        self.num_crossmatch_gaia = None
        self.neighbors_gaia = None

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
        self['gaiaedr3_gmag'] = np.nan
        self['gaiaedr3_bpmag'] = np.nan
        self['gaiaedr3_rpmag'] = np.nan
        self['gaiaedr3_bp_rp'] = np.nan
        self['gaiaedr3_dist'] = np.nan
        self['gaiaedr3_neighbors'] = 0
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

    def assign_conf(self, conf):
        """
        Parse configuration and set class attributes.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)
            
        self.conf = conf

        try:
            self.archive_id = conf.getint('Archive', 'archive_id')
        except ValueError:
            print('Error in configuration file '
                  '([{}], {})'.format('Archive', attr))
        except configparser.Error:
            pass

        for attr in ['sextractor_path', 'scamp_path', 'psfex_path',
                     'solve_field_path', 'wcs_to_tan_path']:
            try:
                setattr(self, attr, conf.get('Programs', attr))
            except configparser.Error:
                pass

        for attr in ['fits_dir', 'index_dir', 'gaia_dir', 'tycho2_dir', 
                     'work_dir', 'write_log_dir', 'write_phot_dir',
                     'write_source_dir', 'write_wcs_dir',
                     'write_db_source_dir', 'write_db_source_calib_dir']:
            try:
                setattr(self, attr, conf.get('Files', attr))
            except configparser.Error:
                pass

        if self.write_log_dir:
            self.enable_log = True

        for attr in ['use_gaia_fits', 'use_tycho2_fits', 
                     'use_ucac4_db', 'use_apass_db',
                     'enable_db_log', 'write_sources_csv']:
            try:
                setattr(self, attr, conf.getboolean('Database', attr))
            except ValueError:
                print('Error in configuration file '
                      '([{}], {})'.format('Database', attr))
            except configparser.Error:
                pass

        for attr in ['ucac4_db_host', 'ucac4_db_user', 'ucac4_db_name', 
                     'ucac4_db_passwd', 'ucac4_db_table',
                     'apass_db_host', 'apass_db_user', 'apass_db_name', 
                     'apass_db_passwd', 'apass_db_table',
                     'output_db_host', 'output_db_user',
                     'output_db_name', 'output_db_passwd']:
            try:
                setattr(self, attr, conf.get('Database', attr))
            except configparser.Error:
                pass

        for attr in ['use_filter', 'use_psf', 'circular_film']:
            try:
                setattr(self, attr, conf.getboolean('Solve', attr))
            except ValueError:
                print('Error in configuration file '
                      '([{}], {})'.format('Solve', attr))
            except configparser.Error:
                pass

        for attr in ['plate_epoch', 'threshold_sigma', 
                     'psf_threshold_sigma', 'psf_model_sigma', 
                     'crossmatch_radius', 'crossmatch_nsigma', 
                     'crossmatch_nlogarea', 'crossmatch_maxradius']:
            try:
                setattr(self, attr, conf.getfloat('Solve', attr))
            except ValueError:
                print('Error in configuration file '
                      '([{}], {})'.format('Solve', attr))
            except configparser.Error:
                pass

        for attr in ['sip', 'skip_bright', 'distort', 'subfield_distort', 
                     'max_recursion_depth', 'force_recursion_depth', 
                     'min_model_sources', 'max_model_sources']:
            try:
                setattr(self, attr, conf.getint('Solve', attr))
            except ValueError:
                print('Error in configuration file '
                      '([{}], {})'.format('Solve', attr))
            except configparser.Error:
                pass

        for attr in ['filter_path', 'astref_catalog', 'photref_catalog']:
            try:
                setattr(self, attr, conf.get('Solve', attr))
            except configparser.Error:
                pass

        # Read UCAC4 and APASS table column names from the dedicated sections,
        # named after the tables
        if conf.has_section(self.ucac4_db_table):
            for attr in self.ucac4_columns.keys():
                try:
                    colstr = conf.get(self.ucac4_db_table, attr)
                    _,typ = self.ucac4_columns[attr]
                    self.ucac4_columns[attr] = (colstr, typ)
                except configparser.Error:
                    pass

        if conf.has_section(self.apass_db_table):
            for attr in self.apass_columns.keys():
                try:
                    colstr = conf.get(self.apass_db_table, attr)
                    _,typ = self.apass_columns[attr]
                    self.apass_columns[attr] = (colstr, typ)
                except configparser.Error:
                    pass

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

    def apply_scanner_pattern(self, plate_solution=None):
        """
        Correct source coordinates for scanner pattern.

        Parameters
        ----------
        plate_solution: PlateSolution instance

        """

        from .solve import PlateSolution
        assert isinstance(plate_solution, PlateSolution)
        assert plate_solution.pattern_ratio is not None

        if plate_solution.pattern_ratio > 1.5:
            y_source = self['y_source']
            self['y_source'] = (y_source - plate_solution.pattern_y(y_source))
        elif plate_solution.pattern_ratio < 2./3.:
            x_source = self['x_source']
            self['x_source'] = (x_source - plate_solution.pattern_x(x_source))

    def crossmatch_gaia(self, plate_solution=None, star_catalog=None):
        """
        Crossmatch sources with Gaia objects, considering multiple solutions.

        Parameters:
        -----------
        plate_solution : :class:`solve.PlateSolution`
            Plate solution with one or more astrometric solutions
        star_catalog : :class:`catalog.StarCatalog`
            External star catalog with Gaia data

        """

        from .solve import PlateSolution
        from .catalog import StarCatalog

        self.log.write('Crossmatching sources with Gaia objects',
                       level=3, event=44)

        if plate_solution is None or plate_solution.num_solutions == 0:
            self.log.write('Cannot crossmatch sources with Gaia objects '
                           'due to missing astrometric solutions!',
                           level=2, event=44)
            return

        if star_catalog is None:
            self.log.write('Cannot crossmatch sources with Gaia objects '
                           'due to missing Gaia catalog data!',
                           level=2, event=44)
            return

        assert isinstance(plate_solution, PlateSolution)
        assert isinstance(star_catalog, StarCatalog)

        # Take parameters from plate_solution
        num_solutions = plate_solution.num_solutions
        solutions = plate_solution.solutions
        mean_pixscale = plate_solution.mean_pixel_scale

        # Number of Gaia stars
        num_gaia = len(star_catalog)

        self.log.write('Number of Gaia stars: {:d}'.format(num_gaia), 
                       level=4, event=44, double_newline=False)

        # Calculate RA and Dec for the plate epoch
        ra_ref = (star_catalog['ra']
                  + (self.plate_epoch - star_catalog['ref_epoch'])
                  * star_catalog['pmra']
                  / np.cos(star_catalog['dec'] * np.pi / 180.) / 3600000.)
        dec_ref = (star_catalog['dec']
                   + (self.plate_epoch - star_catalog['ref_epoch'])
                   * star_catalog['pmdec'] / 3600000.)
        #catalog = SkyCoord(ra_ref, dec_ref, frame='icrs')
        xy_ref = np.empty((0, 2))
        sol_ref = np.empty((0,), dtype=np.int8)
        index_ref = np.empty((0,), dtype=np.int32)

        # Build a list of Gaia stars in image coordinates
        for i in np.arange(plate_solution.num_solutions):
            solution = solutions[i]

            # If there is a column named 'solution_num', then take only
            # reference stars with the current solution number
            if 'solution_num' in star_catalog.columns:
                mask_sol = star_catalog['solution_num'] == i + 1
            else:
                mask_sol = np.full(num_gaia, True)

            w = wcs.WCS(solution['header_wcs'])

            try:
                xr,yr = w.all_world2pix(ra_ref[mask_sol], dec_ref[mask_sol], 1)
            except wcs.NoConvergence as e:
                self.log.write('Failed to convert sky coordinates to '
                               'pixel coordinates for solution {:d}: {}'
                               .format(i + 1, e))
                continue

            mask_inside = ((xr > 0.5) & (xr < plate_solution.imwidth) &
                           (yr > 0.5) & (yr < plate_solution.imheight))
            num_inside = mask_inside.sum()
            xyr = np.vstack((xr[mask_inside], yr[mask_inside])).T
            xy_ref = np.vstack((xy_ref, xyr))
            sol_ref = np.hstack((sol_ref, np.full(num_inside, i + 1)))
            index_ref = np.hstack((index_ref,
                                   np.arange(num_gaia)[mask_sol][mask_inside]))

        # Calculate mean astrometric error
        sigma1 = u.Quantity([sol['scamp_sigma_1'] for sol in solutions
                             if sol['scamp_sigma_1'] is not None])
        sigma2 = u.Quantity([sol['scamp_sigma_2'] for sol in solutions
                             if sol['scamp_sigma_2'] is not None])

        if len(sigma1) > 0 and len(sigma2) > 0:
            mean_scamp_sigma = np.sqrt(sigma1.mean()**2 + sigma2.mean()**2)
        else:
            mean_scamp_sigma = 2. * u.arcsec

        # Crossmatch sources and Gaia stars
        coords_plate = np.vstack((self['x_source'], self['y_source'])).T
        tolerance = ((5. * mean_scamp_sigma / mean_pixscale)
                     .to(u.pixel).value)

        #if (5. * mean_scamp_sigma) < 2 * u.arcsec:
        #    tolerance = ((2 * u.arcsec / mean_pixscale)
        #                 .to(u.pixel).value)

        tolerance_arcsec = (5. * mean_scamp_sigma).to(u.arcsec).value
        self.log.write('Crossmatch tolerance: {:.2f} arcsec ({:.2f} pixels)'
                       .format(tolerance_arcsec, tolerance), level=4, event=44,
                       double_newline=False)

        ind_plate, ind_ref, ds = crossmatch_cartesian(coords_plate, xy_ref, 
                                                      tolerance=tolerance)
        dist_arcsec = (ds * u.pixel * mean_pixscale).to(u.arcsec).value
        ind_gaia = index_ref[ind_ref]
        self['solution_num'][ind_plate] = sol_ref[ind_ref]
        self['match_radius'][ind_plate] = tolerance_arcsec
        self['gaiaedr3_id'][ind_plate] = star_catalog['source_id'][ind_gaia]
        self['gaiaedr3_gmag'][ind_plate] = star_catalog['mag'][ind_gaia]
        self['gaiaedr3_bpmag'][ind_plate] = star_catalog['mag1'][ind_gaia]
        self['gaiaedr3_rpmag'][ind_plate] = star_catalog['mag2'][ind_gaia]
        self['gaiaedr3_bp_rp'][ind_plate] = star_catalog['color_index'][ind_gaia]
        self['gaiaedr3_dist'][ind_plate] = dist_arcsec
        self.num_crossmatch_gaia = len(ind_plate)

        # Mask nan values in listed columns
        for col in ['gaiaedr3_gmag', 'gaiaedr3_bpmag', 'gaiaedr3_rpmag',
                    'gaiaedr3_bp_rp', 'gaiaedr3_dist']:
            self[col] = MaskedColumn(self[col], mask=np.isnan(self[col]))

        # Mask zeros in the ID column
        col = 'gaiaedr3_id'
        self[col] = MaskedColumn(self[col], mask=(self[col] == 0))

        # Store number of crossmatched sources to each solution
        grp = self.group_by('solution_num').groups
        tab_grp = Table(grp.aggregate(len)['solution_num', 'source_num'])
        tab_grp.rename_column('source_num', 'num_gaia_edr3')

        for i in np.arange(plate_solution.num_solutions):
            solution = solutions[i]
            m = tab_grp['solution_num'] == i + 1

            if m.sum() > 0:
                num_gaia_edr3 = tab_grp['num_gaia_edr3'][m].data[0]
                solution['num_gaia_edr3'] = num_gaia_edr3
            else:
                solution['num_gaia_edr3'] = 0

        # Crossmatch: find all neighbours for sources
        kdt_ref = KDT(xy_ref)
        kdt_plate = KDT(coords_plate)
        max_distance = ((20. * mean_scamp_sigma / mean_pixscale)
                        .to(u.pixel).value)

        if (20. * mean_scamp_sigma) < 5 * u.arcsec:
            max_distance = (5 * u.arcsec / mean_pixscale).to(u.pixel).value

        max_dist_arcsec = (max_distance * u.pixel * mean_pixscale).to(u.arcsec).value
        self.log.write('Finding all reference stars around sources within '
                       'the radius of {:.2f} arcsec ({:.2f} pixels)'
                       .format(max_dist_arcsec, max_distance),
                       level=4, event=44)

        mtrx = kdt_plate.sparse_distance_matrix(kdt_ref, max_distance)
        mtrx_keys = np.array([a for a in mtrx.keys()])

        # Check if there are neighbors at all
        if len(mtrx_keys) > 0:
            k_plate = mtrx_keys[:,0]
            k_ref = mtrx_keys[:,1]
            dist = np.fromiter(mtrx.values(), dtype=float) * u.pixel

            # Construct neighbors table
            nbs = Table()
            nbs['source_num'] = self['source_num'][k_plate]
            nbs['gaiaedr3_id'] = star_catalog['source_id'][index_ref[k_ref]]
            nbs['dist'] = dist
            nbs['solution_num'] = sol_ref[k_ref]
            nbs['x_gaia'] = xy_ref[k_ref,0]
            nbs['y_gaia'] = xy_ref[k_ref,1]

            # Create the flag_xmatch column by joining the neighbors table
            # with the source table
            tab = Table()
            tab['source_num'] = self['source_num']
            tab['gaiaedr3_id'] = MaskedColumn(self['gaiaedr3_id']).filled(0)
            tab['flag_xmatch'] = np.int8(1)
            jtab = join(nbs, tab, keys=('source_num', 'gaiaedr3_id'),
                        join_type='left')
            jtab['flag_xmatch'] = MaskedColumn(jtab['flag_xmatch']).filled(0)
            self.neighbors_gaia = jtab

            # Calculate neighbor counts
            source_num, cnt = np.unique(nbs['source_num'].data, return_counts=True)
            mask = np.isin(self['source_num'], source_num)
            ind_mask = np.where(mask)[0]
            self['gaiaedr3_neighbors'][ind_mask] = cnt
        else:
            # Create empty neighbors table
            nbs = Table(names=('source_num', 'gaiaedr3_id', 'dist',
                               'solution_num', 'x_gaia', 'y_gaia',
                               'flag_xmatch'),
                        dtype=('i4', 'i8', 'f4', 'i2', 'f8', 'f8', 'i1'))
            self.neighbors_gaia = nbs

        # Process coordinates again, because solution_num assignments may have changed
        self.process_coordinates(plate_solution=plate_solution)

    def process_coordinates(self, plate_solution=None):
        """
        Calculate HEALPix numbers, (X, Y, Z) on the unit sphere, nearest
        neighbor distance, zenith angle, air mass.

        Parameters:
        -----------
        plate_solution : :class:`solve.PlateSolution`
            Plate solution with one or more astrometric solutions

        """

        self.log.write('Processing source coordinates', level=3, event=60)

        if plate_solution is None or plate_solution.num_solutions == 0:
            self.log.write('Cannot process source coordinates '
                           'due to missing astrometric solutions!',
                           level=2, event=60)
            return

        # Loop over solutions and transform image coordinates to RA and Dec
        for i,solution in enumerate(plate_solution.solutions):
            w = wcs.WCS(solution['header_wcs'])

            # If there is only one solution, then transform coordinates of
            # all sources
            if plate_solution.num_solutions == 1:
                m = np.isfinite(self['x_source'])
            else:
                m = self['solution_num'] == i + 1

            if m.sum() > 0:
                ra, dec = w.all_pix2world(self['x_source'][m],
                                          self['y_source'][m], 1)
                self['ra_icrs'][m] = ra
                self['dec_icrs'][m] = dec

                # Assign astrometric errors
                if (solution['scamp_sigma_1'] is not None
                    and solution['scamp_sigma_2'] is not None):
                    self['ra_error'][m] = solution['scamp_sigma_1']
                    self['dec_error'][m] = solution['scamp_sigma_2']

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
            for i,solution in enumerate(plate_solution.solutions):
                # Find all healpixes inside solution corners
                lat = solution['skycoord_corners'].dec.deg
                lon = solution['skycoord_corners'].ra.deg
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

                solution['healpix_table'] = tab_sum

        # Find nearest neighbours
        coords = SkyCoord(ra_finite, dec_finite, unit=(u.deg, u.deg))
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
        if (self.platemeta and 
                self.platemeta['site_latitude'] and 
                self.platemeta['site_longitude'] and 
                (self.platemeta['numexp'] == 1) and
                self.platemeta['date_avg'] and
                self.platemeta['date_avg'][0]):
            self.log.write('Calculating zenith angle and air mass for sources', 
                           level=3, event=61)
            lon = self.platemeta['site_longitude']
            lat = self.platemeta['site_latitude']
            height = 0.

            if self.platemeta['site_elevation']:
                height = self.platemeta['site_elevation']

            loc = EarthLocation.from_geodetic(lon, lat, height)
            date_avg = Time(self.platemeta['date_avg'][0], 
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
