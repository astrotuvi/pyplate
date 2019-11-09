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
from astropy.table import Table, Column, vstack
from astropy.coordinates import Angle, EarthLocation, SkyCoord, ICRS, AltAz
from astropy.coordinates import match_coordinates_sky
from astropy import units as u
from astropy.time import Time
from astropy.stats import sigma_clip
from scipy.interpolate import InterpolatedUnivariateSpline, SmoothBivariateSpline
from scipy.ndimage.filters import generic_filter
from scipy.linalg import lstsq
from collections import OrderedDict
from .database import PlateDB
from .conf import read_conf
from ._version import __version__

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
    ('solution_num',        ('i2', '%1d', '')),
    ('raj2000',             ('f8', '%11.7f', '')),
    ('dej2000',             ('f8', '%11.7f', '')),
    ('x_sphere',            ('f8', '%10.7f', '')),
    ('y_sphere',            ('f8', '%10.7f', '')),
    ('z_sphere',            ('f8', '%10.7f', '')),
    ('healpix256',          ('i4', '%6d', '')),
    ('raj2000_wcs',         ('f8', '%11.7f', '')),
    ('dej2000_wcs',         ('f8', '%11.7f', '')),
    ('raj2000_sub',         ('f8', '%11.7f', '')),
    ('dej2000_sub',         ('f8', '%11.7f', '')),
    ('raerr_sub',           ('f4', '%7.4f', '')),
    ('decerr_sub',          ('f4', '%7.4f', '')),
    ('astrom_sub_grid',     ('i2', '%3d', '')),
    ('astrom_sub_id',       ('i4', '%5d', '')),
    ('nn_dist',             ('f4', '%6.3f', '')),
    ('zenith_angle',        ('f4', '%7.4f', '')),
    ('airmass',             ('f4', '%7.4f', '')),
    ('natmag',              ('f4', '%7.4f', '')),
    ('natmagerr',           ('f4', '%7.4f', '')),
    ('bmag',                ('f4', '%7.4f', '')),
    ('bmagerr',             ('f4', '%7.4f', '')),
    ('vmag',                ('f4', '%7.4f', '')),
    ('vmagerr',             ('f4', '%7.4f', '')),
    ('natmag_plate',        ('f4', '%7.4f', '')),
    ('natmagerr_plate',     ('f4', '%7.4f', '')),
    ('phot_plate_flags',    ('i2', '%1d', '')),
    ('natmag_correction',   ('f4', '%7.4f', '')),
    ('natmag_sub',          ('f4', '%7.4f', '')),
    ('natmagerr_sub',       ('f4', '%7.4f', '')),
    ('natmag_residual',     ('f4', '%7.4f', '')),
    ('phot_sub_grid',       ('i2', '%3d', '')),
    ('phot_sub_id',         ('i4', '%5d', '')),
    ('phot_sub_flags',      ('i2', '%1d', '')),
    ('phot_calib_flags',    ('i2', '%1d', '')),
    ('color_term',          ('f4', '%7.4f', '')),
    ('color_bv',            ('f4', '%7.4f', '')),
    ('color_bv_err',        ('f4', '%7.4f', '')),
    ('cat_natmag',          ('f4', '%7.4f', '')),
    ('match_radius',        ('f4', '%7.3f', '')),
    ('gaiadr2_id',          ('i8', '%d', '')),
    ('gaiadr2_ra',          ('f8', '%11.7f', '')),
    ('gaiadr2_dec',         ('f8', '%11.7f', '')),
    ('gaiadr2_gmag',        ('f4', '%7.4f', '')),
    ('gaiadr2_bpmag',       ('f4', '%7.4f', '')),
    ('gaiadr2_rpmag',       ('f4', '%7.4f', '')),
    ('gaiadr2_bp_rp',       ('f4', '%7.4f', '')),
    ('gaiadr2_dist',        ('f4', '%6.3f', '')),
    ('gaiadr2_neighbors',   ('i4', '%3d', '')),
    ('ucac4_id',            ('a10', '%s', '')),
    ('ucac4_ra',            ('f8', '%11.7f', '')),
    ('ucac4_dec',           ('f8', '%11.7f', '')),
    ('ucac4_bmag',          ('f4', '%7.4f', '')),
    ('ucac4_vmag',          ('f4', '%7.4f', '')),
    ('ucac4_bmagerr',       ('f4', '%6.4f', '')),
    ('ucac4_vmagerr',       ('f4', '%6.4f', '')),
    ('ucac4_dist',          ('f4', '%6.3f', '')),
    ('ucac4_dist2',         ('f4', '%7.3f', '')),
    ('ucac4_nn_dist',       ('f4', '%7.3f', '')),
    ('tycho2_id',           ('a12', '%s', '')),
    ('tycho2_id_pad',       ('a12', '%s', '')),
    ('tycho2_ra',           ('f8', '%11.7f', '')),
    ('tycho2_dec',          ('f8', '%11.7f', '')),
    ('tycho2_btmag',        ('f4', '%7.4f', '')),
    ('tycho2_vtmag',        ('f4', '%7.4f', '')),
    ('tycho2_btmagerr',     ('f4', '%6.4f', '')),
    ('tycho2_vtmagerr',     ('f4', '%6.4f', '')),
    ('tycho2_hip',          ('i4', '%6d', '')),
    ('tycho2_dist',         ('f4', '%6.3f', '')),
    ('tycho2_dist2',        ('f4', '%7.3f', '')),
    ('tycho2_nn_dist',      ('f4', '%7.3f', '')),
    ('apass_ra',            ('f8', '%11.7f', '')),
    ('apass_dec',           ('f8', '%11.7f', '')),
    ('apass_bmag',          ('f4', '%7.4f', '')),
    ('apass_vmag',          ('f4', '%7.4f', '')),
    ('apass_bmagerr',       ('f4', '%6.4f', '')),
    ('apass_vmagerr',       ('f4', '%6.4f', '')),
    ('apass_dist',          ('f4', '%6.3f', '')),
    ('apass_dist2',         ('f4', '%7.3f', '')),
    ('apass_nn_dist',       ('f4', '%7.3f', ''))
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

    kdt = KDT(coords_ref)
    ds,ind_ref = kdt.query(coords_image, k=1)
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
        self['raj2000'] = np.nan
        self['dej2000'] = np.nan
        self['raj2000_wcs'] = np.nan
        self['dej2000_wcs'] = np.nan
        self['raj2000_sub'] = np.nan
        self['dej2000_sub'] = np.nan
        self['raerr_sub'] = np.nan
        self['decerr_sub'] = np.nan
        self['x_sphere'] = np.nan
        self['y_sphere'] = np.nan
        self['z_sphere'] = np.nan
        self['healpix256'] = -1
        self['nn_dist'] = np.nan
        self['zenith_angle'] = np.nan
        self['airmass'] = np.nan
        self['gaiadr2_ra'] = np.nan
        self['gaiadr2_dec'] = np.nan
        self['gaiadr2_bpmag'] = np.nan
        self['gaiadr2_rpmag'] = np.nan
        self['gaiadr2_bp_rp'] = np.nan
        self['gaiadr2_dist'] = np.nan
        self['gaiadr2_neighbors'] = 0
        self['ucac4_ra'] = np.nan
        self['ucac4_dec'] = np.nan
        self['ucac4_bmag'] = np.nan
        self['ucac4_vmag'] = np.nan
        self['ucac4_bmagerr'] = np.nan
        self['ucac4_vmagerr'] = np.nan
        self['ucac4_dist'] = np.nan
        self['ucac4_dist2'] = np.nan
        self['ucac4_nn_dist'] = np.nan
        self['tycho2_ra'] = np.nan
        self['tycho2_dec'] = np.nan
        self['tycho2_btmag'] = np.nan
        self['tycho2_vtmag'] = np.nan
        self['tycho2_btmagerr'] = np.nan
        self['tycho2_vtmagerr'] = np.nan
        self['tycho2_dist'] = np.nan
        self['tycho2_dist2'] = np.nan
        self['tycho2_nn_dist'] = np.nan
        self['apass_ra'] = np.nan
        self['apass_dec'] = np.nan
        self['apass_bmag'] = np.nan
        self['apass_vmag'] = np.nan
        self['apass_bmagerr'] = np.nan
        self['apass_vmagerr'] = np.nan
        self['apass_dist'] = np.nan
        self['apass_dist2'] = np.nan
        self['apass_nn_dist'] = np.nan
        self['natmag'] = np.nan
        self['natmagerr'] = np.nan
        self['bmag'] = np.nan
        self['bmagerr'] = np.nan
        self['vmag'] = np.nan
        self['vmagerr'] = np.nan
        self['natmag_plate'] = np.nan
        self['natmagerr_plate'] = np.nan
        self['natmag_residual'] = np.nan
        self['natmag_correction'] = np.nan
        self['natmag_sub'] = np.nan
        self['natmagerr_sub'] = np.nan
        self['color_term'] = np.nan
        self['color_bv'] = np.nan
        self['cat_natmag'] = np.nan
        self['phot_calib_flags'] = 0
        self['phot_plate_flags'] = 0
        self['phot_sub_flags'] = 0

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
            x_image = self['x_source']
            self['x_source'] = (x_source - plate_solution.pattern_x(x_source))

    def crossmatch_gaia(self, plate_solution=None):
        """
        Crossmatch sources with Gaia objects, considering multiple solutions.

        Parameters
        ----------
        plate_solution: PlateSolution instance

        """

        from .solve import PlateSolution
        assert isinstance(plate_solution, PlateSolution)
        assert plate_solution.num_solutions > 0

        # Take parameters from plate_solution
        num_solutions = plate_solution.num_solutions
        solutions = plate_solution.solutions
        mean_pixscale = plate_solution.mean_pixscale

        # Read Gaia sources
        if isinstance(self.gaia_files, str):
            try:
                gaia_table = Table.read(self.gaia_files)
            except Exception:
                self.log.write('Cannot read Gaia catalog file {}'
                               .format(self.gaia_files),
                               level=2, event=0)
                return

        # Number of Gaia stars
        num_gaia = len(gaia_table)

        self.log.write('Number of Gaia stars: {:d}'.format(num_gaia), 
                       level=4, event=0)

        # Calculate RA and Dec for the plate epoch
        ra_ref = (gaia_table['ra'] + (self.plate_epoch - 2015.5) 
                  * gaia_table['pmra']
                  / np.cos(gaia_table['dec'] * np.pi / 180.) / 3600000.)
        dec_ref = (gaia_table['dec'] + (self.plate_epoch - 2015.5) 
                   * gaia_table['pmdec'] / 3600000.)
        #catalog = SkyCoord(ra_ref, dec_ref, frame='icrs')
        xy_ref = np.empty((0, 2))
        sol_ref = np.empty((0,), dtype=np.int8)
        index_ref = np.empty((0,), dtype=np.int32)

        # Build a list of Gaia stars in image coordinates
        for i in np.arange(plate_solution.num_solutions):
            solution = solutions[i]

            w = wcs.WCS(solution['header_wcs'])
            xr,yr = w.all_world2pix(ra_ref, dec_ref, 1)
            mask_inside = ((xr > 0) & (xr < plate_solution.imwidth) & 
                           (yr > 0) & (yr < plate_solution.imheight))
            num_inside = mask_inside.sum()
            xyr = np.vstack((xr[mask_inside], yr[mask_inside])).T
            xy_ref = np.vstack((xy_ref, xyr))
            sol_ref = np.hstack((sol_ref, np.full(num_inside, i+1)))
            index_ref = np.hstack((index_ref, np.arange(num_gaia)[mask_inside]))

        # Calculate mean astrometric error
        sigma1 = u.Quantity([sol['scamp_sigma_1'] for sol in solutions])
        sigma2 = u.Quantity([sol['scamp_sigma_2'] for sol in solutions])
        mean_scamp_sigma = np.sqrt(sigma1.mean()**2 + sigma2.mean()**2)

        # Crossmatch sources and Gaia stars
        coords_plate = np.vstack((self['x_source'], self['y_source'])).T
        tolerance = ((5. * mean_scamp_sigma / mean_pixscale)
                     .to(u.pixel).value)

        #if (5. * mean_scamp_sigma) < 2 * u.arcsec:
        #    tolerance = ((2 * u.arcsec / mean_pixscale)
        #                 .to(u.pixel).value)

        ind_plate, ind_ref, ds = crossmatch_cartesian(coords_plate, xy_ref, 
                                                      tolerance=tolerance)
        dist_arcsec = (ds * u.pixel * mean_pixscale).to(u.arcsec).value
        ind_gaia = index_ref[ind_ref]
        self['solution_num'][ind_plate] = sol_ref[ind_ref]
        self['gaiadr2_id'][ind_plate] = gaia_table['source_id'][ind_gaia]
        self['gaiadr2_ra'][ind_plate] = ra_ref[ind_gaia]
        self['gaiadr2_dec'][ind_plate] = dec_ref[ind_gaia]
        self['gaiadr2_gmag'][ind_plate] = gaia_table['phot_g_mean_mag'][ind_gaia]
        self['gaiadr2_bpmag'][ind_plate] = gaia_table['phot_bp_mean_mag'][ind_gaia]
        self['gaiadr2_rpmag'][ind_plate] = gaia_table['phot_rp_mean_mag'][ind_gaia]
        self['gaiadr2_bp_rp'][ind_plate] = gaia_table['bp_rp'][ind_gaia]
        self['gaiadr2_dist'][ind_plate] = dist_arcsec

        # Crossmatch: find all neighbours for sources
        kdt_ref = KDT(xy_ref)
        kdt_plate = KDT(coords_plate)
        max_distance = ((20. * mean_scamp_sigma / mean_pixscale)
                        .to(u.pixel).value)

        if (20. * mean_scamp_sigma) < 5 * u.arcsec:
            max_distance = (5 * u.arcsec / mean_pixscale).to(u.pixel).value

        mtrx = kdt_plate.sparse_distance_matrix(kdt_ref, max_distance)
        mtrx_keys = np.array([a for a in mtrx.keys()])
        k_plate = mtrx_keys[:,0]
        k_ref = mtrx_keys[:,1]
        dist = np.fromiter(mtrx.values(), dtype=float) * u.pixel

        # Construct neighbors table
        nbs = Table()
        nbs['source_num'] = self['source_num'][k_plate]
        nbs['gaia_id'] = gaia_table['source_id'][index_ref[k_ref]]
        nbs['dist'] = dist
        nbs['solution_num'] = sol_ref[k_ref]
        nbs['gaia_x'] = xy_ref[k_ref,0]
        nbs['gaia_y'] = xy_ref[k_ref,1]
        self.neighbors_gaia = nbs

        # Calculate neighbor counts
        source_num, cnt = np.unique(nbs['source_num'].data, return_counts=True)
        mask = np.isin(self['source_num'], source_num)
        ind_mask = np.where(mask)[0]
        self['gaiadr2_neighbors'][ind_mask] = cnt

    def process_source_coordinates(self):
        """
        Combine coordinates from the global and recursive solutions.
        Calculate X, Y, and Z on the unit sphere.

        """

        self.log.to_db(3, 'Processing source coordinates', event=60)

        self.sources['raj2000'] = self.sources['raj2000_wcs']
        self.sources['dej2000'] = self.sources['dej2000_wcs']

        ind = np.where(np.isfinite(self.sources['raj2000_sub']) &
                       np.isfinite(self.sources['dej2000_sub']))

        if len(ind[0]) > 0:
            self.sources['raj2000'][ind] = self.sources['raj2000_sub'][ind]
            self.sources['dej2000'][ind] = self.sources['dej2000_sub'][ind]

        # Calculate X, Y, and Z on the unit sphere
        # http://www.sdss3.org/svn/repo/idlutils/tags/v5_5_5/pro/coord/angles_to_xyz.pro
        phi_rad = np.radians(self.sources['raj2000'])
        theta_rad = np.radians(90. - self.sources['dej2000'])
        self.sources['x_sphere'] = np.cos(phi_rad) * np.sin(theta_rad)
        self.sources['y_sphere'] = np.sin(phi_rad) * np.sin(theta_rad)
        self.sources['z_sphere'] = np.cos(theta_rad)

        if have_healpy:
            ind = np.where(np.isfinite(self.sources['raj2000']) &
                           np.isfinite(self.sources['dej2000']))

            if len(ind[0]) > 0:
                phi_rad = np.radians(self.sources['raj2000'][ind])
                theta_rad = np.radians(90. - self.sources['dej2000'][ind])
                hp256 = healpy.ang2pix(256, theta_rad, phi_rad, nest=True)
                self.sources['healpix256'][ind] = hp256.astype(np.int32)

        # Prepare for cross-match with the UCAC4, Tycho-2 and APASS catalogues

        bool_finite = (np.isfinite(self.sources['raj2000']) &
                       np.isfinite(self.sources['dej2000']))
        num_finite = bool_finite.sum()

        if num_finite == 0:
            self.log.write('No sources with usable coordinates for '
                           'cross-matching', 
                           level=2, event=60)
            return

        ind_finite = np.where(bool_finite)[0]
        ra_finite = self.sources['raj2000'][ind_finite]
        dec_finite = self.sources['dej2000'][ind_finite]
        coorderr_finite = np.sqrt(self.sources['raerr_sub'][ind_finite]**2 +
                                  self.sources['decerr_sub'][ind_finite]**2)
        logarea_finite = np.log10(self.sources['isoarea'][ind_finite])

        # Assign default coordinate errors to sources with no error estimates 
        # from sub-fields
        bool_nanerr = np.isnan(coorderr_finite)
        num_nanerr = bool_nanerr.sum()

        if num_nanerr > 0:
            coorderr_finite[np.where(bool_nanerr)] = 20.

        # Combine cross-match criteria
        if self.crossmatch_radius is not None:
            self.log.write('Using fixed cross-match radius of {:.2f} arcsec'
                           ''.format(float(self.crossmatch_radius)), 
                           level=4, event=60)
            matchrad_arcsec = float(self.crossmatch_radius)*u.arcsec
        else:
            self.log.write('Using scaled cross-match radius of '
                           '{:.2f} astrometric sigmas'
                           ''.format(float(self.crossmatch_nsigma)), 
                           level=4, event=60)
            matchrad_arcsec = (coorderr_finite * float(self.crossmatch_nsigma)
                               * u.arcsec)

        if self.crossmatch_nlogarea is not None:
            self.log.write('Using scaled cross-match radius of '
                           '{:.2f} times log10(isoarea)'
                           ''.format(float(self.crossmatch_nlogarea)), 
                           level=4, event=60)
            logarea_arcsec = (logarea_finite * u.pixel * self.mean_pixscale
                              * float(self.crossmatch_nlogarea))
            matchrad_arcsec = np.maximum(matchrad_arcsec, logarea_arcsec)

        if self.crossmatch_maxradius is not None:
            self.log.write('Using maximum cross-match radius of {:.2f} arcsec'
                           ''.format(float(self.crossmatch_maxradius)), 
                           level=4, event=60)
            maxradius_arcsec = (float(self.crossmatch_maxradius) 
                                * u.arcsec)
            matchrad_arcsec = np.minimum(matchrad_arcsec, maxradius_arcsec)

        self.sources['match_radius'][ind_finite] = matchrad_arcsec

        # Find nearest neighbours
        coords = SkyCoord(ra_finite, dec_finite, unit=(u.deg, u.deg))
        _, ds2d, _ = match_coordinates_sky(coords, coords, nthneighbor=2)
        matchdist = ds2d.to(u.arcsec).value
        self.sources['nn_dist'][ind_finite] = matchdist.astype(np.float32)

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
            self.sources['zenith_angle'] = c_altaz.zen.deg
            coszt = np.cos(c_altaz.zen)
            airmass = ((1.002432 * coszt**2 + 0.148386 * coszt + 0.0096467) 
                       / (coszt**3 + 0.149864 * coszt**2 + 0.0102963 * coszt 
                          + 0.000303978))
            self.sources['airmass'] = airmass

        # Match sources with the UCAC4 catalogue
        self.log.write('Cross-matching sources with the UCAC4 catalogue', 
                       level=3, event=62)

        if self.ra_ucac is None or self.dec_ucac is None:
            self.log.write('Missing UCAC4 data', level=2, event=62)
        else:
            coords = SkyCoord(ra_finite, dec_finite, unit=(u.deg, u.deg))
            catalog = SkyCoord(self.ra_ucac, self.dec_ucac, unit=(u.deg, u.deg))
            ind_ucac, ds2d, ds3d = match_coordinates_sky(coords, catalog, 
                                                         nthneighbor=1)
            ind_plate = np.arange(ind_ucac.size)
            indmask = ds2d < matchrad_arcsec

            ind_plate = ind_plate[indmask]
            ind_ucac = ind_ucac[indmask]
            matchdist = ds2d[indmask].to(u.arcsec).value

            _,ds2d2,_ = match_coordinates_sky(coords, catalog, 
                                              nthneighbor=2)
            matchdist2 = ds2d2[indmask].to(u.arcsec).value
            _,nn_ds2d,_ = match_coordinates_sky(catalog, catalog, 
                                                nthneighbor=2)
            nndist = nn_ds2d[ind_ucac].to(u.arcsec).value

            num_match = len(ind_plate)
            self.db_update_process(num_ucac4=num_match)

            if num_match > 0:
                ind = ind_finite[ind_plate]
                self.sources['ucac4_id'][ind] = self.id_ucac[ind_ucac]
                self.sources['ucac4_ra'][ind] = self.ra_ucac[ind_ucac]
                self.sources['ucac4_dec'][ind] = self.dec_ucac[ind_ucac]
                self.sources['ucac4_bmag'][ind] = self.bmag_ucac[ind_ucac]
                self.sources['ucac4_vmag'][ind] = self.vmag_ucac[ind_ucac]
                self.sources['ucac4_bmagerr'][ind] = self.bmagerr_ucac[ind_ucac]
                self.sources['ucac4_vmagerr'][ind] = self.vmagerr_ucac[ind_ucac]
                self.sources['ucac4_dist'][ind] = (matchdist
                                                   .astype(np.float32))
                self.sources['ucac4_dist2'][ind] = (matchdist2
                                                    .astype(np.float32))
                self.sources['ucac4_nn_dist'][ind] = (nndist
                                                      .astype(np.float32))

                if self.combined_ucac_apass:
                    self.sources['apass_ra'][ind] = self.ra_apass[ind_ucac]
                    self.sources['apass_dec'][ind] = self.dec_apass[ind_ucac]
                    self.sources['apass_bmag'][ind] = self.bmag_apass[ind_ucac]
                    self.sources['apass_vmag'][ind] = self.vmag_apass[ind_ucac]
                    self.sources['apass_bmagerr'][ind] = self.berr_apass[ind_ucac]
                    self.sources['apass_vmagerr'][ind] = self.verr_apass[ind_ucac]

        # Match sources with the Tycho-2 catalogue
        if self.use_tycho2_fits:
            self.log.write('Cross-matching sources with the Tycho-2 catalogue', 
                           level=3, event=63)

            if self.num_tyc == 0:
                self.log.write('Missing Tycho-2 data', level=2, event=63)
            else:
                coords = SkyCoord(ra_finite, dec_finite, unit=(u.deg, u.deg))
                catalog = SkyCoord(self.ra_tyc, self.dec_tyc, 
                                   unit=(u.deg, u.deg))
                ind_tyc, ds2d, ds3d = match_coordinates_sky(coords, catalog,
                                                            nthneighbor=1)
                ind_plate = np.arange(ind_tyc.size)
                indmask = ds2d < matchrad_arcsec

                ind_plate = ind_plate[indmask]
                ind_tyc = ind_tyc[indmask]
                matchdist = ds2d[indmask].to(u.arcsec).value

                _,ds2d2,_ = match_coordinates_sky(coords, catalog, 
                                                  nthneighbor=2)
                matchdist2 = ds2d2[indmask].to(u.arcsec).value
                _,nn_ds2d,_ = match_coordinates_sky(catalog, catalog, 
                                                    nthneighbor=2)
                nndist = nn_ds2d[ind_tyc].to(u.arcsec).value

                num_match = len(ind_plate)
                self.db_update_process(num_tycho2=num_match)

                if num_match > 0:
                    ind = ind_finite[ind_plate]
                    self.sources['tycho2_id'][ind] = self.id_tyc[ind_tyc]
                    self.sources['tycho2_id_pad'][ind] = self.id_tyc_pad[ind_tyc]
                    self.sources['tycho2_ra'][ind] = self.ra_tyc[ind_tyc]
                    self.sources['tycho2_dec'][ind] = self.dec_tyc[ind_tyc]
                    self.sources['tycho2_btmag'][ind] = self.btmag_tyc[ind_tyc]
                    self.sources['tycho2_vtmag'][ind] = self.vtmag_tyc[ind_tyc]
                    self.sources['tycho2_btmagerr'][ind] = self.btmagerr_tyc[ind_tyc]
                    self.sources['tycho2_vtmagerr'][ind] = self.vtmagerr_tyc[ind_tyc]
                    self.sources['tycho2_hip'][ind] = self.hip_tyc[ind_tyc]
                    self.sources['tycho2_dist'][ind] = (matchdist
                                                        .astype(np.float32))
                    self.sources['tycho2_dist2'][ind] = (matchdist2
                                                         .astype(np.float32))
                    self.sources['tycho2_nn_dist'][ind] = (nndist
                                                           .astype(np.float32))

        # Match sources with the APASS catalogue
        if self.use_apass_db:
            self.log.write('Cross-matching sources with the APASS catalogue', 
                           level=3, event=64)

            # Begin cross-match
            if self.num_apass == 0:
                self.log.write('Missing APASS data', level=2, event=64)
            else:
                if self.combined_ucac_apass:
                    bool_finite_apass = (np.isfinite(self.ra_apass) &
                                         np.isfinite(self.dec_apass))
                    num_finite_apass = bool_finite_apass.sum()

                    if num_finite_apass == 0:
                        self.log.write('No APASS sources with usable '
                                       'coordinates for cross-matching', 
                                       level=2, event=64)
                        return

                    ind_finite_apass = np.where(bool_finite_apass)[0]
                    ra_apass = self.ra_apass[ind_finite_apass]
                    dec_apass = self.dec_apass[ind_finite_apass]
                else:
                    ra_apass = self.ra_apass
                    dec_apass = self.dec_apass

                coords = SkyCoord(ra_finite, dec_finite, unit=(u.deg, u.deg))
                catalog = SkyCoord(ra_apass, dec_apass, unit=(u.deg, u.deg))
                ind_apass, ds2d, ds3d = match_coordinates_sky(coords, catalog, 
                                                              nthneighbor=1)
                ind_plate = np.arange(ind_apass.size)
                indmask = ds2d < matchrad_arcsec

                ind_plate = ind_plate[indmask]
                ind_apass = ind_apass[indmask]
                matchdist = ds2d[indmask].to(u.arcsec).value

                _,ds2d2,_ = match_coordinates_sky(coords, catalog, 
                                                  nthneighbor=2)
                matchdist2 = ds2d2[indmask].to(u.arcsec).value
                _,nn_ds2d,_ = match_coordinates_sky(catalog, catalog, 
                                                    nthneighbor=2)
                nndist = nn_ds2d[ind_apass].to(u.arcsec).value

                num_match = len(ind_plate)
                self.db_update_process(num_apass=num_match)
                self.log.write('Matched {:d} sources with APASS'.format(num_match))

                if num_match > 0:
                    ind = ind_finite[ind_plate]

                    if not self.combined_ucac_apass:
                        self.sources['apass_ra'][ind] = self.ra_apass[ind_apass]
                        self.sources['apass_dec'][ind] = self.dec_apass[ind_apass]
                        self.sources['apass_bmag'][ind] = self.bmag_apass[ind_apass]
                        self.sources['apass_vmag'][ind] = self.vmag_apass[ind_apass]
                        self.sources['apass_bmagerr'][ind] = self.berr_apass[ind_apass]
                        self.sources['apass_vmagerr'][ind] = self.verr_apass[ind_apass]

                    self.sources['apass_dist'][ind] = (matchdist
                                                       .astype(np.float32))
                    self.sources['apass_dist2'][ind] = (matchdist2
                                                        .astype(np.float32))
                    self.sources['apass_nn_dist'][ind] = (nndist
                                                          .astype(np.float32))

    def output_sources_db(self, write_csv=None):
        """
        Write extracted sources to the database.

        """

        if write_csv is None:
            write_csv = self.write_sources_csv

        if write_csv:
            self.log.to_db(3, 'Writing sources to database files', event=80)
        else:
            self.log.to_db(3, 'Writing sources to the database', event=80)

        platedb = PlateDB()
        platedb.assign_conf(self.conf)

        if write_csv:
            # Create output directories, if missing
            if (self.write_db_source_dir and 
                not os.path.isdir(self.write_db_source_dir)):
                self.log.write('Creating output directory {}'
                               .format(self.write_db_source_dir), 
                               level=4, event=80)
                os.makedirs(self.write_db_source_dir)

            if (self.write_db_source_calib_dir and 
                not os.path.isdir(self.write_db_source_calib_dir)):
                self.log.write('Creating output directory {}'
                               .format(self.write_db_source_calib_dir), 
                               level=4, event=80)
                os.makedirs(self.write_db_source_calib_dir)
        else:
            # Open database connection
            self.log.write('Open database connection for writing to the '
                           'source and source_calib tables.')
            platedb.open_connection(host=self.output_db_host,
                                    user=self.output_db_user,
                                    dbname=self.output_db_name,
                                    passwd=self.output_db_passwd)

        # Check for identification numbers and write data
        if (self.scan_id is not None and self.plate_id is not None and 
            self.archive_id is not None and self.process_id is not None):
            platedb.write_sources(self.sources, process_id=self.process_id,
                                  scan_id=self.scan_id, plate_id=self.plate_id,
                                  archive_id=self.archive_id, 
                                  write_csv=write_csv)
        else:
            self.log.write('Cannot write source data due to missing '
                           'plate identification number(s).', 
                           level=2, event=80)

        # Close database connection
        if not write_csv:
            platedb.close_connection()
            self.log.write('Closed database connection.')

    def output_sources_csv(self, filename=None):
        """
        Write extracted sources to a CSV file.

        """

        self.log.to_db(3, 'Writing sources to a file', event=81)

        # Create output directory, if missing
        if self.write_source_dir and not os.path.isdir(self.write_source_dir):
            self.log.write('Creating output directory {}'
                           .format(self.write_source_dir), level=4, event=81)
            os.makedirs(self.write_source_dir)

        if filename:
            fn_world = os.path.join(self.write_source_dir, 
                                    os.path.basename(filename))
        else:
            fn_world = os.path.join(self.write_source_dir, 
                                    '{}_sources.csv'.format(self.basefn))

        outfields = list(_source_meta)
        outfmt = [_source_meta[f][1] for f in outfields]
        outhdr = ','.join(outfields)
        delimiter = ','

        # Output CSV file with extracted sources
        self.log.write('Writing output file {}'.format(fn_world), level=4, 
                       event=81)
        np.savetxt(fn_world, self.sources[outfields], fmt=outfmt, 
                   delimiter=delimiter, header=outhdr, comments='')

