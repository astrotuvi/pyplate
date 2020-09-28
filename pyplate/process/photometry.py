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
from astropy.table import Table, vstack
from astropy.coordinates import Angle, EarthLocation, SkyCoord, ICRS, AltAz
from astropy.coordinates import match_coordinates_sky
from astropy import units as u
from astropy.time import Time
from astropy.stats import sigma_clip
from scipy.interpolate import InterpolatedUnivariateSpline, SmoothBivariateSpline
from scipy.ndimage.filters import generic_filter
from scipy.linalg import lstsq
from collections import OrderedDict
from ..database import PlateDB
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


def _rmse(residuals):
    return np.sqrt(np.mean(residuals**2))


class PhotometryProcess:
    """
    Plate photometry process class

    """

    def __init__(self, filename, archive_id=None):
        self.filename = os.path.basename(filename)
        self.archive_id = archive_id
        self.basefn = ''
        self.fn_fits = ''
        self.process_id = None
        self.scan_id = None
        self.plate_id = None

        self.fits_dir = ''
        self.index_dir = ''
        self.gaia_dir = ''
        self.tycho2_dir = ''
        self.work_dir = ''
        self.write_source_dir = ''
        self.write_db_source_dir = ''
        self.write_db_source_calib_dir = ''
        self.write_phot_dir = ''
        self.write_wcs_dir = ''
        self.write_log_dir = ''

        self.astref_catalog = None
        self.photref_catalog = None

        self.use_gaia_fits = False
        self.use_tycho2_fits = False

        self.use_ucac4_db = False
        self.ucac4_db_host = 'localhost'
        self.ucac4_db_user = ''
        self.ucac4_db_name = ''
        self.ucac4_db_passwd = ''
        self.ucac4_db_table = 'ucac4'

        self.use_apass_db = False
        self.apass_db_host = 'localhost'
        self.apass_db_user = ''
        self.apass_db_name = ''
        self.apass_db_passwd = ''
        self.apass_db_table = 'apass'

        self.output_db_host = 'localhost'
        self.output_db_user = ''
        self.output_db_name = ''
        self.output_db_passwd = ''
        self.write_sources_csv = False

        self.sextractor_path = 'sex'
        self.scamp_path = 'scamp'
        self.psfex_path = 'psfex'
        self.solve_field_path = 'solve-field'
        self.wcs_to_tan_path = 'wcs-to-tan'

        self.timestamp = dt.datetime.now()
        self.timestamp_str = dt.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.scratch_dir = None
        self.enable_log = False
        self.log = None
        self.enable_db_log = False
    
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
        self.num_sources = None
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

        self.sources = None
        self.plate_solution = None

        self.scampref = None
        self.scampcat = None
        self.wcs_header = None
        self.wcshead = None
        self.wcs_plate = None
        self.solutions = None
        self.exp_numbers = None
        self.num_solutions = 0
        self.num_iterations = 0
        self.pattern_x = None
        self.pattern_y = None
        self.pattern_ratio = None
        self.astref_tables = []
        self.gaia_files = None
        self.neighbors_gaia = None

        self.phot_cterm_list = []
        self.phot_calib = None
        self.phot_calib_list = []
        self.phot_calibrated = False
        self.calib_curve = None
        self.faint_limit = None
        self.bright_limit = None

        self.id_tyc = None
        self.id_tyc_pad = None
        self.hip_tyc = None
        self.ra_tyc = None
        self.dec_tyc = None
        self.btmag_tyc = None
        self.vtmag_tyc = None
        self.btmagerr_tyc = None
        self.vtmagerr_tyc = None
        self.num_tyc = 0
        
        self.id_ucac = None
        self.ra_ucac = None
        self.dec_ucac = None
        self.raerr_ucac = None
        self.decerr_ucac = None
        self.mag_ucac = None
        self.bmag_ucac = None
        self.vmag_ucac = None
        self.magerr_ucac = None
        self.bmagerr_ucac = None
        self.vmagerr_ucac = None
        self.num_ucac = 0
        
        self.ra_apass = None
        self.dec_apass = None
        self.bmag_apass = None
        self.vmag_apass = None
        self.berr_apass = None
        self.verr_apass = None
        self.num_apass = 0

        self.combined_ucac_apass = None

        self.ucac4_columns = OrderedDict([
            ('ucac4_ra', ('RAJ2000', 'f8')),
            ('ucac4_dec', ('DEJ2000', 'f8')),
            ('ucac4_raerr', ('e_RAJ2000', 'i')),
            ('ucac4_decerr', ('e_DEJ2000', 'i')),
            ('ucac4_mag', ('amag', 'f8')),
            ('ucac4_magerr', ('e_amag', 'f8')),
            ('ucac4_pmra', ('pmRA', 'f8')),
            ('ucac4_pmdec', ('pmDE', 'f8')),
            ('ucac4_id', ('UCAC4', 'a10')),
            ('ucac4_bmag', ('Bmag', 'f8')),
            ('ucac4_vmag', ('Vmag', 'f8')),
            ('ucac4_bmagerr', ('e_Bmag', 'f8')),
            ('ucac4_vmagerr', ('e_Vmag', 'f8'))
        ])

        self.apass_columns = OrderedDict([
            ('apass_ra', ('RAdeg', 'f8')),
            ('apass_dec', ('DEdeg', 'f8')),
            ('apass_bmag', ('B', 'f8')),
            ('apass_vmag', ('V', 'f8')),
            ('apass_bmagerr', ('e_B', 'f8')),
            ('apass_vmagerr', ('e_V', 'f8'))
        ])

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

    def evaluate_color_term(self, sources, solution_num=0):
        """
        Evaluate color term for a given astrometric solution, using the
        source data and reference catalog.

        Parameters
        ----------
        sources: SourceTable object
            Source catalog with plate magnitudes and external catalog
            (Gaia DR2) magnitudes
        solution_num: int
            Astrometric solution number

        """

        cat_mag1 = sources['gaiadr2_bpmag'].data
        cat_mag2 = sources['gaiadr2_rpmag'].data
        plate_mag = sources['mag_auto'].data
        mag_corr = sources['natmag_correction'].data
        mag_err = sources['natmagerr'].data
        # Replace nans with numerical values
        mag_corr[np.isnan(mag_corr)] = 0.
        mag_err[np.isnan(mag_err)] = 1.
        num_calstars = len(sources)

        # Evaluate color term in 3 iterations

        self.log.write('Determining color term: {:d} stars'
                       ''.format(num_calstars),
                       double_newline=False, level=4, event=72)

        if num_calstars < 10:
            self.log.write('Determining color term: too few stars!',
                           level=2, event=72)
            return None

        _,uind1 = np.unique(cat_mag1, return_index=True)
        plate_mag_u,uind2 = np.unique(plate_mag[uind1], return_index=True)
        cat_mag1_u = cat_mag1[uind1[uind2]]
        cat_mag2_u = cat_mag2[uind1[uind2]]
        mag_corr_u = mag_corr[uind1[uind2]]
        mag_err_u = mag_err[uind1[uind2]]

        # Discard faint sources (within 1 mag from the plate limit)
        kde = sm.nonparametric.KDEUnivariate(plate_mag_u
                                             .astype(np.double))
        kde.fit()
        ind_dense = np.where(kde.density > 0.2*kde.density.max())[0]
        plate_mag_lim = kde.support[ind_dense[-1]]
        ind_nofaint = np.where(plate_mag_u < plate_mag_lim - 1.)[0]
        num_nofaint = len(ind_nofaint)

        self.log.write('Determining color term: {:d} stars after discarding '
                       'faint sources'.format(num_nofaint),
                       double_newline=False, level=4, event=72)

        if num_nofaint < 10:
            self.log.write('Determining color term: too few stars after '
                           'discarding faint sources!',
                           level=2, event=72)
            return None

        frac = 0.2

        if num_nofaint < 500:
            frac = 0.2 + 0.3 * (500 - num_nofaint) / 500.

        plate_mag_u = plate_mag_u[ind_nofaint]
        cat_mag1_u = cat_mag1_u[ind_nofaint]
        cat_mag2_u = cat_mag2_u[ind_nofaint]
        mag_corr_u = mag_corr_u[ind_nofaint]
        mag_err_u = mag_err_u[ind_nofaint]

        # Iteration 1
        cterm_list = np.arange(29) * 0.25 - 3.
        stdev_list = []

        for cterm in cterm_list:
            cat_mag = cat_mag2_u + cterm * (cat_mag1_u - cat_mag2_u)
            z = sm.nonparametric.lowess(cat_mag, plate_mag_u,
                                        frac=frac, it=0, delta=0.2,
                                        return_sorted=True)
            s = InterpolatedUnivariateSpline(z[:,0], z[:,1], k=1)
            mag_diff = cat_mag - s(plate_mag_u) - mag_corr_u
            #stdev_val = mag_diff.std()
            stdev_val = np.sqrt(np.sum((mag_diff / mag_err_u)**2) / len(mag_diff))
            stdev_list.append(stdev_val)

            # Store cterm data
            self.phot_cterm_list.append(OrderedDict([
                ('solution_num', solution_num),
                ('iteration', 1),
                ('cterm', cterm),
                ('stdev', stdev_val),
                ('num_stars', len(mag_diff))
            ]))

        if max(stdev_list) < 0.01:
            self.log.write('Color term fit failed!', level=2, event=72)
            return None

        cf = np.polyfit(cterm_list, stdev_list, 4)
        cf1d = np.poly1d(cf)
        extrema = cf1d.deriv().r
        cterm_extr = extrema[np.where(extrema.imag==0)].real
        der2 = cf1d.deriv(2)(cterm_extr)

        try:
            cterm_min = cterm_extr[np.where((der2 > 0) & (cterm_extr > -2.5) &
                                            (cterm_extr < 3.5))][0]
        except IndexError:
            self.log.write('Color term outside of allowed range!',
                           level=2, event=72)
            return None

        # Eliminate outliers (over 1 mag + sigma clip)
        cat_mag = cat_mag2_u + cterm_min * (cat_mag1_u - cat_mag2_u)
        z = sm.nonparametric.lowess(cat_mag, plate_mag_u,
                                    frac=frac, it=3, delta=0.2,
                                    return_sorted=True)
        s = InterpolatedUnivariateSpline(z[:,0], z[:,1], k=1)
        mag_diff = cat_mag - s(plate_mag_u) - mag_corr_u
        ind1 = np.where(np.absolute(mag_diff) <= 1.)[0]
        flt = sigma_clip(mag_diff[ind1], maxiters=None)
        ind_good1 = ~flt.mask
        ind_good = ind1[ind_good1]

        # Iteration 2
        cterm_list = np.arange(29) * 0.25 - 3.
        stdev_list = []

        frac = 0.2

        if len(ind_good) < 500:
            frac = 0.2 + 0.3 * (500 - len(ind_good)) / 500.

        for cterm in cterm_list:
            cat_mag = cat_mag2_u + cterm * (cat_mag1_u - cat_mag2_u)
            z = sm.nonparametric.lowess(cat_mag[ind_good],
                                        plate_mag_u[ind_good],
                                        frac=frac, it=0, delta=0.2,
                                        return_sorted=True)
            s = InterpolatedUnivariateSpline(z[:,0], z[:,1], k=1)
            mag_diff = (cat_mag[ind_good] - s(plate_mag_u[ind_good])
                        - mag_corr_u[ind_good])
            #stdev_val = mag_diff.std()
            stdev_val = np.sqrt(np.sum((mag_diff / mag_err_u[ind_good])**2) / len(mag_diff))
            stdev_list.append(stdev_val)

            # Store cterm data
            self.phot_cterm_list.append(OrderedDict([
                ('solution_num', solution_num),
                ('iteration', 2),
                ('cterm', cterm),
                ('stdev', stdev_val),
                ('num_stars', len(mag_diff))
            ]))

        stdev_list = np.array(stdev_list)

        if max(stdev_list) < 0.01:
            self.log.write('Color term fit failed!', level=2, event=72)
            return None

        cf, cov = np.polyfit(cterm_list, stdev_list, 2,
                             w=1./stdev_list**2, cov=True)
        cterm_min = -0.5 * cf[1] / cf[0]
        cf_err = np.sqrt(np.diag(cov))
        cterm_min_err = np.sqrt((-0.5 * cf_err[1] / cf[0])**2 +
                                (0.5 * cf[1] * cf_err[0] / cf[0]**2)**2)
        p2 = np.poly1d(cf)
        stdev_fit_iter2 = p2(cterm_min)
        stdev_min_iter2 = np.min(stdev_list)
        cterm_minval_iter2 = np.min(cterm_list)
        cterm_maxval_iter2 = np.max(cterm_list)
        num_stars_iter2 = len(mag_diff)

        if cf[0] < 0 or min(stdev_list) < 0.01 or min(stdev_list) > 1:
            self.log.write('Color term fit failed!', level=2, event=72)
            return None

        # Iteration 3
        cterm_list = (np.arange(61) * 0.02 +
                      round(cterm_min*50.)/50. - 0.6)
        stdev_list = []

        for cterm in cterm_list:
            cat_mag = cat_mag2_u + cterm * (cat_mag1_u - cat_mag2_u)
            z = sm.nonparametric.lowess(cat_mag[ind_good],
                                        plate_mag_u[ind_good],
                                        frac=frac, it=0, delta=0.2,
                                        return_sorted=True)
            s = InterpolatedUnivariateSpline(z[:,0], z[:,1], k=1)
            mag_diff = (cat_mag[ind_good] - s(plate_mag_u[ind_good])
                        - mag_corr_u[ind_good])
            #stdev_val = mag_diff.std()
            stdev_val = np.sqrt(np.sum((mag_diff / mag_err_u[ind_good])**2) / len(mag_diff))
            stdev_list.append(stdev_val)

            # Store cterm data
            self.phot_cterm_list.append(OrderedDict([
                ('solution_num', solution_num),
                ('iteration', 3),
                ('cterm', cterm),
                ('stdev', stdev_val),
                ('num_stars', len(mag_diff))
            ]))

        stdev_list = np.array(stdev_list)

        cf, cov = np.polyfit(cterm_list, stdev_list, 2,
                             w=1./stdev_list**2, cov=True)
        cterm = -0.5 * cf[1] / cf[0]
        cf_err = np.sqrt(np.diag(cov))
        cterm_err = np.sqrt((-0.5 * cf_err[1] / cf[0])**2 +
                            (0.5 * cf[1] * cf_err[0] / cf[0]**2)**2)
        p2 = np.poly1d(cf)
        stdev_fit = p2(cterm)
        stdev_min = np.min(stdev_list)
        cterm_minval = np.min(cterm_list)
        cterm_maxval = np.max(cterm_list)
        num_stars = len(mag_diff)
        iteration = 3

        if cf[0] < 0 or cterm < -2 or cterm > 3:
            if cf[0] < 0:
                self.log.write('Color term fit not reliable!',
                               level=2, event=72)
            else:
                self.log.write('Color term outside of allowed range '
                               '({:.3f})!'.format(cterm),
                               level=2, event=72)

            if cterm_min < -2 or cterm_min > 3:
                self.log.write('Color term from previous iteration '
                               'outside of allowed range ({:.3f})!'
                               ''.format(cterm_min),
                               level=2, event=72)
                return None
            else:
                cterm = cterm_min
                cterm_err = cterm_min_err
                stdev_fit = stdev_fit_iter2
                stdev_min = stdev_min_iter2
                cterm_minval = cterm_minval_iter2
                cterm_maxval = cterm_maxval_iter2
                num_stars = num_stars_iter2
                iteration = 2

            self.log.write('Taking color term from previous iteration',
                           level=4, event=72)

        # Create dictionary for calibration results, if not exists
        if self.phot_calib is None:
            self.phot_calib = OrderedDict()
            self.phot_calib['solution_num'] = solution_num
            self.phot_calib['iteration'] = 0

        # Store color term result
        self.phot_calib['color_term'] = cterm
        self.phot_calib['color_term_err'] = cterm_err
        self.phot_calib['cterm_stdev_fit'] = stdev_fit
        self.phot_calib['cterm_stdev_min'] = stdev_min
        self.phot_calib['cterm_range_min'] = cterm_minval
        self.phot_calib['cterm_range_max'] = cterm_maxval
        self.phot_calib['cterm_iterations'] = iteration
        self.phot_calib['cterm_num_stars'] = num_stars

        self.log.write('Plate color term (solution {:d}): {:.3f} ({:.3f})'
                       .format(solution_num, cterm, cterm_err),
                       level=4, event=72)

    def calibrate_photometry_gaia(self, solution_num=None, iteration=1):
        """
        Calibrate extracted magnitudes with Gaia data.

        """

        num_solutions = self.plate_solution.num_solutions

        assert (solution_num is None or 
                (solution_num > 0 and solution_num <= num_solutions))

        self.log.write('Photometric calibration: solution {:d}, iteration {:d}'
                       .format(solution_num, iteration), level=3, event=70)

        # Initialise the flag value
        self.phot_calibrated = False

        if 'METHOD' in self.plate_header:
            pmethod = self.plate_header['METHOD']

            if (pmethod is not None and pmethod != '' 
                and 'direct photograph' not in pmethod 
                and 'focusing' not in pmethod
                and 'test plate' not in pmethod):
                self.log.write('Cannot calibrate photometry due to unsupported'
                               'observation method ({:s})'.format(pmethod),
                               level=2, event=70)
                #self.db_update_process(calibrated=0)
                return

        # Create dictionary for calibration results
        self.phot_calib = OrderedDict()

        # Create output directory, if missing
        if self.write_phot_dir and not os.path.isdir(self.write_phot_dir):
            self.log.write('Creating output directory {}'
                           .format(self.write_phot_dir), level=4, event=70)
            os.makedirs(self.write_phot_dir)

        if self.write_phot_dir:
            fn_cterm = os.path.join(self.write_phot_dir,
                                    '{}_cterm.txt'.format(self.basefn))
            fcterm = open(fn_cterm, 'wb')
            fn_caldata = os.path.join(self.write_phot_dir, 
                                      '{}_caldata.txt'.format(self.basefn))
            fcaldata = open(fn_caldata, 'wb')
            #fn_calcurve = os.path.join(self.write_phot_dir, 
            #                           '{}_calcurve.txt'.format(self.basefn))
            #fcalcurve = open(fn_calcurve, 'wb')
            #fn_cutdata = os.path.join(self.write_phot_dir, 
            #                          '{}_cutdata.txt'.format(self.basefn))
            #fcutdata = open(fn_cutdata, 'wb')
            #fn_cutcurve = os.path.join(self.write_phot_dir, 
            #                           '{}_cutcurve.txt'.format(self.basefn))
            #fcutcurve = open(fn_cutcurve, 'wb')

        # Select sources for photometric calibration
        self.log.write('Selecting sources for photometric calibration', 
                       level=3, event=71)

        if solution_num is None:
            solution_num = 1

        self.phot_calib['solution_num'] = solution_num
        self.phot_calib['iteration'] = iteration

        # For single exposures, exclude blended sources.
        # For multiple exposures, include them, because otherwise the bright
        # end will lack calibration stars.
        if num_solutions == 1:
            bflags = ((self.sources['sextractor_flags'] == 0) |
                      (self.sources['sextractor_flags'] == 2))
        else:
            bflags = self.sources['sextractor_flags'] <= 3

        # Create calibration-star mask
        # Discard very red stars (BP-RP > 2)
        cal_mask = ((self.sources['solution_num'] == solution_num) &
                    (self.sources['mag_auto'] > 0) &
                    (self.sources['mag_auto'] < 90) &
                    bflags &
                    (self.sources['flag_clean'] == 1) &
                    ~self.sources['gaiadr2_bpmag'].mask &
                    ~self.sources['gaiadr2_rpmag'].mask &
                    (self.sources['gaiadr2_bp_rp'].filled(99.) <= 2) &
                    (self.sources['gaiadr2_neighbors'] == 1))

        num_calstars = cal_mask.sum()
        self.phot_calib['num_candidate_stars'] = num_calstars

        if num_calstars == 0:
            self.log.write('No stars for photometric calibration',
                           level=2, event=71)
            #self.db_update_process(calibrated=0)
            return

        self.log.write('Found {:d} calibration-star candidates with '
                       'Gaia magnitudes on the plate'
                       .format(num_calstars), level=4, event=71)

        if num_calstars < 10:
            self.log.write('Too few calibration stars on the plate!',
                           level=2, event=71)
            #self.db_update_process(calibrated=0)
            return

        # Evaluate color term

        if iteration == 1:
            self.log.write('Determining color term using annular bins 1-3', 
                           level=3, event=72)
            cterm_mask = cal_mask & (self.sources['annular_bin'] <= 3)
        else:
            self.log.write('Determining color term using annular bins 1-8', 
                           level=3, event=72)
            cterm_mask = cal_mask & (self.sources['annular_bin'] <= 8)

        self.evaluate_color_term(self.sources[cterm_mask],
                                 solution_num=solution_num)

        # If color term was not determined, we need to terminate the
        # calibration
        if 'color_term' not in self.phot_calib:
            self.log.write('Cannot continue photometric calibration without '
                           'color term', level=2, event=72)
            return

        cterm = self.phot_calib['color_term']
        cterm_err = self.phot_calib['color_term_err']

        #self.db_update_process(color_term=cterm)

        # Use stars in all annular bins
        self.log.write('Photometric calibration using annular bins 1-9', 
                       level=3, event=73)

        # Select stars with unique plate mag values
        plate_mag = self.sources['mag_auto'][cal_mask].data
        plate_mag_u,uind = np.unique(plate_mag, return_index=True)
        ind_calibstar_u = np.where(cal_mask)[0][uind]
        #cal_u_mask = np.zeros_like(cal_mask)
        #cal_u_mask[np.where(cal_mask)[0][uind]] = True
        num_cal_u = len(plate_mag_u)

        self.log.write('{:d} stars with unique magnitude'
                       .format(num_cal_u), 
                       double_newline=False, level=4, event=73)

        if num_cal_u < 10:
            self.log.write('Too few stars with unique magnitude!',
                           double_newline=False, level=2, event=73)
            #self.db_update_process(calibrated=0)
            return

        plate_mag_u = self.sources['mag_auto'][ind_calibstar_u].data
        cat_bmag_u = self.sources['gaiadr2_bpmag'][ind_calibstar_u].data
        cat_vmag_u = self.sources['gaiadr2_rpmag'][ind_calibstar_u].data
        cat_natmag = cat_vmag_u + cterm * (cat_bmag_u - cat_vmag_u)
        self.sources['cat_natmag'][ind_calibstar_u] = cat_natmag

        # Eliminate outliers by constructing calibration curve from
        # the bright end and extrapolate towards faint stars

        # Find initial plate magnitude limit
        kde = sm.nonparametric.KDEUnivariate(plate_mag_u
                                             .astype(np.double))
        kde.fit()
        ind_maxden = np.argmax(kde.density)
        plate_mag_maxden = kde.support[ind_maxden]
        ind_dense = np.where(kde.density > 0.2*kde.density.max())[0]
        brightmag = kde.support[ind_dense[0]]
        plate_mag_lim = kde.support[ind_dense[-1]]
        plate_mag_brt = plate_mag_u.min()
        plate_mag_mid = (plate_mag_brt + 
                         0.5 * (plate_mag_lim - plate_mag_brt))

        if brightmag > plate_mag_mid:
            brightmag = plate_mag_mid

        # Check the number of stars in the bright end
        nb = (plate_mag_u <= plate_mag_mid).sum()

        if nb < 10:
            plate_mag_mid = plate_mag_u[9]

        # Construct magnitude cuts for outlier elimination
        ncuts = int((plate_mag_lim - plate_mag_mid) / 0.5) + 2
        mag_cuts = np.linspace(plate_mag_mid, plate_mag_lim, ncuts)
        ind_cut = np.where(plate_mag_u <= plate_mag_mid)[0]
        ind_good = np.arange(len(ind_cut))
        mag_cut_prev = mag_cuts[0]
        #mag_slope_prev = None

        # Loop over magnitude bins
        for mag_cut in mag_cuts[1:]:
            gpmag = plate_mag_u[ind_cut[ind_good]]
            gcmag = cat_natmag[ind_cut[ind_good]]

            nbright = (gpmag < brightmag).sum()

            if nbright < 20:
                alt_brightmag = (plate_mag_u.min() + 
                                 (plate_mag_maxden - plate_mag_u.min()) * 0.5)
                nbright = (gpmag < alt_brightmag).sum()

            if nbright < 10:
                nbright = 10

            # Exclude bright outliers by fitting a line and checking 
            # if residuals are larger than 2 mag
            ind_outliers = np.array([], dtype=int)
            xdata = gpmag[:nbright]
            ydata = gcmag[:nbright]
            p1 = np.poly1d(np.polyfit(xdata, ydata, 1))
            res = cat_natmag[ind_cut] - p1(plate_mag_u[ind_cut])
            ind_brightout = np.where((np.absolute(res) > 2.) &
                                     (plate_mag_u[ind_cut] <= 
                                      xdata.max()))[0]

            if len(ind_brightout) > 0:
                ind_outliers = np.append(ind_outliers, 
                                         ind_cut[ind_brightout])
                ind_good = np.setdiff1d(ind_good, ind_outliers)
                gpmag = plate_mag_u[ind_cut[ind_good]]
                gcmag = cat_natmag[ind_cut[ind_good]]
                nbright -= len(ind_brightout)

                if nbright < 10:
                    nbright = 10

            # Construct calibration curve
            # Set lowess fraction depending on the number of data points
            frac = 0.2

            if len(ind_good) < 500:
                frac = 0.2 + 0.3 * (500 - len(ind_good)) / 500.

            z = sm.nonparametric.lowess(gcmag, gpmag, 
                                        frac=frac, it=3, delta=0.1, 
                                        return_sorted=True)

            # In case there are less than 20 good stars, use only 
            # polynomial
            if len(ind_good) < 20:
                weights = np.zeros(len(ind_good)) + 1.

                for i in np.arange(len(ind_good)):
                    indw = np.where(np.absolute(gpmag-gpmag[i]) < 1.0)[0]

                    if len(indw) > 2:
                        weights[i] = 1. / gcmag[indw].std()**2

                p2 = np.poly1d(np.polyfit(gpmag, gcmag, 2, w=weights))
                z[:,1] = p2(z[:,0])

            # Improve bright-star calibration
            if nbright > len(ind_good):
                nbright = len(ind_good)

            xbright = gpmag[:nbright]
            ybright = gcmag[:nbright]

            if nbright < 50:
                p2 = np.poly1d(np.polyfit(xbright, ybright, 2))
                vals = p2(xbright)
            else:
                z1 = sm.nonparametric.lowess(ybright, xbright, 
                                             frac=0.4, it=3, delta=0.1, 
                                             return_sorted=True)
                vals = z1[:,1]

            weight2 = np.arange(nbright, dtype=float) / nbright
            weight1 = 1. - weight2
            z[:nbright,1] = weight1 * vals + weight2 * z[:nbright,1]

            # Improve faint-star calibration by fitting a 2nd order
            # polynomial
            # Currently, disable improvement
            improve_faint = False
            if improve_faint:
                ind_faint = np.where(gpmag > mag_cut_prev-6.)[0]
                nfaint = len(ind_faint)

                if nfaint > 5:
                    xfaint = gpmag[ind_faint]
                    yfaint = gcmag[ind_faint]
                    weights = np.zeros(nfaint) + 1.

                    for i in np.arange(nfaint):
                        indw = np.where(np.absolute(xfaint-xfaint[i]) < 0.5)[0]

                        if len(indw) > 2:
                            weights[i] = 1. / yfaint[indw].std()**2

                    p2 = np.poly1d(np.polyfit(xfaint, yfaint, 2, 
                                              w=weights))
                    vals = p2(xfaint)

                    weight2 = (np.arange(nfaint, dtype=float) / nfaint)**1
                    weight1 = 1. - weight2
                    z[ind_faint,1] = weight2 * vals + weight1 * z[ind_faint,1]

            # Interpolate smoothed calibration curve
            s = InterpolatedUnivariateSpline(z[:,0], z[:,1], k=1)

            ind_cut = np.where(plate_mag_u <= mag_cut)[0]
            fit_mag = s(plate_mag_u[ind_cut])

            residuals = cat_natmag[ind_cut] - fit_mag
            mag_cut_prev = mag_cut

            #if b == 0 and self.write_phot_dir:
            #    np.savetxt(fcutdata, np.column_stack((plate_mag_u[ind_cut],
            #                                          cat_natmag[ind_cut], 
            #                                          fit_mag, residuals)))
            #    fcutdata.write('\n\n')
            #    np.savetxt(fcutdata, np.column_stack((gpmag, gcmag, 
            #                                          fit_mag[ind_good], residuals[ind_good])))
            #    fcutdata.write('\n\n')
            #    np.savetxt(fcutcurve, z)
            #    fcutcurve.write('\n\n')

            ind_outliers = np.array([], dtype=int)

            # Mark as outliers those stars that deviate more than 1 mag
            ind_out = np.where(np.absolute(residuals) > 1.0)

            if len(ind_out) > 0:
                ind_outliers = np.append(ind_outliers, ind_cut[ind_out])
                ind_outliers = np.unique(ind_outliers)

            # Additionally clip outliers in small bins
            for mag_loc in np.linspace(plate_mag_brt, mag_cut, 100):
                mag_low = mag_loc - 0.5
                mag_high = mag_loc + 0.5
                ind_loc = np.where((plate_mag_u[ind_cut] > mag_low) &
                                   (plate_mag_u[ind_cut] < mag_high))[0]
                ind_loc = np.setdiff1d(ind_loc, ind_outliers)

                if len(ind_loc) >= 5:
                    rms_res = np.sqrt((residuals[ind_loc]**2).sum())
                    ind_locout = np.where(np.absolute(residuals[ind_loc]) > 
                                          3.*rms_res)[0]

                    if len(ind_locout) > 0:
                        ind_outliers = np.append(ind_outliers, 
                                                 ind_cut[ind_loc[ind_locout]])

                    ind_outliers = np.unique(ind_outliers)

            ind_good = np.setdiff1d(np.arange(len(ind_cut)), 
                                    ind_outliers)

            #flt = sigma_clip(residuals, maxiters=None)
            #ind_good = ~flt.mask
            #ind_good = np.where(np.absolute(residuals) < 3*residuals.std())[0]

            # Stop outlier elimination if there is a gap in magnitudes
            if mag_cut - plate_mag_u[ind_cut[ind_good]].max() > 1.5:
                ind_faintout = np.where(plate_mag_u > mag_cut)[0]

                if len(ind_faintout) > 0:
                    ind_outliers = np.append(ind_outliers, ind_faintout)
                    ind_outliers = np.unique(ind_outliers)
                    ind_good = np.setdiff1d(np.arange(len(plate_mag_u)),
                                            ind_outliers)
                    self.log.write('{:d} faint stars '
                                   'eliminated as outliers'
                                   .format(len(ind_faintout)),
                                   double_newline=False,
                                   level=4, event=73)

                self.log.write('Outlier elimination '
                               'stopped due to a long gap in '
                               'magnitudes!',
                                double_newline=False,
                               level=2, event=73)
                break

            if len(ind_good) < 10:
                self.log.write('Outlier elimination stopped '
                               'due to insufficient number of stars left!',
                                double_newline=False, level=2, event=73)
                break

        num_outliers = len(ind_outliers)
        self.log.write('{:d} outliers eliminated'
                       ''.format(num_outliers), 
                       double_newline=False, level=4, event=73)
        ind_good = np.setdiff1d(np.arange(len(plate_mag_u)), 
                                ind_outliers)
        self.log.write('{:d} stars after outlier '
                       'elimination'.format(len(ind_good)), 
                       double_newline=False, level=4, event=73)

        if len(ind_good) < 10:
            self.log.write('Too few calibration '
                           'stars ({:d}) after outlier elimination!'
                           .format(len(ind_good)), 
                           double_newline=False, level=2, event=73)
            #self.db_update_process(calibrated=0)
            return

        # Continue with photometric calibration without outliers

        # Study the distribution of magnitudes
        kde = sm.nonparametric.KDEUnivariate(plate_mag_u[ind_good]
                                             .astype(np.double))
        kde.fit()
        ind_maxden = np.argmax(kde.density)
        plate_mag_maxden = kde.support[ind_maxden]
        ind_dense = np.where(kde.density > 0.2*kde.density.max())[0]
        plate_mag_lim = kde.support[ind_dense[-1]]
        ind_valid = np.where(plate_mag_u[ind_good] <= plate_mag_lim)[0]
        num_valid = len(ind_valid)

        self.log.write('{:d} calibration stars '
                       'brighter than limiting magnitude'
                       .format(num_valid), 
                       double_newline=False, level=4, event=73)

        #valid_cal_mask = np.zeros_like(cal_u_mask)
        #valid_cal_mask[np.where(cal_u_mask)[0][ind_good[ind_valid]]] = True
        ind_calibstar_valid = ind_calibstar_u[ind_good[ind_valid]]
        self.sources['phot_calib_flags'][ind_calibstar_valid] = 1

        if num_outliers > 0:
            #outlier_mask = np.zeros_like(cal_u_mask)
            #outlier_mask[np.where(cal_u_mask)[0][ind_outliers]]
            ind_calibstar_outlier = ind_calibstar_u[ind_outliers]
            self.sources['phot_calib_flags'][ind_calibstar_outlier] = 2

        cat_natmag = cat_natmag[ind_good[ind_valid]]
        plate_mag_u = plate_mag_u[ind_good[ind_valid]]
        plate_mag_brightest = plate_mag_u.min()
        frac = 0.2

        if num_valid < 500:
            frac = 0.2 + 0.3 * (500 - num_valid) / 500.

        z = sm.nonparametric.lowess(cat_natmag, plate_mag_u, 
                                    frac=frac, it=3, delta=0.1, 
                                    return_sorted=True)

        # Improve bright-star calibration

        # Find magnitude at which the frequency of stars becomes
        # larger than 500 mag^(-1)
        #ind_500 = np.where((kde.density*len(ind_good) > 500))[0][0]
        #brightmag = kde.support[ind_500]

        # Find magnitude at which density becomes larger than 0.05 of
        # the max density
        #ind_dense_005 = np.where(kde.density > 0.05*kde.density.max())[0]
        # Index of kde.support at which density becomes 0.05 of max
        #ind0 = ind_dense_005[0]
        #brightmag = kde.support[ind0]
        #nbright = len(plate_mag_u[np.where(plate_mag_u < brightmag)])

        # Find magnitude at which density becomes larger than 0.2 of
        # the max density
        #brightmag = kde.support[ind_dense[0]]
        #nbright = len(plate_mag_u[np.where(plate_mag_u < brightmag)])

        # Find the second percentile of magnitudes
        nbright = round(num_valid * 0.02)

        # Limit bright stars with 2000
        nbright = min([nbright, 2000])

        if nbright < 20:
            brightmag = (plate_mag_brightest + 
                         (plate_mag_maxden - plate_mag_brightest) * 0.5)
            nbright = len(plate_mag_u[np.where(plate_mag_u < brightmag)])

        if nbright < 5:
            nbright = 5

        if nbright < 50:
            p2 = np.poly1d(np.polyfit(plate_mag_u[:nbright], 
                                      cat_natmag[:nbright], 2))
            vals = p2(plate_mag_u[:nbright])
        else:
            z1 = sm.nonparametric.lowess(cat_natmag[:nbright], 
                                         plate_mag_u[:nbright], 
                                         frac=0.4, it=3, delta=0.1, 
                                         return_sorted=True)
            vals = z1[:,1]

        t = Table()
        t['plate_mag'] = plate_mag_u[:nbright]
        t['cat_natmag'] = cat_natmag[:nbright]
        t['fit_mag'] = vals
        basefn_solution = '{}-{:02d}'.format(self.basefn, solution_num)
        fn_tab = os.path.join(self.scratch_dir, 
                              '{}_bright.fits'.format(basefn_solution))
        t.write(fn_tab, format='fits', overwrite=True)

        # Normalise density to max density of the bright range
        #d_bright = kde.density[:ind0] / kde.density[:ind0].max()
        # Find a smooth density curve and use values as weights
        #s_bright = InterpolatedUnivariateSpline(kde.support[:ind0],
        #                                        d_bright, k=1)
        #weight2 = s_bright(plate_mag_u[:nbright])

        # Linearly increasing weight
        weight2 = np.arange(nbright, dtype=float) / nbright

        weight1 = 1. - weight2

        # Merge two calibration curves with different weights
        z[:nbright,1] = weight1 * vals + weight2 * z[:nbright,1]

        # Interpolate the whole calibration curve
        s = InterpolatedUnivariateSpline(z[:,0], z[:,1], k=1)

        # Store the calibration curve
        self.calib_curve = s

        # Calculate residuals
        residuals = cat_natmag - s(plate_mag_u)

        # Smooth residuals with spline
        X = self.sources['x_source'][ind_calibstar_valid].data
        Y = self.sources['y_source'][ind_calibstar_valid].data

        if num_valid > 100:
            s_corr = SmoothBivariateSpline(X, Y, residuals, kx=5, ky=5)
        elif num_valid > 50:
            s_corr = SmoothBivariateSpline(X, Y, residuals, kx=3, ky=3)
        else:
            s_corr = None

        # Calculate new residuals and correct for dependence on
        # x, y, mag_auto. Do it only if the number of valid
        # calibration stars is larger than 500.
        s_magcorr = None

        if num_valid > 500:
            residuals2 = np.zeros(num_valid)

            for i in np.arange(num_valid):
                residuals2[i] = residuals[i] - s_corr(X[i], Y[i])

            # Create magnitude bins
            plate_mag_srt = np.sort(plate_mag_u)
            bin_mag = [(plate_mag_srt[99] + plate_mag_srt[0]) / 2.]
            bin_hw = [(plate_mag_srt[99] - plate_mag_srt[0]) / 2.]
            ind_lastmag = 99

            while True:
                if plate_mag_srt[ind_lastmag+100] - bin_mag[-1] - bin_hw[-1] > 0.5:
                    bin_edge = bin_mag[-1] + bin_hw[-1]
                    bin_mag.append((plate_mag_srt[ind_lastmag+100] + bin_edge) / 2.)
                    bin_hw.append((plate_mag_srt[ind_lastmag+100] - bin_edge) / 2.)
                    ind_lastmag += 100
                else:
                    bin_mag.append(bin_mag[-1] + bin_hw[-1] + 0.25)
                    bin_hw.append(0.25)
                    ind_lastmag = (plate_mag_srt < bin_mag[-1] + 0.25).sum() - 1

                # If less than 100 sources remain
                if ind_lastmag > num_valid - 101:
                    add_width = plate_mag_srt[-1] - bin_mag[-1] - bin_hw[-1]
                    bin_mag[-1] += add_width / 2.
                    bin_hw[-1] += add_width / 2.
                    break

            # Evaluate natmag correction in magnitude bins
            s_magcorr = []

            for i, (m, hw) in enumerate(zip(bin_mag, bin_hw)):
                binmask = (plate_mag_u > m-hw) & (plate_mag_u <= m+hw)
                #print(m, m-hw, m+hw, binmask.sum())
                smag = SmoothBivariateSpline(X[binmask], Y[binmask],
                                             residuals2[binmask],
                                             kx=3, ky=3)
                s_magcorr.append(smag)

        # Evaluate RMS errors from the calibration residuals
        rmse_list = generic_filter(residuals, _rmse, size=10)
        rmse_lowess = sm.nonparametric.lowess(rmse_list, plate_mag_u, 
                                              frac=0.5, it=3, delta=0.1)
        s_rmse = InterpolatedUnivariateSpline(rmse_lowess[:,0],
                                              rmse_lowess[:,1], k=1)
        rmse = s_rmse(plate_mag_u)

        if self.write_phot_dir:
            np.savetxt(fcaldata, np.column_stack((plate_mag_u, cat_natmag, 
                                                  s(plate_mag_u), 
                                                  cat_natmag-s(plate_mag_u))))
            fcaldata.write('\n\n')
            #np.savetxt(fcalcurve, z)
            #fcalcurve.write('\n\n')

        # Store calibration statistics
        bright_limit = s(plate_mag_brightest).item()
        faint_limit = s(plate_mag_lim).item()

        self.phot_calib['num_calib_stars'] = num_valid
        self.phot_calib['num_bright_stars'] = nbright
        self.phot_calib['num_outliers'] = num_outliers
        self.phot_calib['bright_limit'] = bright_limit
        self.phot_calib['faint_limit'] = faint_limit
        self.phot_calib['mag_range'] = faint_limit - bright_limit
        self.phot_calib['rmse_min'] = rmse.min()
        self.phot_calib['rmse_median'] = np.median(rmse)
        self.phot_calib['rmse_max'] = rmse.max()
        self.phot_calib['plate_mag_brightest'] = plate_mag_brightest
        self.phot_calib['plate_mag_density02'] = kde.support[ind_dense[0]]
        self.phot_calib['plate_mag_brightcut'] = brightmag
        self.phot_calib['plate_mag_maxden'] = plate_mag_maxden
        self.phot_calib['plate_mag_lim'] = plate_mag_lim

        # Append calibration results to the list
        self.phot_calib_list.append(self.phot_calib)

        # Apply photometric calibration to sources
        sol_mask = ((self.sources['solution_num'] == solution_num) &
                    (self.sources['mag_auto'] < 90.))
        num_solstars = sol_mask.sum()
        mag_auto_sol = self.sources['mag_auto'][sol_mask]

        self.log.write('Applying photometric calibration to sources '
                       'in annular bins 1-9',
                       level=3, event=74)

        # Correct magnitudes for positional effects
        if s_corr is not None:
            natmag_corr = self.sources['natmag_correction'][sol_mask]
            xsrc = self.sources['x_source'][sol_mask]
            ysrc = self.sources['y_source'][sol_mask]

            # Do a for-cycle, because SmoothBivariateSpline may crash with
            # large input arrays
            for i in np.arange(num_solstars):
                # Apply first correction (dependent only on coordinates)
                natmag_corr[i] = s_corr(xsrc[i], ysrc[i])

                # Apply second correction (dependent on mag_auto)
                if s_magcorr is not None:
                    corr_list = []

                    for smag in s_magcorr:
                        corr_list.append(smag(xsrc[i], ysrc[i])[0,0])

                    smc = InterpolatedUnivariateSpline(bin_mag, corr_list, k=1)
                    natmag_corr[i] += smc(mag_auto_sol[i])

        # Assign magnitudes and errors
        self.sources['natmag_plate'][sol_mask] = s(mag_auto_sol)
        self.sources['natmagerr_plate'][sol_mask] = s_rmse(mag_auto_sol)
        self.sources['natmag'][sol_mask] = s(mag_auto_sol)
        self.sources['natmagerr'][sol_mask] = s_rmse(mag_auto_sol)

        if s_corr is not None:
            self.sources['natmag_correction'][sol_mask] = natmag_corr
            self.sources['natmag'][sol_mask] += natmag_corr

        self.sources['color_term'][sol_mask] = cterm
        self.sources['natmag_residual'][ind_calibstar_u] = \
                (self.sources['cat_natmag'][ind_calibstar_u] - 
                 self.sources['natmag'][ind_calibstar_u])

        # Apply flags and errors to sources outside the magnitude range 
        # of calibration stars
        brange = (mag_auto_sol < plate_mag_brightest)
        ind = np.where(sol_mask)[0][brange]

        if brange.sum() > 0:
            self.sources['phot_plate_flags'][ind] = 1
            self.sources['natmagerr'][ind] = s_rmse(plate_mag_brightest)
            self.sources['natmagerr_plate'][ind] = s_rmse(plate_mag_brightest)

        brange = (mag_auto_sol > plate_mag_lim)
        ind = np.where(sol_mask)[0][brange]

        if brange.sum() > 0:
            self.sources['phot_plate_flags'][ind] = 2

        # Select stars with known external photometry
        bgaia = (sol_mask &
                 ~self.sources['gaiadr2_bpmag'].mask &
                 ~self.sources['gaiadr2_rpmag'].mask)

        if bgaia.sum() > 0:
            b_v = self.sources['gaiadr2_bp_rp'][bgaia]
            #b_v_err = np.sqrt(self.sources[ind]['apass_bmagerr']**2 +
            #                  self.sources[ind]['apass_vmagerr']**2)

            self.sources['color_bv'][bgaia] = b_v
            #self.sources['color_bv_err'][ind] = b_v_err
            self.sources['vmag'][bgaia] = (self.sources['natmag'][bgaia]
                                           - cterm * b_v)
            self.sources['bmag'][bgaia] = (self.sources['natmag'][bgaia]
                                           - (cterm - 1.) * b_v)
            #vmagerr = np.sqrt(self.sources['natmagerr'][ind]**2 + 
            #                  (cterm_err * b_v)**2 +
            #                  (cterm * b_v_err)**2)
            #bmagerr = np.sqrt(self.sources['natmagerr'][ind]**2 + 
            #                  (cterm_err * b_v)**2 + 
            #                  ((cterm - 1.) * b_v_err)**2)
            #self.sources['vmagerr'][ind] = vmagerr
            #self.sources['bmagerr'][ind] = bmagerr

        try:
            brightlim = min([cal['bright_limit']
                             for cal in self.phot_calib_list
                             if cal['solution_num'] == solution_num
                             and cal['iteration'] == iteration])
            faintlim = max([cal['faint_limit']
                            for cal in self.phot_calib_list
                            if cal['solution_num'] == solution_num
                            and cal['iteration'] == iteration])
            mag_range = faintlim - brightlim
        except Exception:
            brightlim = None
            faintlim = None
            mag_range = None

        if num_valid > 0:
            self.phot_calibrated = True
            self.bright_limit = brightlim
            self.faint_limit = faintlim

            self.log.write('Photometric calibration results (solution {:d}, '
                           'iteration {:d}): '
                           'bright limit {:.3f}, faint limit {:.3f}'
                           .format(solution_num, iteration, brightlim,
                                   faintlim),
                           level=4, event=73)

            #self.db_update_process(bright_limit=brightlim, faint_limit=faintlim,
            #                       mag_range=mag_range, num_calib=num_calib, 
            #                       calibrated=1)
        #else:
            #self.db_update_process(num_calib=0, calibrated=0)

        if self.write_phot_dir:
            fcaldata.close()
            #fcalcurve.close()
            #fcutdata.close()
            #fcutcurve.close()
