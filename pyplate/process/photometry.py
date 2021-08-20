import os
import math
import numpy as np
from astropy.table import Table
from astropy.stats import sigma_clip
from scipy.interpolate import InterpolatedUnivariateSpline, SmoothBivariateSpline
from scipy.ndimage.filters import generic_filter
from collections import OrderedDict
from ..conf import read_conf

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

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

    def __init__(self):
        self.basefn = ''
        self.write_phot_dir = ''
        self.scratch_dir = None
        self.log = None
    
        self.plate_header = None
        self.platemeta = None

        self.sources = None
        self.plate_solution = None

        self.phot_cterm_list = []
        self.phot_calib = None
        self.phot_calib_list = []
        self.phot_calibrated = False
        self.calib_curve = None
        self.faint_limit = None
        self.bright_limit = None

    def assign_conf(self, conf):
        """
        Parse configuration and set class attributes.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)
            
        self.conf = conf

        for attr in ['write_phot_dir']:
            try:
                setattr(self, attr, conf.get('Files', attr))
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

        cat_mag1 = sources['gaiaedr3_bpmag'].data
        cat_mag2 = sources['gaiaedr3_rpmag'].data
        plate_mag = sources['mag_auto'].data
        mag_corr = sources['natmag_correction'].data
        mag_err = sources['natmag_error'].data
        # Replace nans with numerical values
        mag_corr[np.isnan(mag_corr)] = 0.
        mag_err[np.isnan(mag_err)] = 1.
        num_calstars = len(sources)

        # Evaluate color term in 3 iterations

        self.log.write('Determining color term: {:d} stars'
                       ''.format(num_calstars),
                       double_newline=False, level=4, event=72,
                       solution_num=solution_num)

        if num_calstars < 10:
            self.log.write('Determining color term: too few stars!',
                           level=2, event=72, solution_num=solution_num)
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
                       double_newline=False, level=4, event=72,
                       solution_num=solution_num)

        if num_nofaint < 10:
            self.log.write('Determining color term: too few stars after '
                           'discarding faint sources!',
                           level=2, event=72, solution_num=solution_num)
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
        cterm_list = np.arange(33) * 0.25 - 3.
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
            self.log.write('Color term fit failed! '
                           '(iteration 1, num_stars = {:d}, '
                           'max_stdev = {:.3f})'
                           .format(len(mag_diff), max(stdev_list)),
                           level=2, event=72,
                           solution_num=solution_num)
            return None

        cf = np.polyfit(cterm_list, stdev_list, 4)
        cf1d = np.poly1d(cf)
        extrema = cf1d.deriv().r
        cterm_extr = extrema[np.where(extrema.imag==0)].real
        der2 = cf1d.deriv(2)(cterm_extr)

        try:
            cterm_min = cterm_extr[np.where((der2 > 0) & (cterm_extr > -2.5) &
                                            (cterm_extr < 4.5))][0]
        except IndexError:
            self.log.write('Color term outside of allowed range!',
                           level=2, event=72, solution_num=solution_num)
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
        cterm_list = np.arange(33) * 0.25 - 3.
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
            self.log.write('Color term fit failed! '
                           '(iteration 2, num_stars = {:d}, '
                           'max_stdev = {:.3f})'
                           .format(len(mag_diff), max(stdev_list)),
                           level=2, event=72, solution_num=solution_num)
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

        if cf[0] < 0 or min(stdev_list) < 0.01 or min(stdev_list) > 2:
            self.log.write('Color term fit failed! '
                           '(iteration 2, num_stars = {:d}, cf[0] = {:f}, '
                           'min_stdev = {:.3f})'
                           .format(len(mag_diff), cf[0], min(stdev_list)),
                           level=2, event=72, solution_num=solution_num)
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

        if cf[0] < 0 or cterm < -2 or cterm > 4:
            if cf[0] < 0:
                self.log.write('Color term fit not reliable!',
                               level=2, event=72, solution_num=solution_num)
            else:
                self.log.write('Color term outside of allowed range '
                               '({:.3f})!'.format(cterm),
                               level=2, event=72, solution_num=solution_num)

            if cterm_min < -2 or cterm_min > 4:
                self.log.write('Color term from previous iteration '
                               'outside of allowed range ({:.3f})!'
                               ''.format(cterm_min),
                               level=2, event=72, solution_num=solution_num)
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
                           level=4, event=72, solution_num=solution_num)

        # Create dictionary for calibration results, if not exists
        if self.phot_calib is None:
            self.phot_calib = OrderedDict()
            self.phot_calib['solution_num'] = solution_num
            self.phot_calib['iteration'] = 0

        # Store color term result
        self.phot_calib['color_term'] = cterm
        self.phot_calib['color_term_error'] = cterm_err
        self.phot_calib['cterm_stdev_fit'] = stdev_fit
        self.phot_calib['cterm_stdev_min'] = stdev_min
        self.phot_calib['cterm_range_min'] = cterm_minval
        self.phot_calib['cterm_range_max'] = cterm_maxval
        self.phot_calib['cterm_iterations'] = iteration
        self.phot_calib['cterm_num_stars'] = num_stars

        self.log.write('Plate color term (solution {:d}): {:.3f} ({:.3f})'
                       .format(solution_num, cterm, cterm_err),
                       level=4, event=72, solution_num=solution_num)

    def calibrate_photometry_gaia(self, solution_num=None, iteration=1):
        """
        Calibrate extracted magnitudes with Gaia data.

        """

        num_solutions = self.plate_solution.num_solutions

        assert (solution_num is None or 
                (solution_num > 0 and solution_num <= num_solutions))

        self.log.write('Photometric calibration: solution {:d}, iteration {:d}'
                       .format(solution_num, iteration), level=3, event=70,
                       solution_num=solution_num)

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
                               level=2, event=70, solution_num=solution_num)
                return

        # Create dictionary for calibration results
        self.phot_calib = OrderedDict()

        # Create output directory, if missing
        if self.write_phot_dir and not os.path.isdir(self.write_phot_dir):
            self.log.write('Creating output directory {}'
                           .format(self.write_phot_dir), level=4, event=70,
                           solution_num=solution_num)
            os.makedirs(self.write_phot_dir)

        if self.write_phot_dir:
            fn_cterm = os.path.join(self.write_phot_dir,
                                    '{}_cterm.txt'.format(self.basefn))
            fcterm = open(fn_cterm, 'wb')
            fn_caldata = os.path.join(self.write_phot_dir, 
                                      '{}_caldata.txt'.format(self.basefn))
            fcaldata = open(fn_caldata, 'wb')

        # Select sources for photometric calibration
        self.log.write('Selecting sources for photometric calibration', 
                       level=3, event=71, solution_num=solution_num,
                       double_newline=False)

        if solution_num is None:
            solution_num = 1

        self.phot_calib['solution_num'] = solution_num
        self.phot_calib['iteration'] = iteration

        # Store number of Gaia DR2 objects matched with the current solution
        bgaia = (self.sources['solution_num'] == solution_num)
        self.phot_calib['num_gaia_edr3'] = bgaia.sum()

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
                    ~self.sources['gaiaedr3_bpmag'].mask &
                    ~self.sources['gaiaedr3_rpmag'].mask &
                    (self.sources['gaiaedr3_bp_rp'].filled(99.) <= 2) &
                    (self.sources['gaiaedr3_neighbors'] == 1))

        num_calstars = cal_mask.sum()
        self.phot_calib['num_candidate_stars'] = num_calstars

        if num_calstars == 0:
            self.log.write('No stars for photometric calibration',
                           level=2, event=71, solution_num=solution_num)
            return

        self.log.write('Found {:d} calibration-star candidates with '
                       'Gaia magnitudes on the plate'
                       .format(num_calstars), level=4, event=71,
                       solution_num=solution_num)

        if num_calstars < 10:
            self.log.write('Too few calibration stars on the plate!',
                           level=2, event=71, solution_num=solution_num)
            return

        # Evaluate color term

        if iteration == 1:
            self.log.write('Determining color term using annular bins 1-3', 
                           level=3, event=72, solution_num=solution_num)
            cterm_mask = cal_mask & (self.sources['annular_bin'] <= 3)
        else:
            self.log.write('Determining color term using annular bins 1-8', 
                           level=3, event=72, solution_num=solution_num)
            cterm_mask = cal_mask & (self.sources['annular_bin'] <= 8)

        self.evaluate_color_term(self.sources[cterm_mask],
                                 solution_num=solution_num)

        # If color term was not determined, we need to terminate the
        # calibration
        if 'color_term' not in self.phot_calib:
            self.log.write('Cannot continue photometric calibration without '
                           'color term', level=2, event=72,
                           solution_num=solution_num)
            return

        cterm = self.phot_calib['color_term']
        cterm_err = self.phot_calib['color_term_error']

        # Use stars in all annular bins
        self.log.write('Photometric calibration using annular bins 1-9', 
                       level=3, event=73, solution_num=solution_num)

        # Select stars with unique plate mag values
        plate_mag = self.sources['mag_auto'][cal_mask].data
        plate_mag_u,uind = np.unique(plate_mag, return_index=True)
        ind_calibstar_u = np.where(cal_mask)[0][uind]
        #cal_u_mask = np.zeros_like(cal_mask)
        #cal_u_mask[np.where(cal_mask)[0][uind]] = True
        num_cal_u = len(plate_mag_u)

        self.log.write('{:d} stars with unique magnitude'
                       .format(num_cal_u), double_newline=False,
                       level=4, event=73, solution_num=solution_num)

        if num_cal_u < 10:
            self.log.write('Too few stars with unique magnitude!',
                           double_newline=False, level=2, event=73,
                           solution_num=solution_num)
            return

        plate_mag_u = self.sources['mag_auto'][ind_calibstar_u].data
        cat_bmag_u = self.sources['gaiaedr3_bpmag'][ind_calibstar_u].data
        cat_vmag_u = self.sources['gaiaedr3_rpmag'][ind_calibstar_u].data
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
                    self.log.write('{:d} faint stars eliminated as outliers'
                                   .format(len(ind_faintout)),
                                   double_newline=False,
                                   level=4, event=73, solution_num=solution_num)

                self.log.write('Outlier elimination stopped due to a long gap '
                               'in magnitudes!', double_newline=False,
                               level=2, event=73, solution_num=solution_num)
                break

            if len(ind_good) < 10:
                self.log.write('Outlier elimination stopped '
                               'due to insufficient number of stars left!',
                               double_newline=False, level=2, event=73,
                               solution_num=solution_num)
                break

        num_outliers = len(ind_outliers)
        self.log.write('{:d} outliers eliminated'.format(num_outliers),
                       double_newline=False, level=4, event=73,
                       solution_num=solution_num)
        ind_good = np.setdiff1d(np.arange(len(plate_mag_u)), 
                                ind_outliers)
        self.log.write('{:d} stars after outlier elimination'
                       .format(len(ind_good)), double_newline=False,
                       level=4, event=73, solution_num=solution_num)

        if len(ind_good) < 10:
            self.log.write('Too few calibration stars ({:d}) after outlier '
                           'elimination!'.format(len(ind_good)),
                           double_newline=False, level=2, event=73,
                           solution_num=solution_num)
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

        self.log.write('{:d} calibration stars brighter than limiting magnitude'
                       .format(num_valid), double_newline=False, level=4,
                       event=73, solution_num=solution_num)

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
                       level=3, event=74, solution_num=solution_num)

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
        self.sources['natmag'][sol_mask] = s(mag_auto_sol)
        self.sources['natmag_plate'][sol_mask] = s(mag_auto_sol)
        self.sources['natmag_error'][sol_mask] = s_rmse(mag_auto_sol)

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
            self.sources['phot_range_flags'][ind] = 1
            self.sources['natmag_error'][ind] = s_rmse(plate_mag_brightest)

        brange = (mag_auto_sol > plate_mag_lim)
        ind = np.where(sol_mask)[0][brange]

        if brange.sum() > 0:
            self.sources['phot_range_flags'][ind] = 2
            self.sources['natmag_error'][ind] = s_rmse(plate_mag_lim)

        # Select stars with known external photometry
        bgaia = (sol_mask &
                 ~self.sources['gaiaedr3_bpmag'].mask &
                 ~self.sources['gaiaedr3_rpmag'].mask)

        if bgaia.sum() > 0:
            bp_rp = self.sources['gaiaedr3_bp_rp'][bgaia]
            bp_rp_err = 0.

            self.sources['rpmag'][bgaia] = (self.sources['natmag'][bgaia]
                                            - cterm * bp_rp)
            self.sources['bpmag'][bgaia] = (self.sources['natmag'][bgaia]
                                            - (cterm - 1.) * bp_rp)
            rpmagerr = np.sqrt(self.sources['natmag_error'][bgaia]**2 +
                               (cterm_err * bp_rp)**2 +
                               (cterm * bp_rp_err)**2)
            bpmagerr = np.sqrt(self.sources['natmag_error'][bgaia]**2 +
                               (cterm_err * bp_rp)**2 +
                               ((cterm - 1.) * bp_rp_err)**2)
            self.sources['rpmag_error'][bgaia] = rpmagerr
            self.sources['bpmag_error'][bgaia] = bpmagerr

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
                           level=4, event=73, solution_num=solution_num)

        if self.write_phot_dir:
            fcaldata.close()
