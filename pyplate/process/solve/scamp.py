"""Running SCAMP to refine astrometric solutions, and reading its
results."""
import os
import subprocess as sp
import warnings

import numpy as np
import unidecode
from astropy import units as u
from astropy.io import fits, votable
from astropy.table import Table

from ..context import ProcessContext
from .solution import Solution


def new_scampref():
    """
    Create HDUList for SCAMP reference catalogue.

    Returns
    -------
    hdulist : an astropy.io.fits.HDUList object

    """

    hdulist = fits.HDUList()

    hdu = fits.PrimaryHDU()
    hdulist.append(hdu)

    hdummy = hdu.header.copy()
    hdummystr = hdummy.tostring()
    col = fits.Column(name='Field Header Card', format='2880A',
                      array=[hdummystr])

    try:
        tbl = fits.BinTableHDU.from_columns([col])
    except AttributeError:
        tbl = fits.new_table([col])

    tbl.header.set('EXTNAME', 'LDAC_IMHEAD', 'table name', after='TFIELDS')
    tbl.header.set('TDIM1', '(80, 36)', after='TFORM1')
    hdulist.append(tbl)

    col1 = fits.Column(name='X_WORLD', format='1D', unit='deg')
    col2 = fits.Column(name='Y_WORLD', format='1D', unit='deg')
    col3 = fits.Column(name='ERRA_WORLD', format='1E', unit='deg')
    col4 = fits.Column(name='ERRB_WORLD', format='1E', unit='deg')
    col5 = fits.Column(name='MAG', format='1E', unit='mag', disp='F8.4')
    col6 = fits.Column(name='MAGERR', format='1E', unit='mag', disp='F8.4')
    col7 = fits.Column(name='OBSDATE', format='1D', unit='yr',
                       disp='F13.8')

    try:
        tbl = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5,
                                             col6, col7])
    except AttributeError:
        tbl = fits.new_table([col1, col2, col3, col4, col5, col6, col7])

    tbl.header.set('EXTNAME', 'LDAC_OBJECTS', 'table name',
                   after='TFIELDS')
    hdulist.append(tbl)

    return hdulist


def write_reference_catalog(ctx: ProcessContext, astref_table: Table,
                            basefn_solution: str) -> None:
    """Write the reference stars of a solution to
    ``<basefn_solution>_scampref.cat``, the reference catalogue of
    :func:`refine`."""
    numref = len(astref_table)
    scampref = new_scampref()
    hduref = fits.BinTableHDU.from_columns(scampref[2].columns, nrows=numref)
    hduref.data.field('X_WORLD')[:] = astref_table['ra']
    hduref.data.field('Y_WORLD')[:] = astref_table['dec']
    hduref.data.field('ERRA_WORLD')[:] = np.zeros(numref) + 1./3600.
    hduref.data.field('ERRB_WORLD')[:] = np.zeros(numref) + 1./3600.
    hduref.data.field('MAG')[:] = astref_table['mag']
    hduref.data.field('MAGERR')[:] = np.zeros(numref) + 0.1
    hduref.data.field('OBSDATE')[:] = np.zeros(numref) + 2000.
    scampref[2].data = hduref.data
    scampref_file = os.path.join(ctx.work_dir,
                                 '{}_scampref.cat'.format(basefn_solution))
    scampref.writeto(scampref_file, overwrite=True)


def write_tan_ahead(ctx: ProcessContext, basefn_solution: str) -> None:
    """Write the TAN projection of the astrometry.net solution
    (``<basefn_solution>.wcs``) to ``<basefn_solution>.ahead``, the
    starting point of the first SCAMP run."""
    fn_tan = '{}_tan.wcs'.format(basefn_solution)
    cmd = ctx.cfg.programs.wcs_to_tan_path
    cmd += ' -w {}.wcs'.format(basefn_solution)
    cmd += ' -x 1'
    cmd += ' -y 1'
    cmd += ' -W {:f}'.format(ctx.imwidth)
    cmd += ' -H {:f}'.format(ctx.imheight)
    cmd += ' -N 20'
    cmd += ' -o {}'.format(fn_tan)
    ctx.log.write('Subprocess: {}'.format(cmd), level=5, event=32)
    sp.call(cmd, shell=True, stdout=ctx.log.handle,
            stderr=ctx.log.handle, cwd=ctx.work_dir)

    tanhead = fits.getheader(os.path.join(ctx.work_dir, fn_tan))
    tanhead.set('NAXIS', 2)
    ahead = fits.Header()
    ahead.set('NAXIS', 2)
    ahead.set('NAXIS1', ctx.imwidth)
    ahead.set('NAXIS2', ctx.imheight)
    ahead.set('IMAGEW', ctx.imwidth)
    ahead.set('IMAGEH', ctx.imheight)
    ahead.set('CTYPE1', 'RA---TAN')
    ahead.set('CTYPE2', 'DEC--TAN')
    ahead.set('CRPIX1', (ctx.imwidth + 1.) / 2.)
    ahead.set('CRPIX2', (ctx.imheight + 1.) / 2.)
    ahead.set('CRVAL1', tanhead['CRVAL1'])
    ahead.set('CRVAL2', tanhead['CRVAL2'])
    ahead.set('CD1_1', tanhead['CD1_1'])
    ahead.set('CD1_2', tanhead['CD1_2'])
    ahead.set('CD2_1', tanhead['CD2_1'])
    ahead.set('CD2_2', tanhead['CD2_2'])
    aheadfile = os.path.join(ctx.work_dir,
                             '{}.ahead'.format(basefn_solution))
    ahead.totextfile(aheadfile, endcard=True, overwrite=True)


def refine(ctx: ProcessContext, basefn_solution: str, catalog: str,
           crossid_radius: u.Quantity, distort: int, match: bool,
           event: int) -> None:
    """Run SCAMP on a source catalogue with the reference stars of a
    solution.

    Parameters
    ----------
    ctx
        Context of the process.
    basefn_solution
        Base name of the files of the solution; the reference catalogue is
        ``<basefn_solution>_scampref.cat``.
    catalog
        Base name of the source catalogue (``<catalog>.cat``). SCAMP reads
        the starting solution from ``<catalog>.ahead`` and writes
        ``<catalog>.head`` and ``<catalog>_scamp.xml``.
    crossid_radius
        Cross-identification radius.
    distort
        Degree of the distortion polynomial.
    match
        Whether SCAMP matches the pattern first (with small limits on the
        scale, angle and position errors).
    event
        Event number of the log entry.

    """
    cmd = ctx.cfg.programs.scamp_path
    cmd += ' -c {}_scamp.conf {}.cat'.format(ctx.basefn, catalog)
    cmd += ' -ASTREF_CATALOG FILE'
    cmd += ' -ASTREFCAT_NAME {}_scampref.cat'.format(basefn_solution)
    cmd += ' -ASTREFCENT_KEYS X_WORLD,Y_WORLD'
    cmd += ' -ASTREFERR_KEYS ERRA_WORLD,ERRB_WORLD,ERRTHETA_WORLD'
    cmd += ' -ASTREFMAG_KEY MAG'
    cmd += ' -ASTRCLIP_NSIGMA 1.5'
    cmd += ' -FLAGS_MASK 0x00ff'
    cmd += ' -SN_THRESHOLDS 20.0,100.0'

    if match:
        cmd += ' -MATCH Y'
        cmd += ' -PIXSCALE_MAXERR 1.01'
        cmd += ' -POSANGLE_MAXERR 0.1'
        cmd += ' -POSITION_MAXERR 0.2'
    else:
        cmd += ' -MATCH N'

    cmd += ' -CROSSID_RADIUS {:.2f}'.format(crossid_radius
                                            .to(u.arcsec).value)
    cmd += ' -DISTORT_DEGREES {:d}'.format(distort)
    cmd += ' -PROJECTION_TYPE TPV'
    cmd += ' -STABILITY_TYPE EXPOSURE'
    cmd += ' -SOLVE_PHOTOM N'
    cmd += ' -WRITE_XML Y'
    cmd += ' -XML_NAME {}_scamp.xml'.format(catalog)
    cmd += ' -VERBOSE_TYPE LOG'
    cmd += ' -CHECKPLOT_TYPE NONE'
    ctx.log.write('Subprocess: {}'.format(cmd), level=5, event=event)
    sp.call(cmd, shell=True, stdout=ctx.log.handle,
            stderr=ctx.log.handle, cwd=ctx.work_dir)


def parse_results(ctx: ProcessContext, catalog: str
                  ) -> tuple[fits.Header, fits.Header, Table]:
    """Read the results of a SCAMP run of :func:`refine`.

    Returns
    -------
    header_scamp
        The header written by SCAMP.
    header_wcs
        The full WCS header of the image made from it.
    scamp_stats
        The statistics of the XML output.

    Raises
    ------
    FileNotFoundError
        If SCAMP wrote no header.

    """
    fn_scamphead = os.path.join(ctx.work_dir, '{}.head'.format(catalog))

    with open(fn_scamphead, 'r') as f:
        scamphead_str = f.read()

    # Get rid of non-ascii characters
    scamphead_str = unidecode.unidecode(scamphead_str)

    # Create FITS header from string
    header_scamp = fits.Header.fromstring(scamphead_str, sep='\n')

    header_wcs = fits.PrimaryHDU().header
    header_wcs.set('NAXIS', 2)
    header_wcs.set('NAXIS1', ctx.imwidth)
    header_wcs.set('NAXIS2', ctx.imheight)
    header_wcs.set('IMAGEW', ctx.imwidth)
    header_wcs.set('IMAGEH', ctx.imheight)
    header_wcs.extend(header_scamp)

    # Fix SCAMP header if TPV projection is not specified
    if 'PV1_1' in header_wcs and header_wcs['CTYPE1'] == 'RA---TAN':
        header_wcs.set('CTYPE1', 'RA---TPV')
        header_wcs.set('CTYPE2', 'DEC--TPV')

    # Read SCAMP XML output
    fn_xml = os.path.join(ctx.work_dir, '{}_scamp.xml'.format(catalog))
    warnings.filterwarnings('ignore', message='.*W42.*',
                            category=votable.exceptions.VOTableSpecWarning)
    scamp_stats = votable.parse_single_table(fn_xml, verify='warn').to_table()
    return header_scamp, header_wcs, scamp_stats


def set_statistics(solution: Solution, scamp_stats: Table, distort: int,
                   iteration: int) -> None:
    """Copy the astrometric errors and the fit statistics of a SCAMP run
    to the solution."""
    scamp_sigmas = scamp_stats['AstromSigma_Reference'][0,:].quantity
    solution.scamp_sigma_1 = scamp_sigmas[0]
    solution.scamp_sigma_2 = scamp_sigmas[1]
    solution.scamp_sigma_mean = np.sqrt(scamp_sigmas[0]**2 +
                                        scamp_sigmas[1]**2)
    solution.scamp_chi2 = scamp_stats['Chi2_Reference'][0]
    solution.scamp_ndeg = scamp_stats['NDeg_Reference'][0]
    solution.scamp_distort = distort
    solution.scamp_iteration = iteration
