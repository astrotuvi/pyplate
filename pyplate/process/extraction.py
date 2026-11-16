"""Source extraction with SExtractor and PSFEx.

All runs read the inverted image ``<basefn>_inverted.fits`` and write their
files to the scratch directory.
"""
import os
import subprocess as sp
import xml.etree.ElementTree as ET

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .context import ProcessContext
from .sources import SourceTable


def write_sextractor_config(ctx: ProcessContext, params: list[str],
                            config: str, event: int) -> str:
    """
    Write the SExtractor parameter file (``<basefn>_sextractor.param``,
    one output parameter per line) and the configuration file
    (``<basefn>_sextractor.conf``).

    Parameters
    ----------
    ctx
        Context of the process.
    params
        Output parameters of SExtractor.
    config
        Configuration, without ``PARAMETERS_NAME``, which is added.
    event
        Event number of the log entries.

    Returns
    -------
    str
        Name of the configuration file.

    """
    fn_sex_param = ctx.basefn + '_sextractor.param'

    with open(os.path.join(ctx.work_dir, fn_sex_param), 'w') as fconf:
        fconf.write('\n'.join(params))

    cnf = config.replace('@PARAMETERS_NAME@',
                         'PARAMETERS_NAME  {}\n'.format(fn_sex_param))
    fn_sex_conf = ctx.basefn + '_sextractor.conf'
    ctx.log.write('Writing SExtractor configuration file {}'
                  .format(fn_sex_conf), level=4, event=event)
    ctx.log.write('SExtractor configuration file:\n{}'
                  .format(cnf), level=5, event=event)

    with open(os.path.join(ctx.work_dir, fn_sex_conf), 'w') as fconf:
        fconf.write(cnf)

    return fn_sex_conf


def run_sextractor(ctx: ProcessContext, params: list[str], config: str,
                   event: int) -> None:
    """Write the SExtractor files (see :func:`write_sextractor_config`;
    ``config`` marks the place of ``PARAMETERS_NAME`` with
    ``@PARAMETERS_NAME@``) and run SExtractor on the inverted image."""
    fn_sex_conf = write_sextractor_config(ctx, params, config, event)
    cmd = ctx.cfg.programs.sextractor_path
    cmd += ' %s_inverted.fits' % ctx.basefn
    cmd += ' -c %s' % fn_sex_conf
    ctx.log.write('Subprocess: {}'.format(cmd), level=5, event=event)
    sp.call(cmd, shell=True, stdout=ctx.log.handle,
            stderr=ctx.log.handle, cwd=ctx.work_dir)
    ctx.log.write('', timestamp=False, double_newline=False)


def _threshold_config(threshold: float, absolute: bool) -> str:
    """Detection threshold in ADU if absolute, otherwise in sigmas."""
    cnf = 'DETECT_THRESH    {:f}\n'.format(threshold)
    cnf += 'ANALYSIS_THRESH  {:f}\n'.format(threshold)

    if absolute:
        cnf += 'THRESH_TYPE      ABSOLUTE\n'

    return cnf


def _filter_config(filter_path: str | None) -> str:
    """Detection filter, if filter_path is given."""
    if filter_path:
        return ('FILTER           Y\n'
                'FILTER_NAME      {}\n'.format(filter_path))

    return 'FILTER           N\n'


def measure_sky(ctx: ProcessContext) -> tuple[float, float] | None:
    """
    Run SExtractor without detections to measure the sky background.

    Returns
    -------
    tuple or None
        The mean sky level and its standard deviation (ADU), or None if
        SExtractor did not report them.

    """
    cnf = _threshold_config(60000, absolute=True)
    cnf += 'FILTER           N\n'
    cnf += 'SATUR_LEVEL      65000.0\n'
    cnf += 'BACKPHOTO_TYPE   LOCAL\n'
    cnf += 'MAG_ZEROPOINT    25.0\n'
    cnf += '@PARAMETERS_NAME@'
    cnf += 'CATALOG_TYPE     FITS_1.0\n'
    cnf += 'CATALOG_NAME     {}_sky.cat\n'.format(ctx.basefn)
    cnf += 'WRITE_XML        Y\n'
    cnf += 'XML_NAME         sex.xml\n'
    run_sextractor(ctx, ['NUMBER'], cnf, event=21)

    tree = ET.parse(os.path.join(ctx.work_dir, 'sex.xml'))
    root = tree.getroot()

    if root[1][4][15][11].attrib['name'] == 'Background_Mean':
        sky = float(root[1][4][15][19][0][0][8].text)
        sky_sigma = float(root[1][4][15][19][0][0][9].text)
        return sky, sky_sigma

    return None


def extract_catalog(ctx: ProcessContext, threshold: float, absolute: bool,
                    filter_path: str | None) -> None:
    """Extract the sources without a PSF model into ``<basefn>.cat``.

    Parameters
    ----------
    ctx
        Context of the process.
    threshold
        Detection threshold, in ADU if absolute, otherwise in sigmas.
    absolute
        Whether the threshold is in ADU.
    filter_path
        Detection filter, or None for no filter.

    """
    params = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'A_IMAGE', 'B_IMAGE',
              'THETA_IMAGE', 'ERRA_IMAGE', 'ERRB_IMAGE', 'ERRTHETA_IMAGE',
              'ELONGATION', 'XPEAK_IMAGE', 'YPEAK_IMAGE', 'MAG_AUTO',
              'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_ISO',
              'MAGERR_ISO', 'FLUX_ISO', 'FLUXERR_ISO', 'FLUX_MAX',
              'FLUX_RADIUS', 'FWHM_IMAGE', 'ISOAREA_IMAGE', 'BACKGROUND',
              'FLAGS']
    cnf = _threshold_config(threshold, absolute)
    cnf += _filter_config(filter_path)
    cnf += 'DEBLEND_NTHRESH  64\n'
    cnf += 'DEBLEND_MINCONT  0.0001\n'
    cnf += 'SATUR_LEVEL      65000.0\n'
    cnf += 'BACKPHOTO_TYPE   LOCAL\n'
    cnf += 'CLEAN            N\n'
    cnf += 'MAG_ZEROPOINT    25.0\n'
    cnf += '@PARAMETERS_NAME@'
    cnf += 'CATALOG_TYPE     FITS_1.0\n'
    cnf += 'CATALOG_NAME     {}.cat\n'.format(ctx.basefn)
    cnf += 'NTHREADS         0\n'
    run_sextractor(ctx, params, cnf, event=22)


def extract_psf_model_sources(ctx: ProcessContext, threshold: float,
                              absolute: bool, filter_path: str | None
                              ) -> int:
    """Extract the sources for the PSF model into ``<basefn>_psfex.cat``
    (with the same arguments as :func:`extract_catalog`).

    Returns
    -------
    int
        Number of extracted sources.

    """
    params = ['VIGNET(120,120)', 'X_IMAGE', 'Y_IMAGE', 'MAG_AUTO',
              'FLUX_AUTO', 'FLUXERR_AUTO', 'SNR_WIN', 'FLUX_RADIUS',
              'ELONGATION', 'FLAGS']
    cnf = _threshold_config(threshold, absolute)
    cnf += _filter_config(filter_path)
    cnf += 'SATUR_LEVEL      65000.0\n'
    cnf += 'BACKPHOTO_TYPE   LOCAL\n'
    cnf += 'MAG_ZEROPOINT    25.0\n'
    cnf += '@PARAMETERS_NAME@'
    cnf += 'CATALOG_TYPE     FITS_LDAC\n'
    cnf += 'CATALOG_NAME     {}_psfex.cat\n'.format(ctx.basefn)
    run_sextractor(ctx, params, cnf, event=23)

    fn_psfex_cat = os.path.join(ctx.work_dir, ctx.basefn + '_psfex.cat')
    hcat = fits.getheader(fn_psfex_cat, 2)
    return hcat['NAXIS2']


def run_psfex(ctx: ProcessContext) -> None:
    """Build the PSF model ``<basefn>_psfex.psf`` with PSFEx."""
    ctx.log.write('Running PSFEx', level=3, event=24)
    psfex_ver = (sp.check_output([ctx.cfg.programs.psfex_path, '-v']).strip()
                 .decode('utf-8'))
    ctx.log.write('Using {}'.format(psfex_ver), level=4, event=24)

    cnf = 'PHOTFLUX_KEY       FLUX_AUTO\n'
    cnf += 'PHOTFLUXERR_KEY    FLUXERR_AUTO\n'
    cnf += 'PSFVAR_KEYS        X_IMAGE,Y_IMAGE\n'
    cnf += 'PSFVAR_GROUPS      1,1\n'
    cnf += 'PSFVAR_DEGREES     3\n'
    cnf += 'SAMPLE_FWHMRANGE   3.0,50.0\n'
    cnf += 'SAMPLE_VARIABILITY 3.0\n'
    cnf += 'PSF_SIZE           50,50\n'
    cnf += 'CHECKPLOT_TYPE     ellipticity\n'
    cnf += 'CHECKPLOT_NAME     ellipticity\n'
    cnf += 'CHECKIMAGE_TYPE    SNAPSHOTS\n'
    cnf += 'CHECKIMAGE_NAME    snap.fits\n'
    cnf += 'XML_NAME           {}_psfex.xml\n'.format(ctx.basefn)
    cnf += 'VERBOSE_TYPE       LOG\n'
    cnf += 'NTHREADS           2\n'

    fn_psfex_conf = ctx.basefn + '_psfex.conf'
    ctx.log.write('Writing PSFEx configuration file {}'
                  .format(fn_psfex_conf), level=4, event=24)
    ctx.log.write('PSFEx configuration file:\n{}'
                  .format(cnf), level=5, event=24)

    with open(os.path.join(ctx.work_dir, fn_psfex_conf), 'w') as fconf:
        fconf.write(cnf)

    cmd = ctx.cfg.programs.psfex_path
    cmd += ' %s_psfex.cat' % ctx.basefn
    cmd += ' -c %s' % fn_psfex_conf
    ctx.log.write('Subprocess: {}'.format(cmd), level=5, event=24)
    sp.call(cmd, shell=True, stdout=ctx.log.handle,
            stderr=ctx.log.handle, cwd=ctx.work_dir)
    ctx.log.write('', timestamp=False, double_newline=False)


def extract_psf_catalog(ctx: ProcessContext, threshold: float,
                        absolute: bool, filter_path: str | None) -> None:
    """Extract the sources with the PSF model into ``<basefn>.cat-psf``
    (with the same arguments as :func:`extract_catalog`)."""
    params = ['XPEAK_IMAGE', 'YPEAK_IMAGE', 'XPSF_IMAGE', 'YPSF_IMAGE',
              'ERRAPSF_IMAGE', 'ERRBPSF_IMAGE', 'ERRTHETAPSF_IMAGE']
    cnf = _threshold_config(threshold, absolute)
    cnf += _filter_config(filter_path)
    cnf += 'SATUR_LEVEL      65000.0\n'
    cnf += 'BACKPHOTO_TYPE   LOCAL\n'
    cnf += 'MAG_ZEROPOINT    25.0\n'
    cnf += '@PARAMETERS_NAME@'
    cnf += 'CATALOG_TYPE     FITS_1.0\n'
    cnf += 'CATALOG_NAME     {}.cat-psf\n'.format(ctx.basefn)
    cnf += 'PSF_NAME         {}_psfex.psf\n'.format(ctx.basefn)
    cnf += 'NTHREADS         0\n'
    run_sextractor(ctx, params, cnf, event=25)


def psf_model_thresholds(ctx: ProcessContext
                         ) -> tuple[float | None, float | None]:
    """Peak fluxes (ADU) of the ``[Extraction] min_model_sources``-th and
    ``max_model_sources``-th brightest sources in ``<basefn>.cat``: the
    highest and lowest useful thresholds for the PSF-model sources (None
    if there are not that many sources)."""
    fn_cat = os.path.join(ctx.work_dir, ctx.basefn + '.cat')
    t = Table.read(fn_cat)
    num_sources = len(t)
    flux_peak = np.sort(t['FLUX_MAX'])[::-1]
    extraction = ctx.cfg.extraction

    if extraction.min_model_sources < num_sources:
        th_max = flux_peak[extraction.min_model_sources]
    else:
        th_max = None

    if extraction.max_model_sources < num_sources:
        th_min = flux_peak[extraction.max_model_sources]
    else:
        th_min = None

    return th_min, th_max


def build_source_table(ctx: ProcessContext) -> SourceTable:
    """Read the sources from the SExtractor catalogue ``<basefn>.cat``
    into a new source table."""
    xycat = fits.open(os.path.join(ctx.work_dir, ctx.basefn + '.cat'))
    num_sources = len(xycat[1].data)

    # Initialise source table
    sources = SourceTable()
    sources.populate(num_sources=num_sources)
    sources.plate = ctx.plate
    sources.log = ctx.log

    # Copy values from the SExtractor catalog, xycat
    sources.copy_from_sextractor(xycat)

    # Calculate square root of isoarea
    sources['sqrt_isoarea'] = np.sqrt(sources['isoarea'])
    return sources
