"""Running astrometry.net (``solve-field``)."""
from __future__ import annotations
import os
import subprocess as sp

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

from ..context import ProcessContext


def solve_field(ctx: ProcessContext, sources: np.ndarray,
                basefn_solution: str, ref_year: int, sip: int,
                pixel_scale: u.Quantity | None, many_solutions: bool,
                cpulimit: int) -> fits.Header | None:
    """Find an astrometric solution with astrometry.net.

    Writes the sources to ``<basefn_solution>.xy`` and the backend
    configuration (the index files of ``ref_year``), and runs
    ``solve-field``. If it found a solution without the SIP distortions
    that were asked for, it is run again without tweaking.
    ``solve-field`` also writes the SCAMP catalogue
    (``<basefn_solution>.cat``) and the correspondences of the solution
    (``<basefn_solution>.corr``).

    Parameters
    ----------
    ctx
        Context of the process.
    sources
        The sources to solve with (``x_source``, ``y_source``,
        ``mag_auto``, ``flux_auto``).
    basefn_solution
        Base name of the files of the solution.
    ref_year
        Year of the index files.
    sip
        SIP distortion order (0 for none).
    pixel_scale
        If given, the pixel scale is searched within 5% of it.
    many_solutions
        Accept solutions with lower odds (for plates with many
        exposures).
    cpulimit
        CPU time limit in seconds.

    Returns
    -------
    fits.Header or None
        The WCS header of the solution, or None if none was found.

    """
    fn_xy = '{}.xy'.format(basefn_solution)
    fn_match = '{}.match'.format(basefn_solution)
    fn_corr = '{}.corr'.format(basefn_solution)

    # Prepare FITS file with a list of sources and write the file to disk
    xycat = Table()
    xycat['X_IMAGE'] = sources['x_source']
    xycat['Y_IMAGE'] = sources['y_source']
    xycat['MAG_AUTO'] = sources['mag_auto']
    xycat['FLUX'] = sources['flux_auto']
    xycat['X_IMAGE'].unit = 'pixel'
    xycat['Y_IMAGE'].unit = 'pixel'
    xycat['MAG_AUTO'].unit = 'mag'
    xycat['FLUX'].unit = 'count'

    fn_xy = os.path.join(ctx.work_dir, fn_xy)
    xycat.write(fn_xy, format='fits', overwrite=True)

    # Write backend config file
    fconf = open(os.path.join(ctx.work_dir,
                              ctx.basefn + '_backend.cfg'), 'w')
    index_path = os.path.join(ctx.cfg.files.index_dir,
                              'index_{:d}'.format(ref_year))
    fconf.write('add_path {}\n'.format(index_path))
    fconf.write('autoindex\n')
    fconf.write('inparallel\n')
    fconf.close()

    # Construct the solve-field call
    cmd = ctx.cfg.programs.solve_field_path
    cmd += ' {}'.format(fn_xy)
    cmd += ' --width {:d}'.format(ctx.imwidth)
    cmd += ' --height {:d}'.format(ctx.imheight)
    cmd += ' --x-column X_IMAGE'
    cmd += ' --y-column Y_IMAGE'
    cmd += ' --sort-column MAG_AUTO'
    cmd += ' --sort-ascending'
    cmd += ' --backend-config {}_backend.cfg'.format(ctx.basefn)

    cmd += ' --crpix-center'
    cmd += ' --scamp {}.cat'.format(basefn_solution)
    cmd += ' --scamp-config {}_scamp.conf'.format(ctx.basefn)
    cmd += ' --no-plots'
    cmd += ' --out {}'.format(basefn_solution)
    cmd += ' --match {}'.format(fn_match)
    cmd += ' --rdls none'
    cmd += ' --corr {}'.format(fn_corr)
    cmd += ' --overwrite'
    cmd += ' --pixel-error 3'
    cmd += ' --uniformize 100'

    if pixel_scale is not None:
        scale_low = 0.95 * pixel_scale
        scale_high = 1.05 * pixel_scale
        cmd += ' --scale-units arcsecperpix'
        cmd += ' --scale-low {:.3f}'.format(scale_low.value)
        cmd += ' --scale-high {:.3f}'.format(scale_high.value)

    if many_solutions:
        cmd += ' --odds-to-solve 1e8'

    cmd += ' --cpulimit {:d}'.format(cpulimit)
    cmd_no_tweak = cmd + ' --no-tweak'

    if sip > 0:
        cmd += ' --tweak-order {:d}'.format(sip)
    else:
        cmd += ' --no-tweak'

    ctx.log.write('Subprocess: {}'.format(cmd), level=5, event=32)
    sp.call(cmd, shell=True, stdout=ctx.log.handle,
            stderr=ctx.log.handle, cwd=ctx.work_dir)
    ctx.log.write('', timestamp=False, double_newline=False)

    # Check the result of solve-field
    fn_solved = os.path.join(ctx.work_dir, '{}.solved'.format(basefn_solution))
    fn_wcs = os.path.join(ctx.work_dir, '{}.wcs'.format(basefn_solution))

    if os.path.exists(fn_solved) and os.path.exists(fn_wcs):
        wcshead = fits.getheader(fn_wcs)

        # If SIP distortions were not computed, then repeat solving
        # without tweaking (plane TAN projection)
        if sip > 0 and wcshead['CTYPE1'] == 'RA---TAN':
            ctx.log.write('Repeating astrometric solving without SIP distortions',
                          level=4, event=32)
            ctx.log.write('Subprocess: {}'.format(cmd_no_tweak),
                          level=5, event=32)
            sp.call(cmd_no_tweak, shell=True, stdout=ctx.log.handle,
                    stderr=ctx.log.handle, cwd=ctx.work_dir)
            ctx.log.write('', timestamp=False, double_newline=False)

    if os.path.exists(fn_solved) and os.path.exists(fn_wcs):
        return fits.getheader(fn_wcs)

    return None


def read_correspondences(ctx: ProcessContext, basefn_solution: str) -> Table:
    """Read the sources and index stars of a solution
    (``<basefn_solution>.corr``)."""
    return Table.read(os.path.join(ctx.work_dir,
                                   '{}.corr'.format(basefn_solution)))
