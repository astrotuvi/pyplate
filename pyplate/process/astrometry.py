import os
import subprocess as sp
import numpy as np
from astropy.table import Table, vstack


class AstrometryNetIndex:
    """
    Astrometry.net index class

    """

    def __init__(self, *args):
        self.build_index_path = 'build-astrometry-index'
        self.index_dir = './'
        
        if len(args) == 1:
            self.index_dir = args[0]

    def query_gaia(self):
        """
        Query Gaia EDR3 catalogue for bright stars (G < 12) and 
        store results in a FITS file.

        """

        from astroquery.gaia import Gaia

        gaia_dir = self.index_dir

        # Query in two parts to overcome the 3-million-row limit
        query = ('SELECT ra,dec,pmra,pmdec,phot_g_mean_mag,'
                 'phot_bp_mean_mag,phot_rp_mean_mag '
                 'FROM gaiaedr3.gaia_source '
                 'WHERE phot_g_mean_mag<11.5 '
                 'AND astrometric_params_solved=31')
        fn_tab1 = os.path.join(gaia_dir, 'gaiaedr3_pyplate_1.fits')
        job = Gaia.launch_job_async(query, output_file=fn_tab1, 
                                    output_format='fits', dump_to_file=True)

        query = ('SELECT ra,dec,pmra,pmdec,phot_g_mean_mag,'
                 'phot_bp_mean_mag,phot_rp_mean_mag '
                 'FROM gaiaedr3.gaia_source '
                 'WHERE phot_g_mean_mag BETWEEN 11.5 AND 12 '
                 'AND astrometric_params_solved=31')
        fn_tab2 = os.path.join(gaia_dir, 'gaiaedr3_pyplate_2.fits')
        job = Gaia.launch_job_async(query, output_file=fn_tab2, 
                                    output_format='fits', dump_to_file=True)

        # Read two tables and concatenate them
        tab1 = Table.read(fn_tab1)
        tab2 = Table.read(fn_tab2)
        tab = vstack([tab1, tab2], join_type='exact')
        fn_tab = os.path.join(gaia_dir, 'gaiaedr3_pyplate.fits')
        tab.write(fn_tab, format='fits', overwrite=True)

        # Remove partial tables
        os.remove(fn_tab1)
        os.remove(fn_tab2)

    def create_index_year(self, year, max_scale=None, min_scale=None,
                          sort_by='Gmag'):
        """
        Create Astrometry.net index for a given epoch.

        """

        if not max_scale:
            max_scale = 16
        elif max_scale > 19:
            max_scale = 19
        elif max_scale < 1:
            max_scale = 1

        if not min_scale:
            min_scale = 7
        elif min_scale > 19:
            min_scale = 19
        elif min_scale < 1:
            min_scale = 1

        if sort_by != 'Gmag' and sort_by != 'BPmag' and sort_by != 'RPmag':
            sort_by = 'Gmag'

        fn_gaia = os.path.join(self.index_dir, 'gaiaedr3_pyplate.fits')
        gaia_tab = Table.read(fn_gaia)

        year_tab = Table()
        year_tab['RA'] = (gaia_tab['ra'] 
                          + (year - 2016.0 + 0.5)
                          * gaia_tab['pmra'] 
                          / np.cos(gaia_tab['dec'] * np.pi / 180.) 
                          / 3600000.)
        year_tab['Dec'] = (gaia_tab['dec']
                           + (year - 2016.0 + 0.5)
                           * gaia_tab['pmdec'] / 3600000.)

        if sort_by == 'Gmag':
            year_tab['Gmag'] = gaia_tab['phot_g_mean_mag']
        elif sort_by == 'BPmag':
            year_tab['BPmag'] = gaia_tab['phot_bp_mean_mag']
        elif sort_by == 'RPmag':
            year_tab['RPmag'] = gaia_tab['phot_rp_mean_mag']

        fn_year = os.path.join(self.index_dir, 
                               'gaiaedr3_{:d}.fits'.format(year))
        year_tab.write(fn_year, format='fits', overwrite=True)

        year_index_dir = os.path.join(self.index_dir, 
                                      'index_{:d}'.format(year))

        try:
            os.makedirs(year_index_dir)
        except OSError:
            if not os.path.isdir(year_index_dir):
                raise

        for scale_num in np.arange(max_scale, min_scale-1, -1):
            cmd = self.build_index_path
            cmd += ' -i {}'.format(fn_year)
            cmd += ' -S {}'.format(sort_by)
            cmd += ' -P {:d}'.format(scale_num)
            cmd += ' -I {:d}{:02d}'.format(year, scale_num)
            fn_index = 'index_{:d}_{:02d}.fits'.format(year, scale_num)
            cmd += ' -o {}'.format(os.path.join(year_index_dir, fn_index))

            sp.call(cmd, shell=True, cwd=self.index_dir)

    def create_index_loop(self, start_year, end_year, step, max_scale=None, 
                          min_scale=None, sort_by='Gmag'):
        """
        Create Astrometry.net indexes for a set of epochs.

        """

        for year in np.arange(start_year, end_year+1, step):
            self.create_index_year(year, max_scale=max_scale, 
                                   min_scale=min_scale, sort_by=sort_by)

