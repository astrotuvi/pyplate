import os
import numpy as np
from astropy.table import Table, Column, vstack, setdiff
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.gaia import Gaia
from .solve import PlateSolution
from ..conf import read_conf


class StarCatalog(Table):
    """
    Class for external astrometric and photometric catalog

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log = None
        self.gaia_dir = ''
        self.scratch_dir = ''
        self.mag_range = None
        self.name = None

    def query_gaia(self, plate_solution=None, skycoord=None, radius=None,
                   mag_range=[0,20], color_term=None, filename=None):
        """
        Query Gaia DR2 catalogue for all plate solutions and
        store results in FITS files.

        Parameters
        ----------
        plate_solution : :class:`solve.PlateSolution`
            Plate solution containing one or more astrometric solutions
        skycoord : :class:`astropy.coordinates.SkyCoord`
            Sky coordinates for the query center
        radius : :class:`astropy.units.Quantity`
            Angular radius around the center coordinates
        mag_range : list
            A two-element list specifying bright and faint magnitude limits
            for Gaia catalog query.
        color_term : float
            A value characterising plate emulsion:
            natural magnitude = RP + C * (BP-RP)
        filename : str
            Name of FITS file for storing query results

        """

        use_coord_radius = (isinstance(skycoord, SkyCoord) and 
                            isinstance(radius, u.Quantity))

        assert not isinstance(mag_range, str)
        assert len(mag_range) == 2
        assert isinstance(plate_solution, PlateSolution) or use_coord_radius

        psol = plate_solution

        if mag_range[0] is None:
            mag_range[0] = 0

        if mag_range[1] is None:
            mag_range[1] = 99

        if color_term is not None:
            if color_term < 0:
                passband = ('phot_rp_mean_mag - {} * bp_rp'
                            .format(round(np.abs(color_term), 5)))
            else:
                passband = ('phot_rp_mean_mag + {} * bp_rp'
                            .format(round(color_term, 5)))
        else:
            passband = 'phot_g_mean_mag'

        pos_query_str = ('CONTAINS(POINT(\'ICRS\',ra,dec), '
                         'CIRCLE(\'ICRS\',{:f},{:f},{:f}))=1')
        query_cols = ('source_id,ra,dec,ref_epoch,pmra,pmdec,phot_g_mean_mag,'
                      'phot_bp_mean_mag,phot_rp_mean_mag,bp_rp')
        query_str = ('SELECT {{}} '
                     'FROM gaiadr2.gaia_source '
                     'WHERE {{}} '
                     'AND {} >= {} '
                     'AND {} < {} '
                     'AND astrometric_params_solved=31'
                     .format(passband, str(round(mag_range[0], 5)), 
                             passband, str(round(mag_range[1], 5))))

        gaia_files = []

        # Use given coordinates and radius for query
        if use_coord_radius:
            pos_query = (pos_query_str
                         .format(skycoord.ra.to(u.deg).value,
                                 skycoord.dec.to(u.deg).value,
                                 radius.to(u.deg).value))
            query = query_str.format(query_cols, pos_query)

            if self.log is not None:
                self.log.write('Gaia query: {}'.format(query), 
                               level=4, event=0)

            if filename is None:
                filename = ('gaiadr2_{:.2f}_{:.2f}_{:.2f}.fits'
                            .format(skycoord.ra.to(u.deg).value,
                                    skycoord.dec.to(u.deg).value,
                                    radius.to(u.deg).value))
            else:
                filename = os.path.basename(filename)

            fn_tab = os.path.join(self.gaia_dir, filename)
            job = Gaia.launch_job_async(query, output_file=fn_tab, 
                                        output_format='fits', 
                                        dump_to_file=True)
            gaia_files.append(fn_tab)

        # Use astrometric solutions for query
        else:
            half_diag = u.Quantity([sol['half_diag'] for sol in psol.solutions])
            fov_diag = 2 * half_diag[half_diag > 0 * u.deg].mean()

            # If max angular separation between solutions is less than
            # FOV diagonal, then query Gaia once for all solutions. 
            # Otherwise, query Gaia separately for individual solutions.
            if psol.max_sep < fov_diag:
                pos_query = (pos_query_str
                             .format(psol.centroid.ra.to(u.deg).value,
                                     psol.centroid.dec.to(u.deg).value,
                                     psol.radius.to(u.deg).value))
                query = query_str.format(query_cols, pos_query)

                if self.log is not None:
                    self.log.write('Gaia query: {}'.format(query), level=4, event=0)

                fn_tab = os.path.join(self.scratch_dir, 'gaiadr2.fits')
                job = Gaia.launch_job_async(query, output_file=fn_tab, 
                                            output_format='fits', 
                                            dump_to_file=True)
                gaia_files.append(fn_tab)
            else:
                # Loop through solutions
                for i in np.arange(psol.num_solutions):
                    solution = psol.solutions[i]
                    pos_query = (pos_query_str
                                 .format(solution['raj2000'], solution['dej2000'], 
                                         solution['half_diag'].to(u.deg).value))
                    query = query_str.format(query_cols, pos_query)

                    if self.log is not None:
                        self.log.write('Gaia query: {}'.format(query), level=4, event=0)

                    fn_tab = os.path.join(self.scratch_dir, 
                                          'gaiadr2-{:02d}.fits'.format(i+1))
                    job = Gaia.launch_job_async(query, output_file=fn_tab, 
                                                output_format='fits', 
                                                dump_to_file=True)
                    gaia_files.append(fn_tab)

        # Append data from files to the catalog
        self.append_gaia(gaia_files)

        # Add catalog name
        self.name = 'Gaia DR2'

        # Update mag_range
        if self.mag_range is None:
            self.mag_range = mag_range

        if mag_range[0] < self.mag_range[0]:
            self.mag_range[0] = mag_range[0]

        if mag_range[1] > self.mag_range[1]:
            self.mag_range[1] = mag_range[1]

    def append_gaia(self, gaia_files):
        """
        Append data from Gaia query result files.

        Parameters:
        -----------
        gaia_files : list
            List of FITS files with Gaia query results

        """

        assert isinstance(gaia_files, list)

        # Loop through Gaia files
        for gaia_file in gaia_files:
            gaia_table = Table.read(gaia_file)

            # Replace column names with generic names
            gaia_table.rename_column('phot_g_mean_mag', 'mag')
            gaia_table.rename_column('phot_bp_mean_mag', 'mag1')
            gaia_table.rename_column('phot_rp_mean_mag', 'mag2')
            gaia_table.rename_column('bp_rp', 'color_index')

            # If catalog is empty, copy all data from Gaia table
            if len(self) == 0:
                self.columns = gaia_table.columns
            elif len(gaia_table) == 0:
                self.log.write('Gaia query result table is empty!',
                               level=4, event=0)
            else:
                # Find rows in the Gaia table that we do not have yet
                d = setdiff(gaia_table, self, keys=['source_id'])

                # If there are new sources, add them to the catalog
                if len(d) > 0:
                    self.columns = vstack([self, d]).columns

