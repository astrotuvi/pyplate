import os
import numpy as np
import pyvo as vo
import warnings
from astropy.table import Table, Column, MaskedColumn, vstack, setdiff
from astropy.coordinates import SkyCoord
from astropy import units as u
#from astroquery.gaia import Gaia
from .solve import PlateSolution
from ..conf import read_conf
from ..database.database import PlateDB

try:
    import configparser
except ImportError:
    import ConfigParser as configparser


class StarCatalog(Table):
    """
    Class for external astrometric and photometric catalog

    """

    def __init__(self, *args, catalog='Gaia', protocol='TAP', **kwargs):
        """
        Initialise StarCatalog class.

        Parameters
        ----------
        catalog : str
            Name of the external catalog (currently, only Gaia is supported)
        protocol : str
            Catalog query protocol (TAP or SQL)
        """

        super().__init__(*args, **kwargs)

        self.catalog = catalog
        self.protocol = protocol
        self.db_section = None
        self.table_name = None

        self.log = None
        self.gaia_dir = ''
        self.scratch_dir = ''
        self.mag_range = None
        self.name = None

        self.conf = None
        self.db = None

    def assign_conf(self, conf):
        """
        Parse configuration and set class attributes.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        self.conf = conf

        for attr in ['catalog', 'protocol', 'table_name']:
            try:
                setattr(self, attr, conf.get('Catalog', attr))
            except configparser.Error:
                pass

        try:
            self.protocol = self.protocol.upper()
        except TypeError:
            pass

        if self.protocol == 'SQL':
            try:
                self.db_section = conf.get('Catalog', 'db')
            except configparser.Error:
                pass

    def query_gaia_tap(self, query, skycoord, radius):
        """
        Query Gaia EDR3 catalogue with TAP.

        Parameters
        ----------
        query : str
            Query string with placeholders for table name and positional
            constraints
        skycoord : :class:`astropy.coordinates.SkyCoord`
            Sky coordinates for the query center
        radius : :class:`astropy.units.Quantity`
            Angular radius around the center coordinates
        """

        # Initialise TAP service
        tap_service = vo.dal.TAPService('https://gaia.aip.de/tap')

        # Schema and table name in the Gaia database
        table_name = 'gaiaedr3.gaia_source'

        # Suppress warnings
        warnings.filterwarnings('ignore', module='astropy.io.votable')

        # Construct query string
        pos_query_str = ('CONTAINS(POINT(\'ICRS\',ra,dec), '
                         'CIRCLE(\'ICRS\',{:f},{:f},{:f}))=1')
        pos_query = (pos_query_str
                     .format(skycoord.ra.to(u.deg).value,
                             skycoord.dec.to(u.deg).value,
                             radius.to(u.deg).value))
        tap_query = query.format(table_name, pos_query)

        if self.log is not None:
            self.log.write('Gaia TAP query: {}'.format(tap_query),
                           level=4, event=41)

        tap_result = tap_service.run_async(tap_query, queue='2h')
        tab = tap_result.to_table()

        return tab

    def query_gaia_sql(self, query, skycoord, radius):
        """
        Query Gaia EDR3 catalogue with SQL.

        Parameters
        ----------
        query : str
            Query string with placeholders for table name and positional
            constraints
        skycoord : :class:`astropy.coordinates.SkyCoord`
            Sky coordinates for the query center
        radius : :class:`astropy.units.Quantity`
            Angular radius around the center coordinates
        """

        # Construct query string
        pos_query_str = ('pos @ '
                         'SCIRCLE(SPOINT(RADIANS({:f}), RADIANS({:f})), '
                         'RADIANS({:f}))')
        pos_query = (pos_query_str
                     .format(skycoord.ra.to(u.deg).value,
                             skycoord.dec.to(u.deg).value,
                             radius.to(u.deg).value))
        sql_query = query.format(self.table_name, pos_query)

        if self.log is not None:
            self.log.write('Gaia SQL query: {}'.format(sql_query),
                           level=4, event=42)

        gaiadb = PlateDB()
        gaiadb.assign_conf(self.conf, section=self.db_section)
        gaiadb.open_connection()

        res = gaiadb.db.execute_select_query(sql_query)
        cols = [col.strip() for col in
                sql_query.split('FROM')[0].split('SELECT')[1].split(',')]
        dtype = ['i8'] + ['f8'] * (len(cols) - 1)

        if len(res) > 0:
            tab = Table(rows=res, names=cols, dtype=dtype)
        else:
            tab = Table(names=cols, dtype=dtype)

        gaiadb.close_connection()

        return tab

    def query_gaia(self, plate_solution=None, skycoord=None, radius=None,
                   mag_range=[0,20], color_term=None, filename=None,
                   protocol=None):
        """
        Query Gaia EDR3 catalogue for all plate solutions and
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
        protocol : str
            Query protocol ('TAP', 'SQL'). If not specified, the value of the
            class attribute is used.

        """

        use_coord_radius = (isinstance(skycoord, SkyCoord) and
                            isinstance(radius, u.Quantity))

        assert not isinstance(mag_range, str)
        assert len(mag_range) == 2
        assert isinstance(plate_solution, PlateSolution) or use_coord_radius

        if protocol is None:
            protocol = self.protocol

        assert isinstance(protocol, str)

        psol = plate_solution
        protocol = protocol.upper()

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

        # Construct query string
        query_cols = ('source_id,ra,dec,ref_epoch,pmra,pmdec,phot_g_mean_mag,'
                      'phot_bp_mean_mag,phot_rp_mean_mag,bp_rp')
        query_str = ('SELECT {} '
                     'FROM {{}} '
                     'WHERE {{}} '
                     'AND {} >= {} '
                     'AND {} < {} '
                     'AND astrometric_params_solved=31'
                     .format(query_cols,
                             passband, str(round(mag_range[0], 5)),
                             passband, str(round(mag_range[1], 5))))

        gaia_tables = []

        # Use given coordinates and radius for query
        if use_coord_radius:
            if protocol == 'SQL':
                tab = self.query_gaia_sql(query_str, skycoord, radius)
            else:
                tab = self.query_gaia_tap(query_str, skycoord, radius)

            gaia_tables.append(tab)

            if filename is None:
                filename = ('gaiaedr3_{:.2f}_{:.2f}_{:.2f}.fits'
                            .format(skycoord.ra.to(u.deg).value,
                                    skycoord.dec.to(u.deg).value,
                                    radius.to(u.deg).value))
            else:
                filename = os.path.basename(filename)

            #fn_tab = os.path.join(self.gaia_dir, filename)
            #tab.write(fn_tab, format='fits', overwrite=True)

        # Use astrometric solutions for query
        else:
            half_diag = u.Quantity([sol['half_diag'] for sol in psol.solutions])
            fov_diag = 2 * half_diag[half_diag > 0 * u.deg].mean()

            # If max angular separation between solutions is less than
            # FOV diagonal, then query Gaia once for all solutions.
            # Otherwise, query Gaia separately for individual solutions.
            if psol.max_separation < fov_diag:
                if protocol == 'SQL':
                    tab = self.query_gaia_sql(query_str, psol.centroid,
                                              psol.radius)
                else:
                    tab = self.query_gaia_tap(query_str, psol.centroid,
                                              psol.radius)

                gaia_tables.append(tab)

                #fn_tab = os.path.join(self.scratch_dir, 'gaiaedr3.fits')
                #tab.write(fn_tab, format='fits', overwrite=True)
            else:
                # Loop through solutions
                for i in np.arange(psol.num_solutions):
                    solution = psol.solutions[i]
                    sol_skycoord = SkyCoord(ra=solution['ra_icrs']*u.deg,
                                            dec=solution['dec_icrs']*u.deg)

                    if protocol == 'SQL':
                        tab = self.query_gaia_sql(query_str, sol_skycoord,
                                                  solution['half_diag'])
                    else:
                        tab = self.query_gaia_tap(query_str, sol_skycoord,
                                                  solution['half_diag'])

                    tab['solution_num'] = i + 1
                    gaia_tables.append(tab)

                    #fn_tab = os.path.join(self.scratch_dir,
                    #                      'gaiaedr3-{:02d}.fits'.format(i+1))
                    #tab.write(fn_tab, format='fits', overwrite=True)

        # Append data from files to the catalog
        self.append_gaia(gaia_tables)

        # Add catalog name
        self.name = 'Gaia EDR3'

        # Update mag_range
        if self.mag_range is None:
            self.mag_range = mag_range

        if mag_range[0] < self.mag_range[0]:
            self.mag_range[0] = mag_range[0]

        if mag_range[1] > self.mag_range[1]:
            self.mag_range[1] = mag_range[1]

    def append_gaia(self, gaia_tables):
        """
        Append data from Gaia query result tables.

        Parameters:
        -----------
        gaia_tables : list
            List of Astropy tables with Gaia query results

        """

        if self.log is not None:
            self.log.write('Appending data from Gaia query result tables',
                           level=3, event=43)

        assert isinstance(gaia_tables, list)

        # Loop through Gaia files
        for gaia_table in gaia_tables:
            #gaia_table = Table.read(gaia_file)

            # Replace column names with generic names
            gaia_table.rename_column('phot_g_mean_mag', 'mag')
            gaia_table.rename_column('phot_bp_mean_mag', 'mag1')
            gaia_table.rename_column('phot_rp_mean_mag', 'mag2')
            gaia_table.rename_column('bp_rp', 'color_index')

            # Mask nan values in listed columns
            for col in ['mag1', 'mag2', 'color_index']:
                gaia_table[col] = MaskedColumn(gaia_table[col],
                                               mask=np.isnan(gaia_table[col]))

            # If catalog is empty, copy all data from Gaia table
            if len(self) == 0:
                self.columns = gaia_table.columns
            elif len(gaia_table) == 0:
                if self.log is not None:
                    self.log.write('Gaia query result table is empty!',
                                   level=4, event=43)
            else:
                # Find rows in the Gaia table that we do not have yet
                d = setdiff(gaia_table, self, keys=['source_id'])

                # If there are new sources, add them to the catalog
                if len(d) > 0:
                    self.columns = vstack([self, d]).columns

