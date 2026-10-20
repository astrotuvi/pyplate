Using PyPlate
=============

Working with metadata
---------------------

An archive is a collection of photographic plates, logbooks, observer notes,
etc. ``ArchiveReader`` in the ``metadata`` subpackage reads the input files
of an archive named in the configuration (see :doc:`configuration`). The
records it returns (``Archive``, ``Plate``, ``Exposure``, ``SubExposure``,
``Scan``, ``Preview``, ``Logbook``, ``Logpage``) have the columns of their
database tables as attributes.

Read the metadata files::

    from pyplate.config import load_config
    from pyplate.metadata import ArchiveReader

    cfg = load_config('/path/to/pyplate.toml')
    reader = ArchiveReader(cfg)

Get a plate by its plate number, WFPDB ID or scan file name, and compute its
times and coordinates::

    plate = reader.get_plate(filename_scan='plate_0001.fits')
    plate.compute_values(cfg.metadata.evening_date)

    for exposure in plate.exposures:
        print(exposure.ut_start, exposure.jd_mid, exposure.ra_icrs)

All plates, scans, previews and logpages of the archive::

    for plate in reader.iter_plates():
        ...

Creating a FITS header
~~~~~~~~~~~~~~~~~~~~~~

Build the header of a scan from the metadata::

    from pyplate.metadata import PlateHeader

    scan = plate.scans[0]
    header = PlateHeader.from_metadata(plate, scan, cfg)

Write the header to a FITS file in ``[Files] write_fits_dir`` (the file is
copied from ``fits_dir`` if it is not there yet), or to a text file in
``write_header_dir``::

    header.output_to_fits('plate_0001.fits')
    header.output_to_file('plate_0001.hdr')


Writing metadata to the database
--------------------------------

``PlateDB`` writes the records to the database in ``[Database.connection]``
(PostgreSQL, MySQL or SQLite)::

    from pyplate.database import PlateDB

    db = PlateDB(cfg.database, cfg.files)
    db.open_connection()
    db.write_archive(reader.archive)

    for plate in reader.iter_plates():
        plate.compute_values(cfg.metadata.evening_date)
        db.write_plate(plate)

    db.close_connection()

``write_plate()`` also writes the exposures, sub-exposures, scans, previews
and the links to logpages. Create the tables with
``db.create_schema(execute=True)``.


Extracting sources from a plate image and solving astrometry
------------------------------------------------------------

``PlateProcess`` runs the steps on one scan in ``[Files] fits_dir``; the
directories and settings come from the configuration::

    from pyplate.process import PlateProcess, output

    proc = PlateProcess(cfg, 'plate_0001.fits', plate=plate, scan=scan,
                        header=header)
    proc.setup()

    proc.invert_plate()
    proc.extract_sources()
    proc.solve_plate()
    output.write_wcs_header(proc)

    proc.finish()

The steps share ``proc.ctx`` (configuration, metadata, image size, plate
epoch, log and the ``process`` table row). The results are ``proc.sources``,
``proc.solution`` and, after ``crossmatch()`` and
``calibrate_photometry()``, ``proc.phot_calib``. The functions in
``pyplate.process.output`` write them to the database and to files.


Creating a pipeline
-------------------


