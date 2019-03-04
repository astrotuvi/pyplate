Using PyPlate
=============

Working with metadata
---------------------

An archive is a collection of photographic plates, logbooks, observer notes, 
etc. In ``PyPlate``, an archive is represented with the ``Archive`` class in
the ``metadata`` module. An individual plate is represented with the ``Plate``
and a FITS header with ``PlateHeader`` class.

Initialise an archive::

    import pyplate
    archive = pyplate.metadata.Archive()

Importing metadata from files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Metadata can be conveniently imported from the Wide-Field Plate Database 
(WFPDB) files and comma-separated values (CSV) files.

Import metadata from WFPDB files::

    archive.read_wfpdb(wfpdb_dir='/path/to/wfpdb_dir', 
                       fn_maindata='maindata.txt', fn_quality='quality.txt',
                       fn_notes='notes.txt', fn_observer='observer.txt')

Import metadata from CSV files::

    archive.read_csv(csv_dir='/path/to/csv_dir', fn_plate_csv='plates.csv',
                     fn_scan_csv='scans.csv', fn_logpage_csv='logpages.csv')



Creating and managing a FITS header
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an empty header::

    pheader = pyplate.metadata.PlateHeader()
    pheader.populate()

Create header from a FITS file::

    pheader = pyplate.metadata.PlateHeader.from_fits('/path/to/file.fits')

Create header from a text file that contains a FITS header::

    pheader = pyplate.metadata.PlateHeader.from_hdrf('/path/to/file.hdrf')

Update header with metadata::

    pmeta = pyplate.metadata.Plate()
    pheader.update_from_platemeta(pmeta)

Configure input and output directories::

    pheader.fits_dir = '/path/to/fits_dir'
    pheader.write_fits_dir = '/path/to/write_fits_dir'
    pheader.write_header_dir = '/path/to/write_header_dir'

Write header to a FITS file in ``write_fits_dir``::

    pheader.output_to_fits('file.fits')

If the FITS file exists, then its header is updated with ``pheader``. If the 
file does not exist, then ``PyPlate`` assumes that the file with the same name
exists in ``fits_dir``. The file is then opened for reading and an output
file is written to ``write_fits_dir`` with the current header.

Write header to a text file in ``write_header_dir``::

    pheader.output_to_fits('file.hdr')


Extracting sources from a plate image and solving astrometry
------------------------------------------------------------

Initialise plate process::

    proc = pyplate.solve.SolveProcess('file.fits')

Configure process::

    proc.fits_dir = '/path/to/fits_dir'
    proc.tycho2_dir = '/path/to/tycho2_dir'
    proc.work_dir = '/path/to/work_dir'
    proc.write_source_dir = '/path/to/write_source_dir'
    proc.write_wcs_dir = '/path/to/write_wcs_dir'
    proc.write_log_dir = '/path/to/write_log_dir'

Alternatively, specify a configuration file or pass a variable containing 
configuration::

    proc.assign_conf('/path/to/conf_file.cfg')
    
    # Use ArchiveMeta configuration instead
    proc.assign_conf(archive.conf)

Set up process::

    proc.setup()

Extract sources::

    proc.extract_sources()

Finish process::

    proc.finish()


Writing data to database
------------------------


Creating a pipeline
-------------------


