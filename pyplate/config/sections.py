"""Configuration sections as frozen dataclasses.

Each dataclass lists the keys accepted in its section, with their types and
defaults. :func:`pyplate.config.load_config` builds them from a TOML file;
see DESIGN.md, section 3, and ``docs/configuration.rst``.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConnectionConfig:
    """Database connection: ``[Database.connection]``, ``[Catalog.connection]``.

    ``rdbms`` is ``pgsql``, ``mysql`` or ``sqlite``. ``file`` is the database
    file of SQLite (``:memory:`` for an in-memory database). ``schema`` is
    the schema name (PostgreSQL) or database name (MySQL) used as the table
    prefix; ``yaml`` the schema YAML file, by default ``<schema>.yaml``.
    """

    rdbms: str
    host: str = 'localhost'
    port: int | None = None
    socket: str | None = None
    file: str | None = None
    user: str = ''
    password: str = ''
    database: str = ''
    schema: str | None = None
    yaml: str | None = None


@dataclass(frozen=True)
class ArchiveConfig:
    """``[Archive]`` columns and the ``[Logbook.<name>]`` tables.

    ``logbooks`` maps each logbook name to its ``logbook`` columns.
    """

    archive_id: int | None = None
    archive_name: str | None = None
    institute: str | None = None
    logbooks: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class FilesConfig:
    """``[Files]``: input and output directories and file names."""

    csv_dir: str = ''
    plate_csv: list[str] = field(default_factory=list)
    scan_csv: list[str] = field(default_factory=list)
    preview_csv: list[str] = field(default_factory=list)
    logbook_csv: list[str] = field(default_factory=list)
    logpage_csv: list[str] = field(default_factory=list)
    wfpdb_dir: str = ''
    wfpdb_maindata: str | None = None
    wfpdb_quality: str | None = None
    wfpdb_notes: str | None = None
    wfpdb_observer: str | None = None
    fits_dir: str = ''
    tiff_dir: str = ''
    preview_dir: str = ''
    logpage_dir: str = ''
    cover_dir: str = ''
    index_dir: str = ''
    gaia_dir: str = ''
    work_dir: str = ''
    write_log_dir: str = ''
    write_header_dir: str = ''
    write_fits_dir: str = ''
    write_wedge_dir: str = ''
    write_wcs_dir: str = ''
    write_source_dir: str = ''
    write_phot_dir: str = ''
    write_db_source_dir: str = ''
    write_db_source_calib_dir: str = ''
    write_db_source_xmatch_dir: str = ''
    write_db_solution_healpix_dir: str = ''
    create_checksum: bool = False


@dataclass(frozen=True)
class CsvConfig:
    """A ``[CSV."<file name>"]`` table: how one CSV file is read.

    ``kind`` is the ``[Files]`` list that names the file: ``plate``,
    ``scan``, ``preview``, ``logbook`` or ``logpage``. ``columns`` maps keys
    (column names and link keys) to 1-based positions in the CSV rows;
    ``exposure_columns`` the keys of the ``exposure`` sub-table, which go to
    the exposures even where the plate has a column of the same name.
    """

    filename: str
    kind: str
    columns: dict[str, int] = field(default_factory=dict)
    exposure_columns: dict[str, int] = field(default_factory=dict)
    csv_delimiter: str = ','
    csv_quotechar: str = '"'
    csv_list_delimiter: str = ','
    csv_timezone: str | None = None
    csv_encoding: str = 'utf-8'


@dataclass(frozen=True)
class WfpdbConfig:
    """``[WFPDB]``: how the WFPDB files are read."""

    maindata_timezone: str | None = None
    notes_timezone: str | None = None
    exptime_minutes: bool = False
    encoding: str = 'utf-8'


@dataclass(frozen=True)
class ImageConfig:
    """``[Image]``: image metadata and TIFF conversion."""

    logpage_exif_timezone: str | None = None
    preview_exif_timezone: str | None = None
    scan_exif_timezone: str | None = None
    wedge_height: int | None = None
    cut_wedge: bool = False


@dataclass(frozen=True)
class MetadataConfig:
    """``[Metadata]``, ``["Keyword values"]`` and ``[Lookup.<keyword>]``.

    ``keyword_values`` are column values that apply to every plate (and its
    exposures and scans) and override the input files. ``lookup`` maps a
    column name to a table of replacements; the input value is matched
    without regard to case, as in the old configuration files.
    """

    evening_date: bool = False
    german_transliteration: bool = False
    keyword_values: dict[str, Any] = field(default_factory=dict)
    lookup: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class HeaderConfig:
    """``[Header]`` and the sections with fixed FITS header values.

    ``integer_values``, ``float_values``, ``string_values`` and
    ``keyword_comments`` come from ``["Integer values"]``,
    ``["Float values"]``, ``["String values"]`` and
    ``["FITS keyword comments"]``; ``keyword_order`` from
    ``["FITS keyword order"] keywords``.
    """

    detector: str | None = None
    licence: str | None = None
    acknowledgements: str | None = None
    integer_values: dict[str, int] = field(default_factory=dict)
    float_values: dict[str, float] = field(default_factory=dict)
    string_values: dict[str, str] = field(default_factory=dict)
    keyword_comments: dict[str, str] = field(default_factory=dict)
    keyword_order: list[str] | None = None


@dataclass(frozen=True)
class ProgramsConfig:
    """``[Programs]``: paths of the external programs."""

    sextractor_path: str = 'sex'
    scamp_path: str = 'scamp'
    psfex_path: str = 'psfex'
    solve_field_path: str = 'solve-field'
    wcs_to_tan_path: str = 'wcs-to-tan'


@dataclass(frozen=True)
class DatabaseConfig:
    """``[Database]`` and ``[Database.connection]``.

    The results are written to the database when ``connection`` is set.
    """

    enable_db_log: bool = False
    write_sources_csv: bool = False
    write_solution_healpix_csv: bool = False
    dr_num: int = 0
    process_num_digits: int = 6
    source_num_digits: int = 7
    connection: ConnectionConfig | None = None


@dataclass(frozen=True)
class ExtractionConfig:
    """Source extraction settings, from ``[Solve]``."""

    threshold_sigma: float = 4.
    use_filter: bool = False
    filter_path: str | None = None
    use_psf: bool = False
    psf_threshold_sigma: float = 20.
    psf_model_sigma: float = 20.
    min_model_sources: int = 100
    max_model_sources: int = 10000
    circular_film: bool = False


@dataclass(frozen=True)
class SolveConfig:
    """Astrometric solving settings, from ``[Solve]``."""

    plate_epoch: float = 1950.
    sip: int = 3
    skip_bright: int = 10
    distort: int = 3
    allow_force: bool = False
    repeat_find: bool = True


@dataclass(frozen=True)
class CrossmatchConfig:
    """Crossmatch settings, from ``[Solve]``."""

    crossmatch_radius: float | None = None
    crossmatch_nsigma: float = 10.
    crossmatch_nlogarea: float = 2.
    crossmatch_maxradius: float = 20.


@dataclass(frozen=True)
class CatalogConfig:
    """``[Catalog]``: the Gaia catalogue query.

    ``protocol`` is ``TAP`` or ``SQL``; with ``SQL``, ``connection``
    (``[Catalog.connection]``) is the database with the Gaia table
    ``table_name``.
    """

    catalog: str = 'Gaia'
    protocol: str = 'TAP'
    table_name: str | None = None
    connection: ConnectionConfig | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """``[Pipeline]``: which steps and outputs are switched on."""

    read_wfpdb: bool = False
    read_csv: bool = False
    read_fits: bool = False
    output_header_file: bool = False
    output_header_fits: bool = False
    invert_image: bool = False
    extract_sources: bool = False
    classify_artifacts: bool = False
    solve_plate: bool = False
    output_solution_db: bool = False
    output_wcs_file: bool = False
    crossmatch_gaia: bool = False
    calibrate_photometry: bool = False
    output_calibration_db: bool = False
    output_sources_db: bool = False
    output_sources_csv: bool = False
    processes: int = 1
    process_max_tasks: int = 0
    wait_start: float = 1.


@dataclass(frozen=True)
class Config:
    """All configuration sections. ``csv`` maps file names to their tables."""

    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    files: FilesConfig = field(default_factory=FilesConfig)
    csv: dict[str, CsvConfig] = field(default_factory=dict)
    wfpdb: WfpdbConfig = field(default_factory=WfpdbConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    header: HeaderConfig = field(default_factory=HeaderConfig)
    programs: ProgramsConfig = field(default_factory=ProgramsConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    solve: SolveConfig = field(default_factory=SolveConfig)
    crossmatch: CrossmatchConfig = field(default_factory=CrossmatchConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
