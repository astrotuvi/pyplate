# PyPlate redesign

This document describes the planned redesign of the PyPlate package. It
records the decisions taken, the target package structure, the order of work
and how each step is verified. Section 10 lists developments that are planned
but deliberately left out of the design for now.

Terms used:

- **DDL** (Data Definition Language): SQL statements that define the database
  structure (`CREATE`/`DROP`/`ALTER TABLE`, `CREATE INDEX`, triggers).
- **Upsert**: insert a row, or update it if a row with the same key already
  exists, in a single statement.

## 1. Goals and agreed decisions

| # | Decision |
|---|---|
| 1 | `metadata.py` becomes a `metadata/` subpackage with one module per class. |
| 2 | Metadata classes `Archive`, `Plate`, `Exposure`, `SubExposure`, `Scan`, `Preview`, `Logbook` and `Logpage` have exactly the columns of their database table as attributes (excluding `timestamp_insert`/`timestamp_update`). No name mapping exists between Python and the database. |
| 3 | `plate_id` (and every other `*_id`) is the integer database key. `plate_num` is the plate identifier from the input files. |
| 4 | Clean break for configuration files: no deprecated aliases. A key that is not accepted in its section is an error. The configuration format changes from INI to TOML; `ini2toml` converts the current INI files (3.1). |
| 5 | The number of exposures is worked out as in the current code: `numexp` from the inputs, adjusted by the number of original start times (4.3). |
| 6 | Two archive classes: `Archive` (the table row) and `ArchiveReader` (reads the input files). |
| 7 | Database backends: PostgreSQL, MySQL and a new SQLite backend. |
| 8 | The schema YAML moves to a backend-neutral format. The code targets DR5 (`applause_dr5.yaml`) directly. The table name `exposure_sub` is kept. |
| 9 | Crossmatching moves to its own package with a single matcher (`NearestMatcher`, the current behaviour). |
| 10 | Photometry is not restructured. It only gets the changes needed to work with the new inputs, result classes and DR5 column names; it is reworked later (section 10). |
| 11 | `process/` remains. Long functions outside photometry are split, steps pass explicit inputs, the number of class attributes goes down, and dead code is removed. |
| 12 | Python 3.12 or later: `requires-python = ">=3.12"` in `pyproject.toml`, set in step 1. The design uses union types (`X \| None`), dataclass features from 3.10 and `tomllib` from 3.11; 3.12 is the minimum because NumPy (from 2.5) and SciPy (from 1.18) require it. |
| 13 | Type hints are added to code as each step writes or rewrites it, not in a separate pass. Public signatures and dataclass fields are annotated; local variables are not. The syntax is that of 3.12 (`X \| None`, `list[str]`, and `type` statements or `def f[T]` where an alias or a generic is needed), without `typing.Optional` or `from __future__ import annotations`. NumPy and Astropy types are written plainly (`np.ndarray`, `Table`, `fits.Header`), without `numpy.typing` shapes or dtypes. mypy may be run locally but is not a required check. |
| 14 | Docstrings are written in NumPy style (napoleon in `docs/conf.py`) for the code each step writes or rewrites; unchanged code keeps its docstrings until its step. Public modules, classes, methods and functions have a summary line and, where they add to the signature, `Parameters`, `Returns` and `Raises` sections; private helpers have one only when the logic is not obvious. Types are not repeated in docstrings: they come from the type hints, shown by `autodoc_typehints = "description"` (set in step 1). Record classes name their table and refer to `applause_dr5.yaml` for the column descriptions; only attributes without a column are documented in the class. |

## 2. Package layout

```
pyplate/
  __init__.py
  _version.py
  config/
    __init__.py             re-exports Config and load_config()
    sections.py             the section dataclasses
    loader.py               reads the TOML file, validates keys, builds Config
    ini2toml.py             converts INI configurations to TOML (3.1)
  image.py                  PlateConverter (TIFF → FITS, previews)
  metadata/
    __init__.py             re-exports the public classes
    base.py                 DbRecord base: to_row(), from_row(), columns()
    archive.py              Archive
    archive_reader.py       ArchiveReader
    plate.py                Plate
    exposure.py             Exposure
    subexposure.py          SubExposure
    scan.py                 Scan
    preview.py              Preview
    logbook.py              Logbook
    logpage.py              Logpage
    timeutils.py            ST→UT, mid/weighted times, JD/year/HJD
    codes.py                object-type and method code tables
    header.py               PlateHeader + the one field → FITS keyword table
    readers/
      values.py             values of one plate collected from the inputs
      wfpdb.py              WFPDB maindata / quality / notes / observer
      csv.py                CSV files mapped by column name
      fits.py               values from an existing FITS header
      exif.py               image metadata for Preview / Logpage
  database/
    __init__.py
    database.py             PlateDB: generic insert / insert_many / update / upsert / lookups
    backend.py              Backend interface
    db_pgsql.py
    db_mysql.py
    db_sqlite.py
    schema.py               neutral YAML → Schema object
    ddl.py                  Schema → CREATE / DROP / INDEX / TRIGGER SQL for each backend
    applause_dr4.yaml       converted to neutral format (kept to check the generated SQL)
    applause_dr5.yaml       the target schema
  process/
    __init__.py
    context.py              ProcessContext
    process.py              PlateProcess: orchestrator (was Process); Process: the `process` table row
    pipeline.py             PlatePipeline: single_image(), parallel_run()
    log.py                  ProcessLog
    settings.py             (step 1 only) attributes from the configuration, until ProcessContext
    output.py               writing results to the database / CSV
    extraction.py           SExtractor / PSFEx runs, building the source table
    sources.py              SourceTable + coordinate processing, annular bins, edge flags
    catalog.py              StarCatalog (Gaia only)
    artifacts.py            classifying artifacts with the Keras model
    astrometry.py           AstrometryNetIndex (Gaia only; otherwise unchanged)
    photometry.py           PhotometryProcess: current code, adapted only (6.6)
    solve/
      __init__.py
      solve.py              SolveProcess: top-level flow
      anet.py               astrometry.net wrapper
      scamp.py              SCAMP wrapper + refinement
      multiexp.py           finding multi-exposure and scanner patterns
      solution.py           SolutionSet, Solution, ScannerPattern
    crossmatch/
      __init__.py
      reference.py          ReferenceStars: catalogue moved to the plate epoch and projected
      pairs.py              all source–star pairs within the search radius
      tolerance.py          match and search radii from the astrometric errors
      matcher.py            Matcher interface, Association
      nearest.py            NearestMatcher
```

Removed:

- `conf.py`, replaced by the `config` subpackage;
- the Python 2 compatibility code (`unicode`, `ConfigParser`, `unicodecsv`);
- the aliases `ArchiveMeta`, `PlateMeta`, `LogbookMeta`, `LogpageMeta`, `PreviewMeta`;
- `Process.get_reference_catalogs()` (about 360 lines, never called) and the
  UCAC4 / APASS / Tycho-2 attributes and settings in `process.py`, `solve.py`
  and `sources.py`, including the Tycho-2 fallback for astrometric reference
  stars in `solve.py`, so astrometry and photometry use Gaia only;
- the Tycho-2 parts of `AstrometryNetIndex` (`download_tycho2()` and
  building index files with `catalog='tycho'`). With Gaia the only
  catalogue, the `catalog` argument of `create_index_year()` and
  `create_index_loop()` is removed, so callers that pass it must drop it.
  The default `sort_by` of `create_index_loop()` becomes `'Gmag'`; the old
  default `'BTmag'` (a Tycho-2 magnitude) was replaced by `'Gmag'` for Gaia
  anyway, so the indexes stay the same;
- the settings left over from the removed sub-field recursion:
  `subfield_distort`, `max_recursion_depth`, `force_recursion_depth`;
- the stale `tests/test_db_*.py` and `tests/test_schema.py` scripts.

## 3. Configuration (`config/`)

The TOML file is read once with `tomllib` into frozen dataclasses. The
section names stay as in the INI files (`[Archive]`, `[Files]`,
`["Keyword values"]`, ...); the sections named by other settings become
nested tables: `[Logbook.<name>]` replaces the logbook sections listed in
`[Archive] logbooks`, `[CSV."<file name>"]` the CSV sections, and
`[Lookup.<keyword>]` the `[Keyword lookup]` entries and their sections.
Lists (`plate_csv`, ...) are arrays.

| Section dataclass | Contents |
|---|---|
| `FilesConfig` | input and output directories and file names, as in the current code: the CSV and WFPDB file names (`plate_csv`, …, `wfpdb_maindata`, `wfpdb_quality`, `wfpdb_notes`, `wfpdb_observer`) and the directories (`csv_dir`, `wfpdb_dir`, `fits_dir`, `preview_dir`, `logpage_dir`, `cover_dir`, …) |
| `ProgramsConfig` | paths to `sex`, `psfex`, `scamp`, `solve-field`, `wcs-to-tan` |
| `DatabaseConfig` | `[Database]`: `enable_db_log`, `write_sources_csv`, `write_solution_healpix_csv`, `dr_num`, `process_num_digits`, `source_num_digits`; the nested table `[Database.connection]`: `rdbms`, host / port / socket / file, user, password, database, schema, `yaml`. The results are written to the database when `[Database.connection]` is present (now: when `output_db` names a section) |
| `ArchiveConfig` | `[Archive]` columns + the `[Logbook.<name>]` tables |
| `CsvConfig` | for each `[CSV."<file name>"]` table: column → position, `csv_delimiter`, `csv_quotechar`, `csv_list_delimiter`, `csv_timezone` (default time zone of the recorded times, see 4.2), `csv_encoding` (default UTF-8). Plate CSV tables also accept the link keys `fn_cover`, `fn_log`, `fn_pre` and `fn_scan` and the time keys `ut_start_orig`, `ut_end_orig`, `st_start_orig`, `st_end_orig` (see 4.2); scan and preview tables the link keys `plate_num` and `wfpdb_id`, logpage tables `logbook_num`. The identifier column of each kind (`plate_num`, `filename_scan`, `filename`, `logbook_num`) must be mapped |
| `WfpdbConfig` | `[WFPDB]`: `maindata_timezone` (default time zone of the recorded times, see 4.2), `notes_timezone` (time zone of the times in the WFPDB notes file), `exptime_minutes` (exposure times in minutes; default false), `encoding` (encoding of the WFPDB files; default UTF-8) |
| `ImageConfig` | `[Image]`: `logpage_exif_timezone`, `preview_exif_timezone` (time zone of the EXIF dates in logpage and preview images); `scan_exif_timezone`, `wedge_height`, `cut_wedge` (TIFF conversion) |
| `MetadataConfig` | `[Metadata]` (`evening_date`, `german_transliteration`), `["Keyword values"]`, the `[Lookup.<keyword>]` tables (input values matched without regard to case, as the old INI files did) |
| `ExtractionConfig` | from `[Solve]`: thresholds, filter, PSF settings, `circular_film` |
| `SolveConfig` | from `[Solve]`: `sip`, `skip_bright`, `distort`, `plate_epoch`, `allow_force`, `repeat_find` |
| `CrossmatchConfig` | from `[Solve]`: match-radius settings |
| `CatalogConfig` | `[Catalog]`: `catalog`, `protocol`, `table_name`, and the Gaia database in `[Catalog.connection]` (was the section named by `db`) |
| `PipelineConfig` | which steps and outputs are switched on |
| `HeaderConfig` | `[Header]`: values written only to the FITS header, the same for every plate of the archive: `detector`, `licence`, `acknowledgements`; and the fixed keyword values and comments of `["Integer values"]`, `["Float values"]`, `["String values"]`, `["FITS keyword comments"]`, `["FITS keyword order"]` |

- A `Config` object holds all the sections. Each component receives only its
  own section(s). This replaces every `assign_conf()` method and its
  `try/except configparser.Error` blocks.
- Validation: each dataclass lists its accepted keys. Any other key in any
  section raises an error naming the section and key. In metadata sections
  (`[Archive]`, logbook tables, CSV tables, `["Keyword values"]`) the
  accepted keys are the column names of the target classes, plus the parsing
  settings and the documented link and time keys of plate CSV tables
  (`fn_cover`, `fn_log`, `fn_pre`, `fn_scan`, `ut_start_orig`,
  `ut_end_orig`, `st_start_orig`, `st_end_orig`; see 4.2). Database IDs
  (`*_id` except `wfpdb_id`) and `exposure_num` are not accepted. The
  exception is a lookup table, whose keys are data values rather than
  settings.
- Values of metadata columns keep the type of the input: an integer for a
  float column stays an integer, as the old code read it (so the header
  writes `SITEELEV = 41`, not `41.0`). Settings of float type are converted.

Example:

```toml
[Archive]
archive_id = 7
archive_name = "Lippert-Astrograph"
institute = "Hamburger Sternwarte"

[Logbook.LB1]
logbook_num = 1
logbook_title = "Journal I"

[Logbook.LB2]
logbook_num = 2

[Files]
csv_dir = "data"
plate_csv = ["my_plates.csv"]
scan_csv = ["my_scans.csv"]

[CSV."my_plates.csv"]
csv_list_delimiter = ";"
plate_num = 1
date_orig_start = 2
time_orig_start = 3
time_orig_end = 4
numexp = 5
exptime = 6
observer = 7

[CSV."my_scans.csv"]
filename_scan = 1
plate_num = 2
scan_author = 3
scan_date = 4

["Keyword values"]
observatory = "Astrophysikalische Observatorium Potsdam"
site_longitude = 13.06417

[Lookup.observer]
"j. mets" = "Jaan Mets"
```

### 3.1 Converting INI configurations (`ini2toml`)

`python -m pyplate.config.ini2toml old.conf [more.conf ...] new.toml` converts the
current INI configurations (old keys) to TOML with the new keys. It is the
only support for INI: `load_config()` reads TOML only.

- Several input files are merged in order, as `ConfigParser.read()` does, so
  a configuration split into a shared part and local paths (as in
  `tests/ver4/plates`) gives one file.
- Keys are renamed as listed in `tests/ver4/README.md` (section
  "Configuration for the redesigned code"); logbook, CSV and lookup sections
  move to the `Logbook`, `CSV` and `Lookup` tables; comma-separated lists
  become arrays.
- The section named by `[Database] output_db` (for example `[DB_pgsql]`)
  becomes `[Database.connection]`, and `output_db` goes away.
- Settings that move to `[Header]`: `detector`, `licence` and
  `fits_acknowledgements` (as `acknowledgements`) from `[Keyword values]`.
- Settings of removed features are dropped, and the script lists what it
  dropped:
  - `[Database]`: `use_gaia_fits` (the Gaia file is always used),
    `use_tycho2_fits`, `use_ucac4_db`, `use_apass_db`,
    `ucac4_db_host`, `ucac4_db_user`, `ucac4_db_name`, `ucac4_db_passwd`,
    `ucac4_db_table`, `apass_db_host`, `apass_db_user`, `apass_db_name`,
    `apass_db_passwd`, `apass_db_table`, and the unused `output_db_host`,
    `output_db_user`, `output_db_name`, `output_db_passwd`;
  - `[Files]`: `tycho2_dir`;
  - `[Solve]`: `astref_catalog`, `photref_catalog`, `subfield_distort`,
    `max_recursion_depth`, `force_recursion_depth`;
  - the column-name sections named by `ucac4_db_table` and `apass_db_table`;
  - in CSV sections and logbook sections, database IDs (`scan_id`,
    `preview_id`, `logbook_id`, `logpage_id`), which are not read from input
    files, and `method` (derived from `method_code`);
  - keys that the old code never read: `skycond` in CSV sections (the
    column is `sky_conditions`), `[Programs] build_index_path` and
    `xy2sky_path`;
  - `[Parse keywords]` and `[Rename FITS keywords]` (header features that
    are no longer available), and lookup sections switched off in
    `[Keyword lookup]`.
- The section named by `[Catalog] db` becomes `[Catalog.connection]`.
- A CSV section without its identifier key gets it as column 1, because the
  old code always took the identifier from the first column.
- Values are written with the types of the config dataclass fields, so the
  output has proper booleans and numbers. Any other key that no field
  accepts (for example a typo) stops the conversion with the section and key
  named. The TOML text is written
  by the script itself (no TOML writer is needed: the values are strings,
  numbers, booleans and lists of these).
- Comments in the INI files are not carried over.

## 4. Metadata subpackage

### 4.1 Record classes

Every record class is a `@dataclass` that inherits from `DbRecord`:

```python
class DbRecord:
    table: ClassVar[str]
    def columns(cls) -> list[str]            # dataclass fields that are columns
    def to_row(self) -> dict[str, Any]       # column → value
    def from_row(cls, row) -> Self
```

| Class | Table | Related records (not columns) |
|---|---|---|
| `Archive` | `archive` | `logbooks: list[Logbook]` |
| `Plate` | `plate` | `exposures`, `scans`, `previews`, `logpages` |
| `Exposure` | `exposure` | `subexposures: list[SubExposure]` |
| `SubExposure` | `exposure_sub` | — |
| `Scan` | `scan` | — |
| `Preview` | `preview` | — |
| `Logbook` | `logbook` | — |
| `Logpage` | `logpage` | — |

- The fields are written out explicitly, which gives typing and editor
  completion. A test checks them against `applause_dr5.yaml`, which also
  holds the column descriptions (decision 14).
- `plate_logpage` is written from `plate.logpages`, with `logpage_order`
  taken from the list position.

Values that have no column:

| Kind | Examples | Where it goes |
|---|---|---|
| Parsing intermediates | UT/ST start/end strings | Local to the readers |
| Derived values | object-type text, method text, sexagesimal RA/Dec, ISO dates for FITS | Read-only properties or `PlateHeader` |
| Settings written only to the header | `licence`, `detector`, acknowledgements | `PlateHeader` reads them from the `[Header]` config section |
| Links to other tables | former `fn_scan`, `fn_pre`, `fn_cover`, `fn_log` | `plate.scans`, `plate.previews`, `plate.logpages`; in plate CSV files they are still given with `fn_scan`, `fn_pre`, `fn_cover` and `fn_log` (4.2). `plate.logpages[0]` is the cover, or None without one, so the list position stays the `logpage_order` of the old code. `Logpage.logbook_num` (no column) links a logpage to its logbook |
| Computation intermediates | sexagesimal RA/Dec from the unrounded coordinates, the years of sub-exposures | `Exposure.ra_hms`, `dec_dms`; `SubExposure.year_start`, `year_mid`, `year_end` |
| Raw FITS structure | `SIMPLE`, `BITPIX`, `NAXIS`, `BSCALE`, `BZERO`, `EXTEND`, existing `HISTORY` | Defaults in `PlateHeader`, overwritten from the FITS file when one is read; `Scan` gets `naxis1/2`, `minval`, `maxval` |
| File change date | `DATE` | Set by `PlateHeader` from the current time |

### 4.2 Archive and ArchiveReader

```python
@dataclass
class Archive(DbRecord):
    table = 'archive'
    archive_id: int
    archive_name: str | None = None
    institute: str | None = None
    logbooks: list[Logbook] = field(default_factory=list)
```

```python
class ArchiveReader:
    def __init__(self, cfg: Config)          # reads WFPDB / CSV files named in the config
    archive: Archive                         # built from [Archive] + [Logbook.<name>] tables
    def get_plate(self, plate_num=None, wfpdb_id=None, filename_scan=None) -> Plate
    def iter_plates(self) -> Iterator[Plate]
    def iter_scans(self) -> Iterator[Scan]
    def iter_previews(self) -> Iterator[Preview]
    def iter_logpages(self) -> Iterator[Logpage]
```

- `archive_id` is set in one place only: `ArchiveReader` copies it into every
  record it creates; input files never supply it. It is optional for reading:
  without an `[Archive]` section the records have `archive_id = None`. It is
  required for writing: `PlateDB` refuses to write a record without
  `archive_id` and says so (the column is `NOT NULL`).
- The order of precedence is unchanged: WFPDB maindata → quality → observer →
  notes → plate CSV → scan CSV → `["Keyword values"]` → `[Lookup.<keyword>]`.
- How a CSV column maps to a class: a plate CSV fills `Plate` and `Exposure`,
  and `Plate` wins where both have the name. A scan CSV fills `Scan`. Writing
  `exposure.<column>` targets the exposure explicitly (for example
  `exposure.method_code`). In TOML this is a dotted key, so the exposure
  columns form an `exposure` sub-table of the CSV table
  (`[CSV."<file name>".exposure]` is the same).
- Database IDs are never read from input files. They are set when the record
  is written, or looked up in the database.
- Plate–logpage links: a plate CSV gives the logpage filenames of a plate
  with the keys `fn_cover` (cover image) and `fn_log` (list of logbook
  pages), as in the current code. They are not columns: the reader turns them
  into `plate.logpages`, cover first, and the list position gives
  `logpage_order`. In the same way `fn_scan` (list of scans) and `fn_pre`
  (preview) give `plate.scans` and `plate.previews` in the order of the plate
  CSV; without them, the scans and previews linked to the plate in the scan
  and preview CSV files are used.
- The `plate_num` column of a plate CSV identifies the plate; as in the
  current code it may hold the WFPDB ID. A row found by the WFPDB ID of a
  plate from WFPDB files does not change `plate_num`.
- `exposure.method_code` is the plate's `method_code` unless the CSV maps
  `exposure.method_code`, as the current code wrote the same value to the
  plate and to every exposure.
- Start and end times given as UT or sidereal time in their own CSV columns:
  the keys `ut_start_orig`, `ut_end_orig`, `st_start_orig`, `st_end_orig`
  keep their names. They are not columns: the reader combines them into
  `time_orig_start` / `time_orig_end` with a `UT ` or `ST ` prefix, as the
  current code does.
- Time zone of the recorded times: `Plate.tz_orig` (a DR5 column, see 5.4)
  holds `UT`, `ST` or an IANA zone name such as `Europe/Berlin`. A value from
  a CSV column mapped to `tz_orig` (per row) wins; otherwise the reader uses
  the default of the input file: `csv_timezone` for a CSV file,
  `[WFPDB] maindata_timezone` for WFPDB files. `[Keyword values]` is not used for it,
  because those values override CSV values. WFPDB time strings keep the zone
  prefix of the current code (`ST 19:52:08`).

### 4.3 Exposures and sub-exposures

`ArchiveReader` decides the number of exposures after merging all inputs,
using the rule of the current code:

1. Start from `numexp` in the inputs, default 1.
2. If `date_orig_start` is given, count the `time_orig_start` values; no
   values count as 1.
3. Change `numexp` to that count if the count is larger than `numexp`, or
   smaller but greater than 1.
4. With no date, `numexp` stays as given.

| `numexp` given | Start times | Final `numexp` |
|---|---|---|
| 1 | 3 | 3 |
| 4 | 2 | 2 |
| 4 | 1 | 4 |
| 4 | 0 | 4 |
| 2 | 3, no date | 2 |

The reader then creates `numexp` `Exposure` objects and spreads each
per-exposure field with *n* values as in the current code:

| *n* | Fields | Result |
|---|---|---|
| ≥ `numexp` | any | one value per exposure; extra values ignored |
| 1 | `time_orig_start`, `time_orig_end`, `exptime`, `jd_mid_orig`, when the plate has a date | first exposure only, the rest `None` |
| 1 | computed times (`ut_*`, `jd_*`, `year_*`, `hjd_*`) | first exposure only |
| 1 | everything else | copied to every exposure |
| other | any | values by position, missing ones `None` |

Sub-exposures come from the `|`-separated start times of an exposure. As in
the database, an exposure with one time has no sub-exposures; with more,
`len(subexposures) == num_sub`.

`Plate.compute_values()` computes the times of the first exposure, and of
the others only if a start time is given for them (as the current code did).
If only the first exposure is computed, its `num_sub` applies to all. If no
exposure has an exposure time, the exposure times come from the start and
end times.

### 4.4 Computations

The current `Plate.compute_values()` (570 lines) becomes:

- `Exposure.compute_times(site, evening_date)`: `ut_*`, `jd_*`, `year_*`,
  `hjd_*` (start / mid / weighted / end). Weighted values come from the
  sub-exposures.
- `Exposure.compute_coords()`: `ra_icrs`/`dec_icrs` from `ra_orig`/`dec_orig`.
- `SubExposure.compute_times(site, …)`.
- `Plate.compute_values()`: loops over the exposures.
- `timeutils.py`: pure functions (ST→UT, time-zone handling, HJD).

### 4.5 PlateHeader

`PlateHeader(fits.Header)` is built with
`PlateHeader.from_metadata(plate, scan, cfg, fits_header=None, version=None, now=None)`;
`fits_header` (the header of the scan file) gives the FITS structure and the
existing HISTORY lines. One table in `header.py` lists,
for each FITS keyword: the source field, a per-exposure flag (`DT-OBSn`, …)
and the comment. That table replaces the old `_keyword_meta` and most of the
300-line `PlateHeader.__init__`. WCS insertion and writing to a file or FITS
stay as methods.

The header writes the pyplate version and the current UTC time (`HISTORY`
lines and `DATE`). `from_metadata()` and `insert_wcs()` take optional
`version` and `now` arguments, which default to the package version and the
current UTC time; tests pass fixed values.

The recorded Julian date (`JDA-ORIG`, `JDA-ORn`) comes from the new DR5
column `exposure.jd_mid_orig` (see 5.4).

New compared with the current code: the header writes `TZ-ORIG` from
`plate.tz_orig` (see 4.2), at the place the current keyword order already
reserves for it (after `TME-ORn`, before `JDA-ORIG`) and with the existing
default comment. This is a deliberate difference from the DR4 header; see
`tests/ver4/README.md`.

## 5. Database

### 5.1 Layers

```python
class Backend:
    placeholder: str                         # '%s' or '?'
    def connect(self) -> None                # from the ConnectionConfig
    def qualify(self, table) -> str          # schema.table, db.table, or table
    def execute(self, sql, params) -> None
    def query(self, sql, params) -> tuple    # first row, fetched before the commit
    def executemany(self, sql, rows) -> None
    def insert_returning_id(self, sql, params, id_col) -> int
    def upsert_sql(self, table, columns, key_cols) -> str
    def prepare_value(self, column, value)   # MySQL: invalid DATE -> NULL
    def ddl(self, schema: Schema, mode) -> list[str]
```

`PlateDB` builds the SQL text; the backend adds what differs (`RETURNING`
or `lastrowid`, the upsert clause, the placeholder, `NOW()`).

`PlateDB` works the same for every backend:

- `insert(table, row)`, `insert_many(table, rows)`, `update(table, row, key)`,
  `upsert(…)`;
- `write_archive(archive)`, an upsert on `archive_id`;
- `write_plate(plate)`, which writes the plate, exposures, sub-exposures,
  scans, previews and plate–logpage links;
- `write_logbook(logbook)`, `write_logpage(logpage)`;
- result writers, called from `process/output.py`;
- ID lookups by `plate_num`, `wfpdb_id`, `filename_scan` and so on. A lookup
  whose key is `None` is not sent and returns "not found" (the current code
  sends `WHERE key = NULL`, which never matches).

`to_row()` supplies the columns, so all hand-written column loops go away.

The database layer does not depend on a particular schema version: it writes
the columns that the schema YAML and the given rows have. It must keep
working with `applause_dr4.yaml` and rows with DR4 column names after the
code has moved to DR5, so that the DR4 SQL test (7.2 point 3) keeps running.

Upsert syntax for each backend:

```sql
-- PostgreSQL and SQLite
INSERT INTO archive (archive_id, archive_name, institute)
VALUES (...)
ON CONFLICT (archive_id) DO UPDATE
  SET archive_name = EXCLUDED.archive_name,
      institute = EXCLUDED.institute;

-- MySQL
INSERT INTO archive (archive_id, archive_name, institute)
VALUES (...)
ON DUPLICATE KEY UPDATE
  archive_name = VALUES(archive_name),
  institute = VALUES(institute);
```

From MySQL 8.0.20, `VALUES()` in that clause is deprecated in favour of a row
alias (`... AS new ON DUPLICATE KEY UPDATE archive_name = new.archive_name`).
The MySQL backend uses `VALUES()`, which MariaDB (the production server,
Aria tables) supports; MariaDB does not have the row alias.

### 5.2 Backends

- PostgreSQL (`psycopg2`) and MySQL (`pymysql`): unchanged in what they do,
  rewritten to the `Backend` interface.
- Update triggers for `timestamp_update`: PostgreSQL and SQLite. MySQL has
  none, as in the DR4 schema.
- SQLite (standard-library `sqlite3`):
  - no schema prefix;
  - WAL mode and a busy timeout, so `parallel_run()` workers can write one
    after another;
  - SQLite 3.35 or later is required, so inserts use `INSERT … RETURNING`
    as in PostgreSQL;
  - timestamps stored as ISO text;
  - also used for the in-memory database tests.

### 5.3 Neutral schema format

```yaml
schema: applause_dr5
tables:
- name: exposure
  timestamps: true
  primary_key: [exposure_id]
  columns:
  - {name: exposure_id, type: serial}
  - {name: plate_id,    type: int, unsigned: true, null: false}
  - {name: ra_icrs,     type: double}
  - {name: object_name, type: varchar(80)}
  indexes:
  - {name: plate_ind, columns: [plate_id]}
```

- Types: `serial`, `tinyint`, `smallint`, `int`, `bigint`, `real`, `double`,
  `bool`, `char(n)`, `varchar(n)`, `text`, `date`, `datetime`, `timestamp`,
  plus `unsigned`, `null` and `default`.
- Tables: `primary_key`, and `primary_key_name` where the constraint is not
  named `<table>_pky`. Index names are given without the table name, which
  the generated SQL puts in front (`plate_ind` → `exposure_plate_ind`).
- Per-backend escape hatch: `pgsql:`, `mysql:` or `sqlite:` keys give the full
  column definition. The DR4 tables need it for 41 columns, mostly ID columns
  that are `NOT NULL` in MySQL only.
- Generated, not written by hand: `timestamps: true` adds
  `timestamp_insert`/`timestamp_update` and the update triggers for each
  backend.
- Primary keys: generated as `ALTER TABLE … ADD CONSTRAINT` for PostgreSQL
  and MySQL, and inside `CREATE TABLE` for SQLite.
- `DROP INDEX` is generated correctly for each backend: `DROP INDEX name ON
  table` for MySQL, with the schema prefix for PostgreSQL. The current code
  writes the PostgreSQL form without prefix for both.
- Check: DR4, converted to this format, must generate the same PostgreSQL and
  MySQL SQL as the current code, except the `DROP INDEX` statements above
  (`drop_indexes.sql`), which differ on purpose.

### 5.4 DR5 content changes compared with DR4

| Change | Tables |
|---|---|
| Gaia DR4 instead of EDR3, with the release kept in the column names: `gaiaedr3_id` → `gaiadr4_id`, `gaiaiedr3_gmag` → `gaiadr4_gmag` (also fixes the typo), `gaiaedr3_bp_rp` → `gaiadr4_bp_rp`, `gaiaedr3_dist` → `gaiadr4_dist`, `gaiaedr3_neighbors` → `gaiadr4_neighbors` | `source_calib`, `source_xmatch` |
| `num_gaia_edr3` → `num_gaia_dr4` | `solution`, `phot_calib`, `process` |
| new `jd_mid_orig double`: Julian date of the mid-point of the exposure as recorded in the input (the current code writes it only to the FITS header, as `JDA-ORIG`) | `exposure` |
| new `tz_orig varchar(40)`: time zone of the recorded times (`UT`, `ST` or an IANA zone name); the current code reads it but does not store it | `plate` |
| `exposure_sub` name kept; `flag_xmatch`, `x_gaia` and `y_gaia` kept | — |
| Gaia names also in the index names (`gaiadr4_ind`, `gaiadr4_id_ind`) and the primary key of `source_xmatch` | `source_calib`, `source_xmatch` |
| no index `exposure_ind` on `exposure_id`, a column the table does not have (the DR4 index cannot be created) | `source` |

The schema additions for blends and photometry methods are decided with those
developments (section 10).

The Gaia columns are named for DR4 from step 1, but until the code is
switched to Gaia DR4 data (a separate change, once the DR4 data model is
known) they hold Gaia EDR3 values: the TAP queries and
`gaiaedr3_pyplate.fits` stay as they are, and the reference-plate
comparison replays the EDR3 query results.

## 6. Process subpackage

### 6.1 Shared state

```python
@dataclass
class ProcessContext:
    cfg: Config
    filename: str
    plate: Plate
    scan: Scan
    header: PlateHeader
    imwidth: int
    imheight: int
    plate_epoch: float
    work_dir: Path
    log: ProcessLog
    record: Process              # the `process` table row (columns), updated as steps finish
```

Steps receive the context plus explicit inputs and return result objects.
Steps no longer pass state around by copying each other's attribute
dictionaries (`__dict__.update`).

### 6.2 Orchestrator

```python
class PlateProcess:
    ctx: ProcessContext
    sources: SourceTable | None
    solution: SolutionSet | None
    association: Association | None
    phot_calib: list[PhotCalib] | None

    def setup(self)
    def invert_plate(self)
    def extract_sources(self)
    def classify_artifacts(self)
    def solve_plate(self)
    def crossmatch(self)
    def match_catalog(self, star_catalog)   # one crossmatch (6.5)
    def calibrate_photometry(self)   # current depth loop, adapted only (6.6)
    def finish(self)
```

`PlateProcess` (was `Process`, renamed in step 2) drops from about 215
attribute assignments to the context plus four results.
`PlatePipeline.single_image()` runs:

1. `ArchiveReader` → `Plate`
2. → `PlateHeader`
3. → `PlateProcess`
4. → the steps
5. → the writers in `output.py`: `write_solution`, `write_calibration`
   and `write_sources` (database, one connection for all three),
   `write_sources_csv`, and `write_wcs_header` after solving

Each writer is called under its own `[Pipeline]` switch, as the output
methods of `Process` were, so the switches can still be changed on the
`PlatePipeline` object; there is no single `write_results()`.

`parallel_run()` stays as it is.

### 6.3 Result objects follow the same naming rule

| Object | Table |
|---|---|
| `Process` | `process` |
| `SolutionSet` (was `PlateSolution`) | `solution_set` |
| `Solution` (was `AstrometricSolution`) | `solution` |
| `ScannerPattern` | `scanner_pattern` |
| `PhotCalib`, `PhotCalibCurve` | `phot_calib`, `phot_calib_curve` |
| `SourceTable` columns | `source`, `source_calib` |
| `Association.neighbors` columns | `source_xmatch` |

`SourceTable` may have extra working columns. The writer selects only each
table's columns.

### 6.4 Splitting the long functions (photometry excluded)

| Current | Becomes |
|---|---|
| `Process.extract_sources` (~720 lines) | `PlateProcess.extract_sources` runs `extraction.py`: `measure_sky`, `extract_catalog`, `extract_psf_model_sources`, `run_psfex`, `extract_psf_catalog` (each writes its files with `write_sextractor_config` and runs `run_sextractor`), `build_source_table`; then `SourceTable.compute_annular_bins`, `flag_edge_sources` and `apply_psf_coordinates` (`sources.py`) |
| `Process.classify_artifacts` | `artifacts.predict` (the Keras model) |
| `SolveProcess.solve_plate` (~260) | `select_solve_sources`, `multiexp.find_pattern()`, a loop over `find_next_solution()` and `add_solution()` (unique or duplicate), `improve_solutions()` twice, `solution_set()` |
| `find_astrometric_solution` (~585) | `find_next_solution`: `select_sources` (by exposure number, by nearest-neighbour pattern for multiple exposures, or in force mode); `anet.solve_field` (writes the source list and backend config, runs `solve-field`, retries without SIP); `Solution.from_anet`; `select_reference_stars`; `refine_solution` (`scamp.write_reference_catalog`, `scamp.write_tan_ahead`, `scamp.refine`, `scamp.parse_results`); `remove_matched_sources` |
| `improve_astrometric_solutions` (~370) | `improve_solutions`: isolated reference stars and sources, `crossmatch_cartesian`, `multiexp.find_scanner_pattern`, the pattern table, and for each solution `scamp.refine`, `scamp.parse_results`, `scamp.set_statistics` |
| `find_multiexp_pattern` (~235) | `multiexp.find_pattern` and its helpers: star clusters, cluster centres, scale and rotation across the plate, exposure offsets, exposure numbers |
| `SourceTable.crossmatch_gaia` (~230) | the `crossmatch/` package, run by `PlateProcess.match_catalog` (6.5) |

`ProcessLog` moves to `process/log.py`. Event and level numbers stay as they
are.

### 6.5 Crossmatching

Crossmatching decides which Gaia stars belong to which plate sources. The
current `SourceTable.crossmatch_gaia()` does this in one method and stores the
result in two places: the best match goes into source-table columns, and every
nearby star goes into a separate table (`neighbors_gaia`), which is written to
`source_xmatch`. The redesign keeps exactly this result. It is produced by a
separate package and collected in one object, `Association`.

#### Steps

`PlateProcess.match_catalog(star_catalog)` runs, for each catalogue
(`crossmatch()` with G < 12, and every iteration of
`calibrate_photometry()`):

```python
refs = ReferenceStars.from_catalog(catalog, solution_set, ctx.plate_epoch, log)  # reference.py
radii = tolerance.radii(solution_set)                                             # tolerance.py
pairs = find_pairs(sources, refs, radii.search)                                   # pairs.py
association = NearestMatcher(radii).associate(sources, refs, pairs)               # nearest.py
sources.add_matches(association, refs)
```

1. **Reference stars** (`reference.py`): Gaia positions are moved to the
   plate epoch with proper motions and converted to pixel coordinates with
   each solution's WCS. Only stars inside the image are kept. On a plate with
   several exposures a star appears once per solution, so a reference star is
   identified by its catalogue row and its solution (`index`,
   `solution_num`, `xy`).
2. **Radii** (`tolerance.py`): the current SCAMP-sigma logic gives two
   radii, in pixels and in arcsec (`Radii`, with the mean pixel scale):
   - the match radius (5σ), within which a star can be the match of a source;
   - the search radius (20σ, at least 5″), within which a star counts as a
     neighbour.
3. **Pairs** (`pairs.py`): every source–star pair closer than the search
   radius: `source_index`, `ref_index` and the distance in pixels.
4. **Matching** (`nearest.py`): `NearestMatcher` picks, for each source, the
   nearest reference star within the match radius, as the current code does.
   It returns an `Association`.
5. **Copying to the source table**: `SourceTable.add_matches()` copies the
   match and the matched star's catalogue values into the source columns, and
   the number of neighbours. As in the current code, a source that is not
   matched keeps the values of an earlier call (section 9). The number of
   sources of each solution in the source table is stored in
   `Solution.num_gaia_dr4`. Coordinates are then processed again, as now,
   because matches may change a source's `solution_num`.

#### The `Association` object

```python
class Matcher(Protocol):
    def associate(self, sources: SourceTable, refs: ReferenceStars,
                  pairs: Table) -> Association: ...

@dataclass
class Association:
    matches: Table      # one row per matched source
    neighbors: Table    # one row per source–star pair within the search radius
```

`matches`: the best match of each source; sources without a match have no
row. Besides the columns below it has the working columns `source_index`
and `ref_index` (row of the source table, index of the reference star).

| Column | Meaning |
|---|---|
| `source_num` | plate source |
| `solution_num` | solution (exposure) the matched star belongs to |
| `gaiadr4_id` | matched Gaia star |
| `gaiadr4_dist` | distance between source and star (arcsec) |
| `match_radius` | match radius used (arcsec) |
| `gaiadr4_neighbors` | number of Gaia stars within the search radius |

`neighbors`: all source–star pairs within the search radius, including the
match. This table is written to `source_xmatch` unchanged.

| Column | Meaning |
|---|---|
| `source_num` | plate source |
| `gaiadr4_id` | Gaia star |
| `solution_num` | solution the star's position comes from |
| `dist` | distance between source and star (pixels) |
| `x_gaia`, `y_gaia` | star's position on the plate (pixels) |
| `flag_xmatch` | 1 if this pair is the source's match, otherwise 0 |

Example: source 17 lies near three Gaia stars, and the nearest one is within
the match radius.

`matches`:

| source_num | solution_num | gaiadr4_id | gaiadr4_dist | match_radius | gaiadr4_neighbors |
|---|---|---|---|---|---|
| 17 | 1 | 4057…221 | 0.8 | 2.5 | 3 |

`neighbors`:

| source_num | gaiadr4_id | solution_num | dist | x_gaia | y_gaia | flag_xmatch |
|---|---|---|---|---|---|---|
| 17 | 4057…221 | 1 | 0.8 | 1032.4 | 877.1 | 1 |
| 17 | 4057…305 | 1 | 4.6 | 1035.0 | 879.9 | 0 |
| 17 | 4057…118 | 1 | 7.9 | 1028.7 | 873.0 | 0 |

#### Where the results end up

| Result | Destination |
|---|---|
| `matches` + catalogue values of the matched star (`gaiadr4_gmag`, `gaiadr4_bp_rp`, BP and RP magnitudes) | `SourceTable` columns; written to `source_calib` |
| `neighbors` (of the last crossmatch, `PlateProcess.association`) | `source_xmatch` |
| number of matched sources per solution | `Solution.num_gaia_dr4`; written to `solution` |

Photometry reads the matches from the source table, as now. A source with
`gaiadr4_neighbors == 1` has a single Gaia star nearby and is a candidate
calibration star.

#### Why a separate object

`SourceTable` no longer does any matching; it only receives the result. The
matching method can be replaced without changing the source table, photometry
or database writing: any matcher only has to return the two tables above. A
future blend-aware matcher (section 10) can mark several stars per source in
`neighbors` with the same structure.

### 6.6 Photometry (adapted only)

`photometry.py` (`calibrate_photometry_gaia`, `evaluate_color_term`) and
`Process.calibrate_photometry` keep their structure. Changes:

1. Inputs: `PhotometryProcess(ctx, sources, solution)` replaces
   `photproc.__dict__.update(self.__dict__)`. It uses `log`, `basefn`,
   `write_phot_dir`, the plate header, the sources and the solution set.
2. DR5 column names: `gaiaedr3_*` → `gaiadr4_*`, `num_gaia_edr3` → `num_gaia_dr4`
   (done in step 1 for all of `process/`, because the step-1 pipeline
   writes to the DR5 schema).
3. Result types: `PlateSolution` → `SolutionSet`. `PhotometryProcess`
   still collects the calibrations as `phot_calib` dicts. At the end of
   `PlateProcess.calibrate_photometry` they are converted to `PhotCalib`
   records, each with its calibration curve (`PhotCalib.curve`, not a
   column), which become `PlateProcess.phot_calib`. `output.py` samples
   the curves of the last iteration and writes them as `PhotCalibCurve`
   records. `PhotCalib` and `PhotCalibCurve` are defined in
   `photometry.py`.

## 7. Testing

### 7.1 Unit and regression tests

| Test | What it checks |
|---|---|
| `test_schema_consistency.py` | Every record and result class's columns equal its DR5 table |
| `test_metadata.py` (ported) | Readers, the `numexp` cases in 4.3, sub-exposures, time computations |
| `test_metadata_regression.py` | New results equal the old `compute_values` results in `tests/ver4/metadata/expected.json` (WFPDB, `tests/data` and mock archives; see `tests/ver4/README.md`) |
| `test_timeutils.py` | ST→UT, HJD, evening dates |
| `test_config.py` | Parsing, unknown-key errors; `ini2toml` converts the old-format configurations in `tests/ver4` to exactly their TOML versions (`metadata/*.toml`, `header/header.toml`, `plates/plates.toml`) |
| `test_crossmatch.py` | The `crossmatch/` steps give the results of the old `crossmatch_gaia` for its three calls on a part of each reference plate (`tests/ver4/crossmatch`, cut from the reference run with `gen_crossmatch.py`), run at the old epoch 1950: the source columns after each call and the neighbour tables are identical |
| `test_header.py` | `PlateHeader.from_metadata()` with the mock inputs in `tests/ver4/header/mock_header.json` gives exactly `header.txt`, and after `insert_wcs()` `header_wcs.txt` (see `tests/ver4/README.md`) |
| Reference-plate comparison | The old pipeline (`run_reference_plates.py`) and the new code give the same results for the reference plates LA02333 and LA02334: sources, solutions, crossmatch, calibration and the result SQL, run with `tests/run_reference_plates.py`, compared exactly with `compare_reference_plates.py --dr5` (two runs of the old code with the same Gaia results were identical, apart from times, paths and CPU times in astrometry.net header comments, which the comparison ignores); run on the machine with the external programs and data (see `tests/ver4/README.md`, section "Reference plates") |

### 7.2 Database tests

The database tests need no database server.

Test files: `test_ddl.py` (points 1 and the DR5 SQL of 7.3),
`test_database.py` (point 2 and 7.3), `test_db_queries.py` (point 3).

1. Generated SQL. Before any refactoring, capture the `get_schema_sql()`
   output of the current code (`create_schema`, `drop_schema`,
   `create_indexes`, `drop_indexes`) for DR4, PostgreSQL and MySQL, and commit
   it as `tests/ver4/ddl/`. `ddl.py` must reproduce it from the neutral DR4
   YAML (compared statement by statement, whitespace normalised), except the
   corrected `DROP INDEX` statements (5.3).
2. SQLite round trip. Create the DR5 schema in an in-memory SQLite database,
   write an archive, a plate with several exposures and sub-exposures, a
   scan, previews and logpages, read them back and compare. Write the archive
   twice to check the upsert.
3. Same SQL as the DR4 code. `tests/ver4/queries/*/mock.sql` holds the SQL
   that every `PlateDB` write, update and lookup method of the current code
   sends, called with mock data that fill every column; the inputs are in
   `tests/ver4/mock_data.json`. The test runs the same calls through the new
   database layer with the DR4 schema (neutral format) and DR4 column names,
   records the SQL with the cursor and rendering in `tests/ver4/sqlrender.py`,
   and compares it with `mock.sql`. Any difference must be a deliberate,
   documented change. See `tests/ver4/README.md`.

### 7.3 Tests of new behaviour

The references in `tests/ver4` check that the new code does the same as the
current code. The following parts of step 1 are new or differ on purpose, so
the references do not cover them; each needs its own unit tests, written
together with the code:

| Test | What it checks |
|---|---|
| Archive upsert | The SQL that `write_archive()` sends for PostgreSQL and MySQL equals expected SQL written by hand (5.1); SQLite runs it in the round trip (7.2 point 2) |
| Rules of `PlateDB` | A lookup whose key is `None` sends no SQL and returns "not found"; a record without `archive_id` is refused with a clear error (4.2, 5.1) |
| New DR5 columns | `tz_orig`: a value from the CSV row wins over the file default (`csv_timezone`, `[WFPDB] maindata_timezone`); `jd_mid_orig` is read and written; the header writes `TZ-ORIG` and `JDA-ORIG` from them (4.2, 4.5) |
| DR5 schema content | The neutral `applause_dr5.yaml` equals `applause_dr4.yaml` plus exactly the changes listed in 5.4, so accidental edits are found |
| DR5 SQL for PostgreSQL and MySQL | The generated DR5 SQL is saved once, reviewed, and compared in later runs, because only SQLite executes DR5 in the tests |
| Batches | Writing about 2,500 source rows sends `executemany` in batches of 1000 rows with no row lost or repeated |
| NumPy values | The backends pass NumPy numbers to the database driver as plain Python values (the current PostgreSQL code registers converters for them) |
| Encodings | WFPDB and CSV files are read with `[WFPDB] encoding` and `csv_encoding`; the Latin-1 copy of the POT015 records gives the results of archive `pot015` (see `tests/ver4/README.md`) |

## 8. Order of work

| Step | Content | Check |
|---|---|---|
| 0 | Capture the reference SQL files (7.2) and the old code's output for the reference plates | — |
| 1 | `config` subpackage with `ini2toml`, metadata subpackage, `Archive` / `ArchiveReader`, database layer, neutral YAML, DR5 schema, SQLite backend; TOML configuration files for the archives, the header and the reference plates in `tests/ver4`, made with `ini2toml` (the references use the old INI format); `process/` adapted to the new config, metadata and database layer only as much as needed for the pipeline to keep running, with the DR5 Gaia column names (6.6); the crossmatch epoch stays the old default until step 3 | 7.1 `test_schema_consistency.py`, `test_metadata.py`, `test_metadata_regression.py`, `test_timeutils.py`, `test_config.py`, `test_header.py`, reference-plate comparison; 7.2; 7.3 |
| 2 | `ProcessContext`, result classes, `Process` renamed to `PlateProcess`, `output.py`, dead code removed, photometry adapted (6.6) | reference-plate comparison (7.1) |
| 3 | Long functions split (6.4), `solve/` package, `crossmatch/` with `NearestMatcher`, crossmatch at the plate epoch | reference-plate comparison, `test_crossmatch.py` |

## 9. Open items

Behaviour of the old code that the redesign keeps, so that the results stay
the same, but that is probably not intended. Fixing it changes the results,
so each fix is a separate, deliberate change:

| Behaviour | Consequence |
|---|---|
| When the photometric calibration fails for all solutions, the calibrations of the finished iterations are still written to `phot_calib`, without curves. The old code shared one list object between `Process` and `PhotometryProcess` (`__dict__.update`), so the partial results reached the output. `PlateProcess.calibrate_photometry` now does this explicitly. | `phot_calib` rows for a process with `calibrated = 0`. |
| After solving, `PlatePipeline.single_image()` calls `sources.process_coordinates()` without the solutions; the call only logs a warning. The coordinates are computed only by the crossmatch, which calls it with the solutions. | With `crossmatch_gaia` off, the sources get no sky coordinates (`ra_icrs`, `dec_icrs`, HEALPix, Galactic and ecliptic coordinates, air mass). |
| The crossmatch is repeated with each catalogue of the photometric calibration, and `add_matches()` only overwrites the sources matched in that call. The first catalogue is limited in G (< 12), the later ones in the plate passband (RP + c·(BP−RP)), so very red stars (BP−RP ≈ 6–7) of the first catalogue are missing from the later ones, and their sources keep the first match. | The source table (and `source_calib`) holds matches with stars that are not in the final catalogue, and that are not in `source_xmatch`; `Solution.num_gaia_dr4` counts them, `process.num_gaia_dr4` does not (for the reference plates 7 of 13,317 sources for LA02333, 2 of 3,254 for LA02334). |

## 10. Future developments

### 10.1 Planned for DR5, not yet designed

| Development | How the main design leaves room for it | Likely schema additions (decided later) |
|---|---|---|
| Blend-aware matching | A new `Matcher` implementation; `Association.neighbors` and `source_xmatch` already hold several stars per source. | per-pair flux share and role; per-source number of contributors |
| Gaia XP spectra as photometric reference | Part of the photometry rework (10.2). | which reference method was used; colour-term columns NULL when unused |
| Photometry rework | `photometry.py` is self-contained with explicit inputs, so it can be replaced without touching the other steps. | — |

### 10.2 Candidate design for the photometry rework (to be decided later)

Kept as a starting point, not a commitment.

Layout:

```
process/photometry/
  calibrate.py          PhotometryProcess: iterations and depth loop
                        (moved from Process.calibrate_photometry)
  curve.py              curve fitting, outlier removal, bright-end improvement,
                        position correction, error estimation
  reference.py          PhotometricReference interface
  ref_colorterm.py      ColorTermReference
  ref_xp.py             XPSpectraReference
  results.py            PhotCalib, PhotCalibCurve
```

Reference interface:

```python
class PhotometricReference(Protocol):
    name: str                                            # → phot_calib.ref_method
    def catalog_columns(self) -> list[str]
    def fit(self, sources, association, mask, ctx) -> RefFit
    def reference_mag(self, catalog_rows, fit) -> np.ndarray   # per star, plate passband
    def catalog_mag_limit(self, fit) -> float
    def derived_mags(self, sources, association, fit) -> dict
```

- `ColorTermReference`: the current method, with the colour-term fit,
  `rp + c·(bp−rp)`, the `19 + c` depth limit and the BP/RP back-conversion.
- `XPSpectraReference`: synthetic magnitudes in the plate passband
  (emulsion × filter × atmosphere) from Gaia XP spectra, with no colour term.
  XP spectra exist only for G ≲ 17.65, so deep plates may need a hybrid: XP
  where available, and below that a transformation calibrated on the XP
  stars. One reference can wrap another.

Splitting `calibrate_photometry_gaia`: `select_calibration_stars` →
`reference.reference_mag` → `curve.eliminate_outliers` → `curve.fit_curve` →
`curve.improve_bright_end` → `curve.position_correction` →
`curve.estimate_errors` → `apply_calibration`.

Link to crossmatching:

- `Association.is_single()` replaces the direct `gaiadr4_neighbors == 1` check.
- `Association.blend_mag(ref_mag)` = −2.5·log10 Σ 10^(−0.4 mᵢ) gives the
  combined reference magnitude of a blend, the same for every reference
  method.
- A blend matcher would receive the `PhotometricReference`, so flux shares
  are calculated in the plate passband.

Order of work:

1. Wrap the current code in the interface, and check that the output is
   identical.
2. Split it into `curve.py`.
3. Add XP.
4. Add blends.
