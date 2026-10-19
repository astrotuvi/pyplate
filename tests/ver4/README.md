# DR4 database SQL reference

This directory holds output of the code before the redesign (DESIGN.md),
with the inputs needed to produce it again: the SQL that the database code
sends for the `applause_dr4` schema, a plate FITS header, and the metadata
computed from archive input files. After the redesign, tests run the same
inputs through the new code and check that the results are unchanged.

## Contents

| File | Contents |
|---|---|
| `ddl/{pgsql,mysql}/{create_schema,drop_schema,create_indexes,drop_indexes}.sql` | Table, index and trigger SQL from `get_schema_sql()` |
| `queries/{pgsql,mysql}/mock.sql` | SQL from every `PlateDB` write, update and lookup method, called with mock data |
| `mock_data.json` | The inputs of those method calls, in database column names, without pyplate classes |
| `sqlrender.py` | Recording cursor, SQL rendering and JSON decoding. No pyplate imports; used by the generators and by the test of the redesigned code |
| `sqlrecord.py` | Sets up the old `PlateDB` with a recording cursor (generators only) |
| `gen_ddl.py` | Generates `ddl/` |
| `gen_mock_queries.py` | Generates `queries/*/mock.sql` and `mock_data.json` |
| `header/header.txt`, `header/header_wcs.txt` | Plate FITS header from mock metadata, before and after the WCS is inserted |
| `header/mock_header.json` | The inputs of that header |
| `gen_mock_header.py` | Generates `header/` |
| `metadata/` | Archive inputs (old-format configuration, WFPDB and CSV files) and `expected.json`, the metadata the old code computes from them |
| `gen_metadata_regression.py` | Generates `metadata/expected.json` |
| `make_metadata_inputs.py` | Creates the FITS files and images in `metadata/` (inputs, committed; needs no pyplate) |
| `plates/` | Configuration of the reference-plate run (`plates.conf`; `paths.local.example` as template for the local paths) and the published summary of the reference run (`reference/`) |
| `run_reference_plates.py` | Runs the old pipeline on the reference plates and saves the results (section "Reference plates") |
| `compare_reference_plates.py` | Compares two reference-plate runs |

## Regenerating the reference files

The generators run against the old code, as of commit `ed2a193` (the code
under `pyplate/` had not changed since that commit when the files were
generated).
They need numpy, astropy and PyYAML, but no database driver and no database
connection. From the repository root:

```
PYTHONPATH=. python tests/ver4/gen_ddl.py
PYTHONPATH=. python tests/ver4/gen_mock_queries.py
PYTHONPATH=. python tests/ver4/gen_mock_header.py
PYTHONPATH=. python tests/ver4/gen_metadata_regression.py
```

`make_metadata_inputs.py` is not part of the regeneration: the FITS files and
images it made are committed inputs. Run it only to recreate them.

After the redesign, check out the old code in a separate worktree to run
them, for example `git worktree add ../pyplate-dr4 ed2a193`, and copy the
`tests/ver4` scripts there.

The rest of this file describes the SQL reference, then the header
reference (section "Plate header"), the metadata reference (section
"Metadata regression") and the reference plates (section "Reference
plates").

## How the SQL was recorded

The old `PlateDB` kept its real `DB_pgsql` / `DB_mysql` backend, so the
backend logic ran unchanged, but the database cursor was replaced by
`RecordingCursor` (`sqlrender.py`):

- `execute(sql, params)` records the statement.
- An `INSERT` gets an ID 1, 2, 3, ... counted per table, where the table is
  the name as written in the SQL, including the schema prefix
  (`applause_dr4.plate`). PostgreSQL gets it from `fetchone()` after
  `RETURNING`; the MySQL backend removes `RETURNING` and reads `lastrowid`.
- A `SELECT` finds one row with the values 9001, 9002, ... for the selected
  columns (a single column is returned as a scalar by both backends). In
  cases with `"lookups_find": false`, a `SELECT` finds no rows.
- `executemany(sql, rows)` records a comment line
  `-- executemany: N rows` followed by one statement per row.
- `commit()` does nothing.

`write_process_start()` writes the hostname and the pyplate version. These
were replaced by `mockhost` and `mock-version` (`socket.gethostname` and
`pyplate.database.database.__version__` were patched). `PlateDB.dr_num` was
4; `process_num_digits` and `source_num_digits` had their defaults (6, 7).

## How values are written into the SQL

There is no database driver, so `render()` and `literal()` in
`sqlrender.py` put the parameter values into the `%s` placeholders:

| Value | PostgreSQL | MySQL |
|---|---|---|
| `None` | `NULL` | `NULL` |
| `True` / `False` | `TRUE` / `FALSE` | `1` / `0` |
| integer | digits | digits |
| float | `repr()` | `repr()` |
| NaN | `'NaN'` | `NULL` |
| string | quoted, `'` doubled | quoted, `'` doubled, `\` doubled |

The test of the redesigned code must use these functions unchanged.
`format_queries()` returns the recorded queries in the format of `mock.sql`:
for each case, a line `-- <label>`, the statements, each ending in `;`, and
an empty line.

## mock_data.json

Top-level keys: `description`, `schema`, `dr_num`, `hostname`,
`pyplate_version`, `select_result`, `insert_ids` and `cases`. Each case has:

- `label`: the comment line in `mock.sql`;
- `method`: the old `PlateDB` method that was called;
- `lookups_find`: whether a `SELECT` finds a row in this case;
- `args`, `kwargs`: the arguments, after conversion described below.

All values are in DR4 column names. Values that JSON cannot hold are
marked, and `sqlrender.decode()` converts them back: `{"nan": true}`,
`{"inf": 1}` or `{"inf": -1}`, `{"quantity": value, "unit": unit}` (astropy
Quantity) and `{"header": [[keyword, value, comment], ...]}` (FITS header).

Arguments by method:

| Method | First argument | Notes |
|---|---|---|
| `write_plate` | Plate: `plate` columns, `plate_id`, `exposures` | Each exposure: `exposure` columns (without `plate_id`), `exposure_num`, `subexposures`. Each sub-exposure: `exposure_sub` columns (without `plate_id`), `subexposure_num`. An exposure with `num_sub` = 1 has an empty `subexposures` list. `plate_id` null means a new plate. |
| `write_plate_logpage` | `plate_id`, `plate_num`, `archive_id`, `wfpdb_id`, `logpages` | `logpages` is a list of logpage filenames; `logpage_order` is the list position (the cover first, order 0). `plate_id` null means look it up. |
| `write_scan`, `update_scan` | `scan` columns, `scan_id`, `plate_id`, `plate_num`, `wfpdb_id` | `plate_id` / `scan_id` null means look it up; `plate_num` and `wfpdb_id` are the lookup keys. `update_scan` may have `filecols` in `kwargs`. |
| `write_preview` | `preview` columns, `preview_id`, `plate_id`, `plate_num`, `wfpdb_id` | As for scans. |
| `write_logbook` | `logbook` columns | |
| `write_logpage` | `logpage` columns, `logbook_num` | `logbook_id` null means look it up by `logbook_num` and `archive_id`. |
| `write_platesolution` | `solution_set` columns | In the old code an object with attributes. |
| `write_solution` | `solution` columns | |
| `write_solution_healpix`, `write_sources`, `write_source_xmatches` | List of rows (astropy Table in the old code) | Source rows have the columns of both `source` and `source_calib`, plus `source_num`. |
| `write_scanner_pattern` | One row (astropy Table row in the old code) | |
| `write_phot_calib`, `write_calib_curve` | Dictionary of columns | |
| others | Plain values | IDs, filenames, process columns as keyword arguments |

The JSON was checked by rebuilding the old-style objects from it alone and
running the old methods: the SQL was identical to `mock.sql` for both
backends.

## Writing the SQL test for the redesigned code

The redesigned code targets DR5, but this reference is DR4. The test checks
the database subpackage only: run the new database layer with the DR4 schema
(`applause_dr4.yaml` in the new neutral format) and pass it rows with the DR4
column names from the JSON. The new record classes, which use DR5 names, are
tested elsewhere.

Outline:

```python
data = json.load(open('tests/ver4/mock_data.json'))

for rdbms in ['pgsql', 'mysql']:
    cursor = RecordingCursor(select_ids=True)
    db = ...  # new PlateDB with the DR4 schema, its rdbms backend using cursor,
              # dr_num, hostname and version from data
    blocks = []

    for case in data['cases']:
        cursor.select_ids = case['lookups_find']
        start = len(cursor.queries)
        args = decode(case['args'])
        kwargs = decode(case['kwargs'])
        ADAPTERS[case['method']](db, *args, **kwargs)  # one per old method
        blocks.append((case['label'], cursor.queries[start:]))

    expected = open(f'tests/ver4/queries/{rdbms}/mock.sql').read()
    assert format_queries(rdbms, blocks) == expected
```

The new backends must pass `%s` placeholders and a parameter tuple to
`cursor.execute()` / `cursor.executemany()`, and handle `RETURNING` and
`SELECT` results as described above.

## Behaviour of the old code that the SQL depends on

The expected SQL contains these details. The new code must reproduce them,
or the difference must be a deliberate change, documented when it is made.

Statements and column order:

- `INSERT` columns: first the columns the method sets itself, then the table
  columns in the order of the YAML schema, limited to those present in the
  input. Fixed leading columns:
  - plate: `plate_id` only when given;
  - exposure: `exposure_num`;
  - exposure_sub: `exposure_id`, `plate_id`, `exposure_num`,
    `subexposure_num`;
  - scan: `plate_id`, then `scan_id` when given;
  - preview: `plate_id`, then `preview_id` when given;
  - solution_set, phot_calib, phot_calib_curve: `process_id`, `scan_id`,
    `plate_id`, `archive_id`;
  - solution, scanner_pattern: `solutionset_id`, `process_id`, `scan_id`,
    `plate_id`, `archive_id`;
  - solution_healpix: `solution_id`, `solutionset_id`, `process_id`,
    `scan_id`, `plate_id`, `archive_id`, `solution_num`;
  - source, source_calib, source_xmatch: `source_id`, `process_id`,
    `scan_id`, `plate_id`, `archive_id`;
  - process: fixed list `scan_id`, `plate_id`, `archive_id`, `filename`,
    `hostname`, `use_psf`, `pyplate_version`;
  - process_log: fixed list `process_id`, `scan_id`, `plate_id`,
    `archive_id`, `level`, `event`, `solution_num`, `message`.
- `RETURNING <id>` (PostgreSQL) on plate, exposure, scan, preview, logbook,
  logpage, solution_set, solution, phot_calib and process; not on
  exposure_sub, plate_logpage, scanner_pattern, phot_calib_curve,
  solution_healpix, source, source_calib, source_xmatch and process_log.
- `update_scan`: `SET` all scan columns except `scan_id`, in schema order,
  or only `file_datetime`, `file_size`, `fits_checksum`, `fits_datasum` with
  `filecols`; `WHERE scan_id=%s`.
- `update_process`: `SET` in the order of the keyword arguments (not schema
  order), skipping unknown names and `None` values; no statement if nothing
  is left.
- `write_process_end`: `timestamp_end=NOW()` is part of the SQL text.
- Sources and cross-matches: `executemany` in batches of 1000 rows (the mock
  data have 2 rows, so batch boundaries are not covered).

Lookups:

- Plate ID: `get_plate_id(plate_num, archive_id)`, then
  `get_plate_id_wfpdb(wfpdb_id)` if nothing was found. `write_plate_logpage`
  repeats the plate lookup for every logpage, followed by the logpage lookup.
- Lookups compare with `=%s` also when the value is `NULL`, which never
  matches in SQL. The redesigned code does not send a lookup whose key is
  `None` (DESIGN.md section 5.1). No lookup in `mock.sql` has a `NULL` key,
  so `mock.sql` is not affected.

Values:

- `write_plate`, MySQL only: a value in a column of type `DATE` (`date_orig`)
  before year 1000 or not a valid date is written as `NULL`.
- Astropy Quantity → its value; FITS header → `tostring(sep='\\n')`, so the
  cards are separated by a literal backslash-n; `True`/`False` → 1/0 (solution
  sets and solutions only).
- NaN and infinity → `NULL` in solution sets, solutions, sources, source
  calibration, cross-matches and HEALPix rows; not in other methods.
- `source_calib`: `healpix256` < 0 → `NULL`, `gaiaedr3_id` = 0 → `NULL`.
- `source_id` = dr_num × 10^(process_num_digits + source_num_digits) +
  process_id × 10^source_num_digits + source_num.
- `write_process_start`: `use_psf` → 1 or 0.

Table-creation SQL (`ddl/`):

- MySQL `drop_indexes.sql` is identical to the PostgreSQL one
  (`DROP INDEX IF EXISTS name`, which MySQL does not accept).
- PostgreSQL `DROP INDEX` statements have no schema prefix.

The redesigned `ddl.py` corrects both (DESIGN.md section 5.3): MySQL
`DROP INDEX name ON table`, and PostgreSQL with the schema prefix. The test
compares the other files exactly; for `drop_indexes.sql` it expects the
corrected statements, with the same index names in the same order.

## Plate header

`gen_mock_header.py` builds a plate header from mock metadata with the old
`PlateHeader`, in the same steps as `PlatePipeline.single_image()`:
`assign_conf()`, `assign_platemeta()`, `update_from_platemeta()`,
`assign_values()`, `update_comments()`, `rewrite()`, `reorder()`. The result
is `header/header.txt`. Then `insert_wcs()` adds a mock WCS header, and the
result is `header/header_wcs.txt`.

The files contain the header cards, one 80-character card per line
(`Header.tostring(sep='\n')`, with a final newline). `output_to_file()`
writes the same cards without separators, padded to FITS blocks.

The plate has two exposures, and every metadata key that has a FITS keyword
has a value. Generated values are distinct (integers 101, 102, ..., floats
201.25, 202.25, ..., strings `<key><exposure index>`), so a value under the
wrong keyword shows up as a difference. Some values are chosen to cover
special cases:

- a time with a colon (`TMS-ORIG = '21:30:00'`), which needs an astropy
  workaround;
- non-ASCII text (`OBSERVER = 'Jõgi Müller'`, written as `'Jogi Muller'`);
- a missing value in a list (`FOCUS2`, written as an empty card `FOCUS2 =`);
- acknowledgements with two paragraphs, wrapped at 72 characters;
- existing HISTORY lines.

### Fixed version and time

The header code writes the pyplate version and the current UTC time:

- `update_from_platemeta()` adds
  `HISTORY Header updated with PyPlate v<version> at <time>`;
- `insert_wcs()` adds `HISTORY WCS added with PyPlate v<version> at <time>`
  and sets `DATE` to the time.

The generator replaced the version with `mock-version` and the time with
`2001-02-03T04:05:06` (both in `mock_header.json`). The redesigned header
code takes them as the optional `version` and `now` arguments of
`PlateHeader.from_metadata()` and `insert_wcs()` (DESIGN.md section 4.5).

### mock_header.json

| Key | Contents |
|---|---|
| `pyplate_version`, `utcnow` | The fixed version and time |
| `plate` | `plate` columns, and `exposures`: one entry per exposure with `exposure_num` and the `exposure` columns. The header uses no sub-exposures. |
| `scan` | `scan` columns |
| `header_only` | Values without a database column, by old metadata key: `{"fits_keyword": ..., "value": ...}`, and `"derived": true` for values the new code derives from columns |
| `config` | Configuration sections used by the header code: `Integer values`, `Float values`, `String values`, `FITS keyword comments` |
| `wcs` | The WCS header inserted with `insert_wcs()`, as `[keyword, value, comment]` cards |

Column names are DR4 names, but the plate, exposure and scan columns used in
the header are the same in DR5, so the new record classes can be built from
them directly.

How the old code takes values from the metadata:

- A per-exposure value is written once under the main keyword with the value
  of exposure 1 (`DATEORIG`, `EXPTIME`, `RA`, ...) and, when there is more
  than one exposure, under numbered keywords for each exposure (`DATEOR1`,
  `DATEOR2`, ...). The `exposures` entries give the values in order.
- `TIMEFLAG` and `COORFLAG` come from the exposure columns `flag_time` and
  `flag_coord` of exposure 1.
- `DATEORIG` / `DATEORn` come from the exposure column `date_orig_start`.

Derived values (`"derived": true`) were set as the old `compute_values()`
does; the new code must derive them the same way:

| Keyword | Derived from | Rule |
|---|---|---|
| `RA`, `RAn` | exposure `ra_icrs` | `Angle(ra_icrs, deg).to_string(unit=hour, sep=':', precision=1, pad=True)` |
| `DEC`, `DECn` | exposure `dec_icrs` | `Angle(dec_icrs, deg).to_string(unit=deg, sep=':', precision=1, pad=True)` |
| `OBJTYPE`, `OBJTYPn` | exposure `object_type_code` | `Plate._object_types[code]` (object-type table in `codes.py` after the redesign) |
| `METHOD` | plate `method_code` | `Plate._method_codes[code]` (method table in `codes.py`) |

The old code computes the RA/Dec strings and `ra_icrs`/`dec_icrs` from the
same unrounded angle and rounds `ra_icrs`/`dec_icrs` to 4 decimals. Deriving
the strings from the rounded columns can therefore differ in the last digit
for real data. In the mock data they are consistent by construction.

The other `header_only` values have no column, and the redesign has to
decide where they come from (see DESIGN.md section 4.1):

| Old key | FITS keyword | Source after the redesign (DESIGN.md sections 3, 4.1, 4.5) |
|---|---|---|
| `fits_simple`, `fits_bitpix`, `fits_naxis`, `fits_bscale`, `fits_bzero`, `fits_extend` | `SIMPLE`, `BITPIX`, `NAXIS`, `BSCALE`, `BZERO`, `EXTEND` | Defaults in `PlateHeader`, overwritten from the FITS file when one is read; the test passes the values from `header_only` |
| `jd_avg_orig` | `JDA-ORIG`, `JDA-ORn` | New DR5 exposure column `jd_mid_orig` (one value per exposure, in order) |
| `detector`, `licence` | `DETNAM`, `LICENCE` | `[Header]` config section: `detector`, `licence` |
| `fits_acknowledgements` | `COMMENT` lines in the Acknowledgements section | `[Header]` config section: `acknowledgements` |
| `fn_scan` | `FN-SCNn` | Filenames of `plate.scans` |
| `fn_pre` | `FN-PRE` | Filename of the preview in `plate.previews` |
| `fn_cover`, `fn_log` | `FN-COVER`, `FN-LOGn` | Filenames of `plate.logpages` (cover first) |
| `date` | `DATE` | Set by `PlateHeader` from the current time (fixed in the test); `insert_wcs()` sets it again |
| `fits_history` | `HISTORY` | Existing history lines, taken from the FITS file; the test passes them from `header_only` |

### Behaviour of the old header code

The expected header contains these details; the new code must reproduce
them:

- Keyword order, section separators and section titles follow
  `PlateHeader._default_order`. Keywords not in that list (here the
  `MOCKINT`, `MOCKFLT`, `MOCKSTR` keywords from the configuration) come after
  the final separator.
- Comments come from `PlateHeader._default_comments`, with numbered keywords
  using the `n` entry (`EXPTIMn` → `EXPTIM2`); `[FITS keyword comments]` in
  the configuration overrides them (`OBSERVER`).
- `[Integer values]`, `[Float values]` and `[String values]` set keywords
  with fixed values.
- Strings are passed through `unidecode` (and German transliteration if
  `german_transliteration` is set; not set here).
- A missing value is written as an empty card (`KEY     =`, with comment);
  an empty string as `' '`.
- Empty COMMENT and HISTORY cards are dropped, so the empty line between
  acknowledgement paragraphs disappears.
- `insert_wcs()` inserts the WCS cards after the WCS separator, copies
  HISTORY cards, drops COMMENT cards and skips keywords already in the
  header (`NAXIS1` here).

### Intended differences in the header

The redesigned header writes `TZ-ORIG` from the new plate column `tz_orig`
(DESIGN.md sections 4.2 and 4.5); the old code never writes it.
`mock_header.json` has no `tz_orig`, so the test sets `plate.tz_orig = 'ST'`
and expects one extra card in `header.txt` and `header_wcs.txt`, between
`TME-OR2` and `JDA-ORIG`:

```
TZ-ORIG = 'ST      '           / time zone of the recorded time (ST = sidereal) 
```

The position and the comment are the ones the old keyword order and default
comments already have for `TZ-ORIG`. Other differences should be treated as
regressions unless they are deliberate and documented here.

### Writing the header test for the redesigned code

Outline:

```python
data = json.load(open('tests/ver4/header/mock_header.json'))
plate, scan = ...   # new Plate (with Exposure objects) and Scan from
                    # data['plate'] and data['scan']; plate.tz_orig = 'ST'
cfg = ...           # configuration with data['config'] and the header_only
                    # values that become configuration
header = PlateHeader.from_metadata(plate, scan, cfg, ...)   # with the
                    # fixed version and time, and the other header_only values
expected = with_tz_orig(open('tests/ver4/header/header.txt').read())
                    # the TZ-ORIG card inserted after TME-OR2
assert header.tostring(sep='\n') + '\n' == expected

header.insert_wcs(wcs_from(data['wcs']))
expected = with_tz_orig(open('tests/ver4/header/header_wcs.txt').read())
assert header.tostring(sep='\n') + '\n' == expected
```

The JSON was checked by rebuilding the old `Plate` from it alone and running
the old header code: both header files were reproduced exactly.

## Metadata regression

`gen_metadata_regression.py` reads five archives with the old `Archive`
(`read_wfpdb()` and/or `read_csv()`, then `get_platemeta()` for each plate,
`get_previewmeta()`, `get_logbookmeta()` and `get_logpagemeta()`), runs
`Plate.compute_values()`, and writes the results to `metadata/expected.json`.

### Inputs

| Archive | Configuration | Input files | Covers |
|---|---|---|---|
| `pot015` | `pot015.conf` | `wfpdb/POT015maindata_st.txt` | Real WFPDB maindata: 8 records of the AIP Potsdam series POT015 (Kapteyn Selected Areas). Sidereal times converted to UT, including a time `24:24:16`; precession of the recorded coordinates; heliocentric JD; exposure times in minutes; object type and method codes (direct photograph, objective prism); a record without time and exposure time and with time flag `M`; a non-ASCII emulsion field |
| `my_archive` | `my_archive.conf` | `csv/my_plates.csv`, `csv/my_scans.csv` | Copy of `tests/data`: times in UT, local time (`Europe/Berlin`) and sidereal time, given per row; a plate with two exposures, the second with two sub-exposures; four scans |
| `mock` | `mock_archive.conf` | `csv/mock_plates.csv` | Plates `m1`–`m5`: the `numexp` rules in DESIGN.md section 4.3. Plate `m6`: two exposures with their own coordinates, the second with two sub-exposures. `evening_date = true` |
| `full` | `full_archive.conf` | `wfpdb_full/` (maindata, quality, notes, observer), `csv/full_*.csv` (plates, scans, previews, logpages), `fits/`, `previews/`, `logpages/`, `covers/` | Mock WFPDB archive with all four WFPDB files: quality values (including a non-numeric temperature), notes with exposure times, objects, coordinates, focus, observer notes, recorded JD and free text, observer names; `[Keyword lookup]` (`j. mets` → `Jaan Mets`); a plate observed after midnight with `evening_date`; a plate number with suffix (`2A`); coordinate and time flags. Values read from FITS files (scan columns, minimum and maximum pixel value, file size and time); preview and logpage images (JPEG with and without EXIF date, PNG, TIFF), with and without `[Image] logpage_exif_timezone`; a cover image found in `cover_dir`; logbooks from configuration sections; plate–logpage links (`fn_cover`, `fn_log`) |
| `times` | `times_archive.conf` | `csv/times_plates.csv` | Start and end times given as UT or sidereal time in their own columns; local time zones; times after midnight with `evening_date`, including two exposures; a plate with both a local time and a UT column |

The POT015 records were selected from
`AIP/Mimosa/digiplates/pyheaders/POT015maindata_st.TXT`. That file is in
Latin-1, which the old `read_wfpdb()` cannot read (it opens files as UTF-8),
so the subset is stored as UTF-8. The non-ASCII emulsion field `÷àñà` is
probably Cyrillic text in cp1251 read as Latin-1. The redesigned WFPDB
reader takes the file encoding from `[WFPDB] encoding` (DESIGN.md section 3;
the CSV sections have `csv_encoding`). Because the subset is UTF-8, the test
configuration can leave the default.

`wfpdb_latin1/POT015maindata_st.txt` holds the same 8 records with the
original Latin-1 bytes. The old code cannot read it, so there is no separate
reference: the test reads it with `pot015.toml` (new format) plus
`encoding = latin-1`, and expects exactly the results of archive `pot015`.

`compute_values()` sets the key `date` (FITS keyword `DATE`) to the current
UTC time. The generator replaced it with `2001-02-03T04:05:06` (`utcnow` in
`expected.json`).

The old code takes `file_datetime` of scans, previews and logpages from the
file modification time, which git does not keep. The generator sets the
modification time of every file in `fits/`, `previews/`, `logpages/` and
`covers/` to `2002-03-04T05:06:07` UTC (`file_mtime` in `expected.json`)
before reading; the test must do the same. `make_metadata_inputs.py`
recreates these files (FITS with astropy, images with Pillow).

### expected.json

```
{"utcnow": ..., "file_mtime": ..., "archives": [
  {"name": ..., "conf": ..., "dirs": {"csv_dir": "csv", ...},
   "plates": [{"input_id": ..., "columns": {...}, "other": {...}}, ...],
   "scans": [{"input_filename": ..., "columns": {...}, "plate_num": ...,
              "other": {...}}, ...],
   "previews": [{"input_filename": ..., "columns": {...},
                 "input_plate_id": ..., "plate_num": ..., "wfpdb_id": ...}],
   "logbooks": [{"input_id": ..., "columns": {...}}],
   "logpages": [{"input_filename": ..., "columns": {...},
                 "logbook_num": ...}]}]}
```

- `dirs`: the `[Files]` directory settings the generator set, relative to
  `tests/ver4/metadata`.

- `input_id`: the plate identifier the old code used (its metadata key
  `plate_id`: the CSV identifier, or for WFPDB `POT015_000317`).
- `columns`: the values the old `PlateDB.write_plate()` would write, taken
  with `Plate.get_value()` exactly as `write_plate()` does, in DR4 column
  names: `plate` columns and `exposures`, each with `exposure_num`, the
  `exposure` columns and `subexposures` (`subexposure_num` and the
  `exposure_sub` columns). For scans, the columns the old `write_scan()`
  would write. These tables have the same columns in DR5.
- `other`: metadata keys without a database column that have a value, by old
  key (for example `ra`, `dec`, `object_type`, `method`, `tz_orig`,
  `st_start_orig`, `jd_avg_orig`, `fn_cover`, `fn_log`).
- Previews, logbooks, logpages: `columns` holds the values the old
  `write_preview()`, `write_logbook()` and `write_logpage()` would write
  (without the IDs these methods set or look up). For previews,
  `input_plate_id` is the plate identifier in the preview CSV, and
  `plate_num` and `wfpdb_id` are the keys used to look up the plate; for
  logpages, `logbook_num` is the key used to look up the logbook.
- NaN and infinity are written as `{"nan": true}` and `{"inf": 1 or -1}`
  (`sqlrender.decode()` converts them back).

How the old code fills exposures (from `Plate.get_value()`), visible in the
results:

- A list with one value is used for all exposures, except the computed time
  values (`ut_*`, `jd_*`, `year_*`, `hjd_*`), which are `None` for exposures
  after the first. For example, plate `m1` has one coordinate for three
  exposures, so `ra_icrs` is repeated, but `hjd_mid` is only set for
  exposure 1.
- A value that is not a list (for example `method_code`, `flag_time`) is
  written to every exposure.

Other details of the old code visible in the results:

- `flag_coord` from WFPDB maindata is not stripped (`' '` when empty), while
  `flag_time` is (`''`).
- A non-numeric WFPDB quality temperature (`n/a`) becomes `None`.
- Plate `t4` gives a local time and a UT time; the old code uses the local
  time (22:00 Berlin → 21:00 UT) and ignores the UT column (20:00).
- Image dates: without an EXIF time zone the EXIF date is kept as
  `2015-06-07 08:09:10`; with `[Image] logpage_exif_timezone` it is converted
  to UTC and written as `2016-01-02T08:20:30`.
- A TIFF file with a DateTime tag cannot be read: Pillow returns the tag as a
  tuple, which the old code indexes as a string. The TIFF input has no
  DateTime tag.

### Configuration for the redesigned code

The configuration files are in the old INI format. With the clean break
(DESIGN.md section 3) the test needs TOML equivalents (`pot015.toml`, ...),
made with `ini2toml` (DESIGN.md section 3.1), which applies these renames.
The renames that apply to these files:

| Section | Old key | New key |
|---|---|---|
| CSV sections | `plate_id` | `plate_num` |
| CSV sections | `date_orig` | `date_orig_start` |
| CSV sections | `tms_orig` | `time_orig_start` |
| CSV sections | `tme_orig` | `time_orig_end` |
| Scan CSV | `filename` | `filename_scan` |
| Scan CSV | `datescan` | `scan_date` |
| Preview CSV | `plate_id` | `plate_num` |

Unchanged: `numexp`, `exptime`, `observer`, `ra_orig`, `dec_orig`,
`object_name`, `object_type_code`, `scan_author`, `tz_orig` (now a plate
column in DR5, see DESIGN.md sections 4.2 and 5.4), the `[Keyword values]`
keys, `csv_list_delimiter`, `csv_timezone`, `evening_date`, and `[Archive]`.
In the plate CSV of the mock archive, `method_code` goes to the plate, which
wins over the exposure when both have the column (DESIGN.md section 4.2); the
old code wrote the same value to the plate and to every exposure.

Unchanged as well, although they are not columns (DESIGN.md sections 3 and
4.2): the plate CSV keys `fn_cover` and `fn_log` (logpage links) and
`ut_start_orig`, `ut_end_orig`, `st_start_orig`, `st_end_orig` (start and end
times as UT or sidereal time); `[WFPDB] notes_timezone`; the `[Image]`
settings `logpage_exif_timezone` and `preview_exif_timezone`; and in
`[Files]` the WFPDB file names (`wfpdb_maindata`, `wfpdb_quality`,
`wfpdb_notes`, `wfpdb_observer`) and the directories (`fits_dir`,
`preview_dir`, `logpage_dir`, `cover_dir`).

Other settings in these files (DESIGN.md sections 3 and 4.2):

- `[WFPDB] maindata_timezone` and `exptime_minutes` keep their names and
  meaning in the new `[WFPDB]` section.
- `my_archive.conf` has no `[Archive]` section, so `archive_id` is `None`.
  `ArchiveReader` accepts that for reading; only writing to the database
  requires `archive_id`.

### Intended differences

The redesigned code is expected to differ from `expected.json` here:

- `plate_num`: the old code leaves it `None` for CSV plates (the identifier
  is in its `plate_id` key, see `input_id`). With the new configuration it
  is the CSV identifier (`plate1`, `m1`, ...). For WFPDB plates the old value
  (`317`, from the WFPDB record) stays.
- Scans: `plate_num` is `None` in the old code; the new code links scans to
  plates by `plate_num`.
- `archive_id`: set by `ArchiveReader` for every record.
- `tz_orig`: in `other` in `expected.json`; in the redesigned code it is the
  plate column `tz_orig`, with the same values (`Europe/Berlin`, `UT`, `ST`).
  Exception: the old code sets it for WFPDB records only when a time is
  recorded, so POT015 record 004252 (no time) has none; with the file
  default of DESIGN.md section 4.2 it gets `ST`.

Other differences should be treated as regressions unless they are
deliberate and documented.

### Writing the metadata test for the redesigned code

Outline:

```python
data = json.load(open('tests/ver4/metadata/expected.json'))
set_file_times(data['file_mtime'])    # files in fits/, previews/,
                                      # logpages/, covers/
for archive in data['archives']:
    reader = ArchiveReader(new_format_config(archive))  # directories from
                                                        # archive['dirs']
    for expected in archive['plates']:
        plate = reader.get_plate(plate_num=...)         # from input_id
        plate.compute_values()  # with the clock fixed to data['utcnow']
        compare(plate, decode(expected['columns']))     # exposures and
                                                        # sub-exposures in order
        # derived values: plate.exposures[i].ra string etc. against
        # expected['other']['ra'][i] and so on
    # scans, previews, logbooks and logpages the same way

# Latin-1 input: pot015 with encoding = latin-1 on wfpdb_latin1/ gives the
# results of archive pot015
```

Compare values exactly. The old code rounds the computed values (JD to 5
decimals, years to 8); a change in rounding is a change in results.

## Reference plates

`run_reference_plates.py` runs the old pipeline end to end on two real
plates and saves its results, so that the redesigned pipeline can be run on
the same plates and compared. It needs the external programs (SExtractor,
PSFEx, SCAMP, astrometry.net), the astrometry.net index files and Gaia
files, the plate scans and the metadata CSV files, so it runs only on a
machine that has them.

### Plates

| Scan | Plate | Exposures | Header of the scan |
|---|---|---|---|
| `LA02333_x.fits` | Hamburg Lippert-Astrograph LA02333, field at RA 01:30.9, Dec +71°, 1928-08-19/20 | 4 × 480 s (method code 2) | Header and WCS written by pyplate 3.0.1 in 2017; kept unchanged (checksum in the manifest) |
| `LA02334_x.fits` | LA02334, same field and night | 1 × 480 s (method code 1) | Raw (10 cards) |

The old code reads metadata from the scan header before the CSV files. For
LA02333 this gives empty strings (`''`) instead of `None` for about 17 text
fields, the old `CHECKSUM`/`DATASUM` values and the old `HISTORY` lines; no
times, coordinates, exposure data or WCS values come from the header. The
redesigned FITS reader may store `None` instead of `''` for these: compare
them as equal, or document the difference.

### Configuration

| File | In git | Contents |
|---|---|---|
| `plates/plates.conf` | yes | The DR5 development configuration without paths and without the `[DB_pgsql]` section; `output_header_fits` off (the old code would call `PlateDB.update_scan()` without a database), `output_sources_csv` off (the full source table is saved instead) |
| `plates/paths.local` | no (`*.local` in `.gitignore`) | Machine-specific paths: `[Reference] output_dir` (where results go), the input directories in `[Files]`, `filter_path` in `[Solve]`, the programs in `[Programs]` |
| `plates/paths.local.example` | yes | Template for `paths.local` |

The driver reads `plates.conf` and then `paths.local`. It sets the output
directories of the pipeline (work, logs, headers, WCS) inside the run
directory, so runs do not touch other output.

### What the driver does

It runs `PlatePipeline.single_image()` for each scan. Unlike a script that
calls the `Process` steps directly, `single_image()` builds the plate header
from the metadata, so both plates get their epoch (1928.63) and `METHOD`
from the CSV metadata, not from the scan header (a raw scan would otherwise
get the configuration default 1970, for which there is no index file).

The pyplate code is not changed; methods are wrapped at run time:

| Wrapped | Saved |
|---|---|
| `StarCatalog.query_gaia_tap()` | Every Gaia query result (`gaia/gaia_NN.vot`, VOTable binary2) and its parameters (`gaia/queries.json`). With `--replay-gaia RUN` the saved results of another run are returned instead of querying the archive; the run stops if a query differs |
| `SourceTable.crossmatch_gaia()` | For every call `NN`: the Gaia stars (`crossmatch/xmatch_NN_catalog_in.fits`), the solutions used (`xmatch_NN_solution.json`), the match columns (`xmatch_NN_matches.fits`) and the neighbour table (`xmatch_NN_neighbors.fits`); for the first call also the source table before matching (`xmatch_01_sources_in.fits`) |
| `Process.finish()` | Before the scratch directory is removed: the full source table (`sources.fits`), the scanner pattern and HEALPix tables, and `summary.json` with the process values, all astrometric solutions (with WCS headers), the photometric calibration and its curves evaluated on 0.1 mag steps, and statistics of the main source columns |
| `PlateDB` | A recording database (`sqlrender.RecordingCursor`), so the result writes run without a database server. The scan lookup returns the mock IDs 9001/9002, the process gets ID 1, and all other lookups find nothing (in particular no plate epoch). `results.sql` holds all SQL; `results_small.sql` leaves out HEALPix and source tables |

Fixed values: the header and metadata time is `2001-02-03T04:05:06`, the
pyplate version written to headers and the process table is
`reference-version`, and the host name is `reference-host`. The process
duration in `results.sql` is real (the comparison ignores it), as are the
log times.

### Running

From the repository root, on the machine with the data:

```
PYTHONPATH=. python tests/ver4/run_reference_plates.py run1
PYTHONPATH=. python tests/ver4/run_reference_plates.py run2 --replay-gaia run1
python tests/ver4/compare_reference_plates.py run1 run2
PYTHONPATH=. python tests/ver4/run_reference_plates.py run1 --publish
```

`run1` queries the Gaia archive (TAP, `gaiaedr3.gaia_source` at
gaia.aip.de); `run2` reuses those results, so differences between the runs
come from the pipeline only. A run directory is never overwritten.

Result of the first two runs (2026-10-16): with the same Gaia results, the
old pipeline is deterministic. All source columns, all crossmatch outputs,
all values in `summary.json` and all lines of `results.sql` are identical,
except in the FITS header written by astrometry.net for each solution
(`header_anet`): its `DATE` card, and `COMMENT` cards with the work directory
and the CPU time. The comparison ignores `COMMENT`, `HISTORY` and `DATE`
cards and the process duration, and then finds no differences. The
redesigned code is therefore compared exactly, apart from the intended
differences below.

| Plate | Sources | Solutions | Calibration iterations | Gaia queries | Crossmatch calls |
|---|---|---|---|---|---|
| LA02333 | 29,969 | 4 | 4 | 3 | 3 |
| LA02334 | 18,714 | 1 | 4 | 3 | 3 |

### Where the results are

| Location | Contents |
|---|---|
| `<output_dir>/<run>/` (outside git) | Everything: `manifest.json`, and per plate `sources.fits`, `summary.json`, `results.sql`, `results_small.sql`, `header.txt`, `crossmatch/`, `gaia/`, logs, headers, WCS files |
| `plates/reference/` (in git, made with `--publish`) | The manifest and, per plate, `summary.json`, `header.txt`, `results_small.sql` and `gaia/queries.json` (about 640 kB in all). The `COMMENT` and `HISTORY` cards of the stored solution headers are left out (they hold local paths, split over several cards by astrometry.net), and other local paths are replaced by placeholders such as `${output_dir}`; publishing stops if a home-directory path remains |

The manifest records the pyplate commit, the Python, package and program
versions, and SHA-256 checksums of the inputs (scans, metadata CSV files,
`plates.conf`, the astrometry.net index files of the plate year,
`gaiaedr3_pyplate.fits`) and of every output file, by name only. It shows
whether a later run used the same inputs and whether the external results
are still the ones that were recorded. Back up the external directory; it is
not in git.

### Comparing the redesigned code

The redesigned pipeline is run on the same scans, with `--replay-gaia` of
the reference run or an equivalent replay of `gaia/gaia_NN.vot`, and its
results are written in the same layout. Its configuration is made with
`python -m pyplate.config.ini2toml plates/plates.conf plates/paths.local <file>.toml`;
the result holds the local paths, so it stays outside git, like
`paths.local`. `compare_reference_plates.py
--dr5` maps the DR5 column names (`gaiadr4_*`, `num_gaia_dr4`) back to DR4
before comparing, including `gaiadr4_gmag` to the misspelt DR4 column
`gaiaiedr3_gmag` in the result SQL.

Intended differences:

- Crossmatch epoch: the old `extract_sources()` creates the `SourceTable`
  without the configuration, so `crossmatch_gaia()` moves the Gaia stars to
  the default epoch 1950, not to the plate epoch (1928.63); see
  `source_table_plate_epoch` in `xmatch_NN_solution.json`. The redesigned code
  uses the plate epoch, so matches change for stars with large proper motion,
  and with them the calibration.
- DR5 column names, `tz_orig` and `TZ-ORIG`, and the metadata differences of
  section "Metadata regression".
- For LA02333, empty strings from the scan header (see "Plates").
