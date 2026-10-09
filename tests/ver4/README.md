# DR4 database SQL reference

This directory holds output of the code before the redesign (DESIGN.md),
with the inputs needed to produce it again: the SQL that the database code
sends for the `applause_dr4` schema, and a plate FITS header. After the
redesign, tests run the same inputs through the new code and check that the
SQL and the header are unchanged.

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
```

After the redesign, check out the old code in a separate worktree to run
them, for example `git worktree add ../pyplate-dr4 ed2a193`, and copy the
`tests/ver4` scripts there.

The rest of this file first describes the SQL reference, then the header
reference (section "Plate header").

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
  matches in SQL.

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
code needs a way to set both, for example arguments or attributes that
default to the package version and the current time.

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

| Old key | FITS keyword | Likely source after the redesign |
|---|---|---|
| `fits_simple`, `fits_bitpix`, `fits_naxis`, `fits_bscale`, `fits_bzero`, `fits_extend` | `SIMPLE`, `BITPIX`, `NAXIS`, `BSCALE`, `BZERO`, `EXTEND` | FITS file structure (FITS reader) |
| `jd_avg_orig` | `JDA-ORIG`, `JDA-ORn` | Recorded Julian date per exposure. DESIGN.md treats it as a parsing intermediate, but the header writes it, so it needs a place |
| `detector`, `licence` | `DETNAM`, `LICENCE` | Configuration |
| `fits_acknowledgements` | `COMMENT` lines in the Acknowledgements section | Configuration |
| `fn_scan` | `FN-SCNn` | Filenames of the scans of the plate |
| `fn_pre` | `FN-PRE` | Preview filename |
| `fn_cover`, `fn_log` | `FN-COVER`, `FN-LOGn` | Logpage filenames (cover first) |
| `date` | `DATE` | File change date; `insert_wcs()` overwrites it with the current time |
| `fits_history` | `HISTORY` | Existing history lines |

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

### Writing the header test for the redesigned code

Outline:

```python
data = json.load(open('tests/ver4/header/mock_header.json'))
plate, scan = ...   # new Plate (with Exposure objects) and Scan from
                    # data['plate'] and data['scan']
cfg = ...           # configuration with data['config'] and the header_only
                    # values that become configuration
header = PlateHeader.from_metadata(plate, scan, cfg, ...)   # with the
                    # fixed version and time, and the other header_only values
expected = open('tests/ver4/header/header.txt').read()
assert header.tostring(sep='\n') + '\n' == expected

header.insert_wcs(wcs_from(data['wcs']))
expected = open('tests/ver4/header/header_wcs.txt').read()
assert header.tostring(sep='\n') + '\n' == expected
```

The JSON was checked by rebuilding the old `Plate` from it alone and running
the old header code: both header files were reproduced exactly.
