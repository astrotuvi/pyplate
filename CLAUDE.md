# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Memory of Agreements

**Once an assumption is confirmed, it is binding for the session.**

- If a previous answer established a convention (debounce timing, naming pattern, error format), apply it to all subsequent related work without re-asking.
- If you need to break a previous agreement, state which one you’re breaking and why, before implementing.
- At session start, surface any unresolved decisions from prior turns.

## What this is

PyPlate is a Python package for processing scanned astronomical photographic plates: metadata ingestion, FITS header generation, source extraction, astrometric solving, photometric calibration, and writing results to a SQL database (APPLAUSE schema).

## Commands

```bash
pip install -e .            # basic install
pip install -e .[ml]        # + scikit-learn/tensorflow (artifact classification)
pip install -e .[pgsql]     # + psycopg2; [mysql] for pymysql

pytest tests/test_metadata.py                        # the only runnable pytest suite
pytest tests/test_metadata.py::test_plate_ut         # single test

cd docs && make html        # Sphinx docs (published on readthedocs)
```

`tests/test_db_*.py` and `tests/test_schema.py` are stale scripts (they import `pyplate.config.local` and `pyplate.db_pgsql`, which no longer exist) and need a live database; don't treat their failures as regressions.

The version string lives in `pyplate/_version.py`; `setup.py` parses its last line.

## Architecture

Four subpackages/modules, all configured from a single INI file read via `conf.read_conf()` (a `ConfigParser`). Nearly every class has an `assign_conf(conf)` method that accepts either a path or a `ConfigParser`, and pulls its attributes from sections like `[Archive]`, `[Files]`, `[Programs]`, `[Database]`, `[Keyword values]`, plus per-CSV-file sections describing column mappings. See `docs/configuration.rst` and `tests/data/my_archive.conf`.

- **`metadata.py`**: `Archive` reads plate/scan/logbook metadata from WFPDB files or CSVs and produces `Plate` objects (OrderedDicts). `Plate.compute_values()` derives UT/JD/sidereal times, exposures, etc. from the original logbook data (multi-exposure plates store per-exposure lists). `PlateHeader` (subclass of `astropy.io.fits.Header`) builds a standardised FITS header from a `Plate`.
- **`image.py`**: `PlateConverter` converts raw scans (TIFF) to FITS and previews.
- **`process/`**: `Process` (`process.py`) is the per-image orchestrator. It composes `SourceTable` (`sources.py`), `StarCatalog` (`catalog.py`, Gaia/Tycho-2/UCAC4/APASS references), `SolveProcess`/`PlateSolution` (`solve.py`, astrometry incl. multi-exposure pattern finding), and `PhotometryProcess` (`photometry.py`). `pipeline.py`'s `PlatePipeline.single_image()` is the canonical end-to-end flow (metadata → header → `setup` → `extract_sources` → `classify_artifacts` → `solve_plate` → crossmatch → `calibrate_photometry` → DB/CSV output → `finish`); `parallel_run()` runs it over many files with multiprocessing.
- **`database/`**: `PlateDB` (`database.py`) is a backend-neutral facade that delegates to `db_pgsql` or `db_mysql` based on `rdbms`. Table definitions are not hard-coded: they come from a YAML schema (`applause_dr4.yaml`) parsed by `db_yaml.py`, which also generates CREATE/DROP SQL per backend. Schema changes go in the YAML.

### External dependencies

Processing shells out to external programs via `subprocess`: SExtractor (`sex`), PSFEx, SCAMP, and astrometry.net (`solve-field`, `wcs-to-tan`). Paths are overridable in the `[Programs]` config section. Large reference catalogues and astrometry.net index files are read from directories in `[Files]`.

Artifact classification uses a bundled Keras 3 model (`process/artifact_model.keras`; `artifact_model.h5` is the legacy format). Keras is an optional import guarded by `have_keras`. Note that `MANIFEST.in` currently only lists the `.h5` file.

### Logging

`ProcessLog` writes to a per-plate log file and, when `enable_db_log` is set, to the `process_log` DB table. `log.write()` calls carry numeric `level` and `event` codes that are stored in the database, so keep existing event numbers stable.
