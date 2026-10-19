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

Python 3.12 or later. On this machine the conda env `applause` matches the versions of the reference run (tests/ver4/plates/reference/manifest.json); the system `python3` is too old.

```bash
pip install -e .            # basic install
pip install -e .[ml]        # + scikit-learn/tensorflow (artifact classification)
pip install -e .[pgsql]     # + psycopg2; [mysql] for pymysql

pytest tests                                         # all tests (no database server needed)
pytest tests/test_metadata.py::test_plate_ut         # single test

python -m pyplate.config.ini2toml old.conf new.toml  # convert an old INI configuration

cd docs && make html        # Sphinx docs (published on readthedocs)
```

The reference-plate comparison (DESIGN.md 7.1) needs the external programs and data, and runs only on the machine with `tests/ver4/plates/paths.local`: `PYTHONPATH=. python tests/run_reference_plates.py <run> --replay-gaia run1`, then `python tests/ver4/compare_reference_plates.py run1 <run> --dr5`.

The version string lives in `pyplate/_version.py`; `pyproject.toml` reads it via `[tool.setuptools.dynamic]`.

## Architecture

The package is being redesigned in steps (DESIGN.md, section 8; `tests/ver4` holds the output of the old code that the new code is compared with). Configuration is a TOML file read by `pyplate.config.load_config()` into frozen dataclasses (`Config` with one dataclass per section); every key is validated. See `docs/configuration.rst` and `tests/data/my_archive.toml`.

- **`config/`**: `sections.py` (the dataclasses), `loader.py` (TOML → `Config`, key and type checks), `ini2toml.py` (converts old INI configurations).
- **`metadata/`**: record classes (`Archive`, `Plate`, `Exposure`, `SubExposure`, `Scan`, `Preview`, `Logbook`, `Logpage`), dataclasses whose fields are exactly the columns of their DR5 tables (`DbRecord.to_row()`/`from_row()`), plus related records (`plate.exposures`, `plate.scans`, ...). `ArchiveReader` reads WFPDB and CSV files (and the scan FITS file) into them; `Plate.compute_values()` computes UT/JD/HJD and ICRS coordinates per exposure (`timeutils.py`). `PlateHeader.from_metadata()` builds the FITS header from the keyword table in `header.py`.
- **`image.py`**: `PlateConverter` converts raw scans (TIFF) to FITS and previews.
- **`process/`**: `Process` (`process.py`) is the per-image orchestrator. It composes `SourceTable` (`sources.py`), `StarCatalog` (`catalog.py`, Gaia), `SolveProcess`/`PlateSolution` (`solve.py`, astrometry incl. multi-exposure pattern finding), and `PhotometryProcess` (`photometry.py`). Their `assign_conf(cfg)` take a `Config` (`settings.py`, until `ProcessContext` replaces them in step 2). `pipeline.py`'s `PlatePipeline.single_image()` is the canonical end-to-end flow (metadata → header → `setup` → `extract_sources` → `classify_artifacts` → `solve_plate` → crossmatch → `calibrate_photometry` → DB/CSV output → `finish`); `parallel_run()` runs it over many files with multiprocessing.
- **`database/`**: `PlateDB` (`database.py`) writes records and results the same way for every backend (`db_pgsql`, `db_mysql`, `db_sqlite`; `backend.py`). Table definitions come from the backend-neutral schema YAML (`applause_dr5.yaml`; `applause_dr4.yaml` is kept to check the generated SQL against the old code), read by `schema.py`; `ddl.py` generates the CREATE/DROP/INDEX/TRIGGER SQL per backend. Schema changes go in the YAML.

### External dependencies

Processing shells out to external programs via `subprocess`: SExtractor (`sex`), PSFEx, SCAMP, and astrometry.net (`solve-field`, `wcs-to-tan`). Paths are overridable in the `[Programs]` config section. Large reference catalogues and astrometry.net index files are read from directories in `[Files]`.

Artifact classification uses a bundled Keras 3 model (`process/artifact_model.keras`). Keras is an optional import guarded by `have_keras`.

### Logging

`ProcessLog` writes to a per-plate log file and, when `enable_db_log` is set, to the `process_log` DB table. `log.write()` calls carry numeric `level` and `event` codes that are stored in the database, so keep existing event numbers stable.
