"""Attributes that the process classes take from the configuration.

Process, SolveProcess and SourceTable set these attributes in their
``assign_conf()``. Step 2 of the redesign replaces them with ProcessContext
(DESIGN.md section 6.1).
"""
import dataclasses
from typing import Any

from ..config import Config

# [Files] directories used by the process classes
FILES = ['fits_dir', 'index_dir', 'gaia_dir', 'work_dir', 'write_log_dir',
         'write_phot_dir', 'write_source_dir', 'write_wcs_dir',
         'write_db_source_dir', 'write_db_source_calib_dir',
         'write_db_source_xmatch_dir', 'write_db_solution_healpix_dir']


def conf_attributes(conf: Config) -> dict[str, Any]:
    """Attribute values from the configuration, by attribute name."""
    attrs = {'archive_id': conf.archive.archive_id}
    attrs.update(dataclasses.asdict(conf.programs))
    attrs.update({k: getattr(conf.files, k) for k in FILES})
    attrs.update(dataclasses.asdict(conf.extraction))
    attrs.update(dataclasses.asdict(conf.solve))
    attrs.update(dataclasses.asdict(conf.crossmatch))
    db = conf.database
    attrs.update({'enable_db_log': db.enable_db_log,
                  'write_sources_csv': db.write_sources_csv,
                  'write_solution_healpix_csv': db.write_solution_healpix_csv,
                  'output_db': db.connection is not None,
                  # The Gaia file is always used
                  'use_gaia_fits': True})

    if attrs['write_log_dir']:
        attrs['enable_log'] = True

    return attrs
