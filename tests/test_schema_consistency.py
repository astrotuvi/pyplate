"""The record classes have exactly the columns of their DR5 tables."""
import pytest

from pyplate.database.schema import TIMESTAMP_COLUMNS, load_schema
from pyplate.metadata import (Archive, Exposure, Logbook, Logpage, Plate,
                              Preview, Scan, SubExposure)

SCHEMA = load_schema('applause_dr5.yaml')


@pytest.mark.parametrize('cls', [Archive, Plate, Exposure, SubExposure, Scan,
                                 Preview, Logbook, Logpage])
def test_record_columns(cls):
    table = SCHEMA[cls.table]
    assert cls.columns() == [c for c in table.column_names
                             if c not in TIMESTAMP_COLUMNS]


@pytest.mark.parametrize('cls', [Archive, Plate, Exposure, SubExposure, Scan,
                                 Preview, Logbook, Logpage])
def test_row_round_trip(cls):
    record = cls(**{c: i for i, c in enumerate(cls.columns())})
    assert cls.from_row(record.to_row()) == record
