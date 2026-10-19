"""Base class of the record classes (one dataclass per database table)."""
import dataclasses
from typing import Any, ClassVar, Self


def related(**kwargs: Any) -> Any:
    """A dataclass field that is not a column (related records, values
    derived during computations). Defaults to an empty list."""
    if 'default' not in kwargs:
        kwargs.setdefault('default_factory', list)

    return dataclasses.field(metadata={'column': False}, **kwargs)


class DbRecord:
    """Base of the record classes.

    The dataclass fields of a subclass are the columns of its table
    (:attr:`table`), apart from fields made with :func:`related`.
    """

    table: ClassVar[str]

    @classmethod
    def columns(cls) -> list[str]:
        """Names of the fields that are columns, in field order."""
        return [f.name for f in dataclasses.fields(cls)
                if f.metadata.get('column', True)]

    def to_row(self) -> dict[str, Any]:
        """Return the columns as a dictionary."""
        return {c: getattr(self, c) for c in self.columns()}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Self:
        """Create a record from a row; keys that are not columns are
        ignored (for example ``timestamp_insert``)."""
        cols = set(cls.columns())
        return cls(**{k: v for k, v in row.items() if k in cols})
