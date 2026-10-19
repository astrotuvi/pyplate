"""SQLite backend (standard library sqlite3)."""
import sqlite3

from .backend import Backend

# Seconds to wait for a lock held by another process
BUSY_TIMEOUT = 60


class SqliteBackend(Backend):
    """SQLite 3.35 or later (for INSERT ... RETURNING).

    Tables have no schema prefix. The database is opened in WAL mode with a
    busy timeout, so parallel workers can write one after another.
    Timestamps are stored as ISO text.
    """

    rdbms = 'sqlite'
    placeholder = '?'
    now_sql = 'CURRENT_TIMESTAMP'

    def __init__(self, cfg=None, schema_name=None):
        super().__init__(cfg, schema_name=None)

    def connect(self) -> None:
        if sqlite3.sqlite_version_info < (3, 35):
            raise RuntimeError('SQLite 3.35 or later is needed, found {}'
                               .format(sqlite3.sqlite_version))

        if not self.cfg.file:
            raise ValueError('[Database.connection] file is needed for SQLite')

        self.connection = sqlite3.connect(self.cfg.file, timeout=BUSY_TIMEOUT)
        self.cursor = self.connection.cursor()

        if self.cfg.file != ':memory:':
            self.cursor.execute('PRAGMA journal_mode=WAL')
