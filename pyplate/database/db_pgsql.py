"""PostgreSQL backend (psycopg2)."""
import time

from .backend import Backend

try:
    import psycopg2
except ImportError:
    psycopg2 = None


class PgsqlBackend(Backend):
    """PostgreSQL. The connection is retried every 10 s until it opens."""

    rdbms = 'pgsql'

    def connect(self) -> None:
        if psycopg2 is None:
            raise ImportError('PostgreSQL needs psycopg2 '
                              '(pip install pyplate[pgsql])')

        cfg = self.cfg

        while True:
            try:
                self.connection = psycopg2.connect(
                    user=cfg.user, password=cfg.password, host=cfg.host,
                    port=cfg.port, database=cfg.database)
                break
            except psycopg2.Error as e:
                print('PostgreSQL server reports an exception, trying again')
                print(e.args)
                time.sleep(10)

        self.cursor = self.connection.cursor()
