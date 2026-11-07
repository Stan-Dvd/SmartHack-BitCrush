from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from .config import settings

pool = ConnectionPool(
    conninfo=settings.database_url,
    min_size=1,
    max_size=10,
    kwargs={"row_factory": dict_row},
    open=False,
)

def open_pool():
    pool.open()

def close_pool():
    pool.close()

def get_connection():
    return pool.connection()
