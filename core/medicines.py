"""Autocomplete over the bundled offline medicine-name index.

The index (an FTS5 sqlite DB at MEDICINES_DB_PATH) is built once, offline, by
scripts/build_medicine_index.py from the India, Singapore and US datasets plus a
generic (active-ingredient) backbone. At runtime this module only reads it — no
network — so the medication flow stays on-device.

Schema (one FTS5 table):
    name    TEXT   brand or generic name (indexed, prefix-searchable)
    generic TEXT   active ingredient(s) (indexed)
    region  TEXT   IN | SG | US | generic   (UNINDEXED, used as a filter)
    mtype   TEXT   brand | generic          (UNINDEXED)
"""
import re
import sqlite3
from pathlib import Path

from core.config import MEDICINES_DB_PATH

_TOKEN_RE = re.compile(r'[A-Za-z0-9]+')
VALID_REGIONS = {'IN', 'SG', 'US', 'generic'}


def is_available() -> bool:
    return Path(MEDICINES_DB_PATH).exists()


def _connect() -> sqlite3.Connection | None:
    if not is_available():
        return None
    conn = sqlite3.connect(f'file:{MEDICINES_DB_PATH}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _match_expr(query: str) -> str | None:
    """Turn a user query into a safe FTS5 prefix-match expression.

    Only alphanumeric tokens are kept (dropping quotes/operators that could be
    interpreted as FTS syntax); each becomes a prefix term, AND-ed together.
    """
    tokens = _TOKEN_RE.findall(query)
    if not tokens:
        return None
    return ' '.join(f'{t}*' for t in tokens)


def search(query: str, region: str | None = None, limit: int = 20) -> list[dict]:
    """Prefix-search medicine names. Returns [{name, generic, region, type}, ...].

    `region` (IN/SG/US) restricts brand results to that country; generic-backbone
    rows (region='generic') are always included so an active-ingredient search
    works regardless of country. Empty list if the index isn't built.
    """
    match = _match_expr(query)
    if match is None:
        return []
    conn = _connect()
    if conn is None:
        return []
    sql = ['SELECT name, generic, region, mtype FROM medicines '
           'WHERE medicines MATCH ?']
    params: list = [match]
    if region:
        sql.append('AND (region = ? OR region = ?)')
        params += [region, 'generic']
    sql.append('ORDER BY bm25(medicines) LIMIT ?')
    params.append(limit)
    try:
        rows = conn.execute(' '.join(sql), params).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [{'name': r['name'], 'generic': r['generic'],
             'region': r['region'], 'type': r['mtype']} for r in rows]
