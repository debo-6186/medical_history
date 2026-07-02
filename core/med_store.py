"""Structured store of the medications the user is actually taking.

Populated by the medication-capture screen (never auto-read from a prescription).
Each saved medication also drafts a pending 'medication' reminder (see the
/api/medications route); this table keeps the full structured record — dosage,
frequency, tenure — for "what am I currently on" queries and future cancellation,
none of which should leave the device.

Personal data → lives under rag_db/ (git-ignored).

Schema (one table, `medications`):
    id          INTEGER
    name        TEXT   medicine name (as picked from the index)
    generic     TEXT   active ingredient(s), if known
    region      TEXT   IN | SG | US
    dosage      TEXT   free text, e.g. "500 mg"
    frequency   TEXT   e.g. "once daily", "twice daily"
    tenure      TEXT   free text, e.g. "7 days", "ongoing"
    start_date  TEXT   ISO date the course starts
    source_doc_id TEXT the prescription this was entered against, if any
    reminder_id INTEGER  the drafted pending reminder
    created_at  TEXT
"""
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from core.config import MED_STORE_DB_PATH

_COLUMNS = (
    'id', 'name', 'generic', 'region', 'dosage', 'frequency', 'tenure',
    'timing', 'start_date', 'source_doc_id', 'reminder_id', 'created_at',
)

_UNIT_DAYS = {'day': 1, 'week': 7, 'month': 30, 'year': 365}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    Path(MED_STORE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(MED_STORE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        '''CREATE TABLE IF NOT EXISTS medications (
               id            INTEGER PRIMARY KEY AUTOINCREMENT,
               name          TEXT NOT NULL,
               generic       TEXT,
               region        TEXT,
               dosage        TEXT,
               frequency     TEXT,
               tenure        TEXT,
               timing        TEXT,
               start_date    TEXT,
               source_doc_id TEXT,
               reminder_id   INTEGER,
               created_at    TEXT NOT NULL
           )'''
    )
    # Migrate DBs created before the timing column existed.
    cols = {row[1] for row in conn.execute('PRAGMA table_info(medications)')}
    if 'timing' not in cols:
        conn.execute('ALTER TABLE medications ADD COLUMN timing TEXT')
    return conn


def _row(row: sqlite3.Row) -> dict:
    return {c: row[c] for c in _COLUMNS}


def add(name: str, *, generic: str = '', region: str = '', dosage: str = '',
        frequency: str = '', tenure: str = '', timing: str = '',
        start_date: str | None = None, source_doc_id: str | None = None,
        reminder_id: int | None = None) -> dict:
    with _connect() as conn:
        cur = conn.execute(
            '''INSERT INTO medications
               (name, generic, region, dosage, frequency, tenure, timing,
                start_date, source_doc_id, reminder_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
            (name, generic, region, dosage, frequency, tenure, timing,
             start_date, source_doc_id, reminder_id, _now()),
        )
        row = conn.execute('SELECT * FROM medications WHERE id = ?',
                           (cur.lastrowid,)).fetchone()
    return _row(row)


def list_medications() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            'SELECT * FROM medications ORDER BY id DESC').fetchall()
    return [_row(r) for r in rows]


def tenure_to_count(tenure: str, frequency: str) -> int:
    """Occurrence COUNT for the reminder's RRULE from a free-text tenure.

    Returns 0 (open-ended) when no duration is stated. Daily-type frequencies
    count in days; weekly counts in weeks. "twice daily" etc. still yields a
    single daily series (a documented limitation — split intake times later).
    """
    m = re.search(r'(\d+)\s*(day|week|month|year)', (tenure or '').lower())
    if not m:
        return 0
    days = int(m.group(1)) * _UNIT_DAYS[m.group(2)]
    if 'week' in (frequency or '').lower():
        return max(1, round(days / 7))
    return days
