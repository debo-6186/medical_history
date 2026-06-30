"""Local pending-reminders store (stdlib sqlite3).

A reminder is *drafted* here first — by the chat scheduling node, by medication
capture (PR 3), or by follow-up extraction on ingest (PR 4) — and only pushed to
Google Calendar after the user confirms it via the REST API. Drafting and
listing need no Google auth; this keeps the privacy-sensitive outbound step
behind an explicit human approval and out of the MODEL_LOCK span.

Schema (one table, `pending_reminders`):
    id            INTEGER  primary key
    kind          TEXT     'medication' | 'followup' | 'manual'
    title         TEXT     full, local title (never truncated here)
    proposed_text TEXT     the human-editable text shown for confirmation
    start         TEXT     ISO 8601 start datetime (local), nullable
    rrule         TEXT     RFC-5545 RRULE (recurrence), nullable
    status        TEXT     'pending' | 'scheduled' | 'dismissed'
    gcal_event_id TEXT     set once pushed to Google Calendar
    created_at    TEXT     ISO 8601 UTC
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from core.config import REMINDERS_DB_PATH

VALID_KINDS = {'medication', 'followup', 'manual'}
VALID_STATUSES = {'pending', 'scheduled', 'dismissed'}

_COLUMNS = (
    'id', 'kind', 'title', 'proposed_text', 'start', 'rrule', 'status',
    'gcal_event_id', 'created_at',
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    Path(REMINDERS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(REMINDERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        '''CREATE TABLE IF NOT EXISTS pending_reminders (
               id            INTEGER PRIMARY KEY AUTOINCREMENT,
               kind          TEXT NOT NULL,
               title         TEXT NOT NULL,
               proposed_text TEXT NOT NULL,
               start         TEXT,
               rrule         TEXT,
               status        TEXT NOT NULL DEFAULT 'pending',
               gcal_event_id TEXT,
               created_at    TEXT NOT NULL
           )'''
    )
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {col: row[col] for col in _COLUMNS}


def add(kind: str, title: str, proposed_text: str,
        start: str | None = None, rrule: str | None = None) -> dict:
    """Insert a 'pending' reminder and return it as a dict."""
    if kind not in VALID_KINDS:
        raise ValueError(f'invalid kind: {kind!r}')
    with _connect() as conn:
        cur = conn.execute(
            '''INSERT INTO pending_reminders
               (kind, title, proposed_text, start, rrule, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'pending', ?)''',
            (kind, title, proposed_text, start, rrule, _now()),
        )
        row = conn.execute(
            'SELECT * FROM pending_reminders WHERE id = ?', (cur.lastrowid,)
        ).fetchone()
    return _row_to_dict(row)


def list_reminders(status: str | None = None) -> list[dict]:
    """All reminders (newest first), optionally filtered by status."""
    with _connect() as conn:
        if status is not None:
            rows = conn.execute(
                'SELECT * FROM pending_reminders WHERE status = ? '
                'ORDER BY id DESC', (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                'SELECT * FROM pending_reminders ORDER BY id DESC'
            ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get(reminder_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            'SELECT * FROM pending_reminders WHERE id = ?', (reminder_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def update_fields(reminder_id: int, **fields) -> dict | None:
    """Patch editable columns (proposed_text, title, start, rrule). Used when
    the user edits a proposal before confirming."""
    editable = {'title', 'proposed_text', 'start', 'rrule'}
    sets = {k: v for k, v in fields.items() if k in editable}
    if not sets:
        return get(reminder_id)
    assignments = ', '.join(f'{k} = ?' for k in sets)
    with _connect() as conn:
        conn.execute(
            f'UPDATE pending_reminders SET {assignments} WHERE id = ?',
            (*sets.values(), reminder_id),
        )
    return get(reminder_id)


def mark_scheduled(reminder_id: int, gcal_event_id: str) -> dict | None:
    """Record that the reminder was pushed to Google Calendar."""
    with _connect() as conn:
        conn.execute(
            "UPDATE pending_reminders SET status = 'scheduled', "
            'gcal_event_id = ? WHERE id = ?',
            (gcal_event_id, reminder_id),
        )
    return get(reminder_id)


def mark_dismissed(reminder_id: int) -> dict | None:
    with _connect() as conn:
        conn.execute(
            "UPDATE pending_reminders SET status = 'dismissed' WHERE id = ?",
            (reminder_id,),
        )
    return get(reminder_id)
