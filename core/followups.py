"""Follow-up extraction: turn a just-ingested record into pending reminders.

When a record is saved (free-text history) or a document is ingested (report /
prescription), this scans the text for FUTURE actions the patient must schedule
for themselves — "repeat CMV after 4 months", "next HRCT in 6 months", "review
in 2 weeks" — and drafts each as a *pending* reminder (core/reminders.py). It
does NOT push anything to Google: background ingest can't prompt the user, so the
drafts are surfaced later for approval via GET /api/reminders.

The due date is resolved against the relevant date IN the record (the visit or
report date) when one is present — e.g. seen 2026-05-26 + "after 4 months" ->
2026-09-26 — falling back to Today otherwise.

Every model call here MUST run with MODEL_LOCK held (like the rest of the app).
"""
import json
from datetime import date

import ollama

from core import reminders
from core.calendar import build_rrule
from core.config import CHAT_MODEL, CHAT_NUM_CTX, MODEL_KEEP_ALIVE

FOLLOWUP_EXTRACT_PROMPT = (
    'You read a personal medical record and extract any FUTURE follow-up '
    'actions the patient must schedule for themselves — repeating a test, a '
    'next scan, a review visit, or a "recheck in N weeks/months" instruction. '
    'Reply with ONLY a JSON object:\n'
    '  {"followups": [{"title": "<short action>", "date": "<YYYY-MM-DD>", '
    '"frequency": "once|daily|weekly|monthly", "count": <integer>}]}\n\n'
    'date — resolve the interval against the RELEVANT date in the record (the '
    'visit or report date) when the record gives one, NOT against Today. '
    'Example: record says "saw doctor on 2026-05-26 ... repeat CMV after 4 '
    'months" -> date 2026-09-26. If the record has no anchor date, resolve '
    'against Today.\n'
    'Most follow-ups are one-off — use "once" with count 0 unless the record '
    'clearly asks for a repeating schedule. Keep title short and free of '
    'dosage detail (e.g. "Repeat CMV test", "HRCT chest scan"). If there is no '
    'follow-up action at all, return {"followups": []}. Output only the JSON.'
)


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def extract_followups(text: str) -> list[dict]:
    """Return [{title, date, frequency, count}, ...]; [] on none/failure."""
    today = date.today().isoformat()
    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {'role': 'system', 'content': FOLLOWUP_EXTRACT_PROMPT},
                {'role': 'user', 'content': f'Today is {today}.\n\nRecord:\n{text}'},
            ],
            think=False,
            format='json',
            keep_alive=MODEL_KEEP_ALIVE,
            options={'num_ctx': CHAT_NUM_CTX, 'temperature': 0.0,
                     'num_predict': 256},
        )
        items = json.loads(response['message']['content']).get('followups', [])
    except Exception:  # noqa: BLE001 - never let extraction break ingest
        return []

    out: list[dict] = []
    for item in items:
        title = str(item.get('title') or '').strip()
        if not title:
            continue
        out.append({
            'title': title,
            'date': (str(item.get('date') or '').strip() or None),
            'frequency': str(item.get('frequency') or 'once').strip().lower(),
            'count': _int(item.get('count')),
        })
    return out


def create_pending_followups(text: str) -> list[dict]:
    """Extract follow-ups and persist each as a pending reminder. Returns the
    created reminders. MUST be called with MODEL_LOCK held."""
    created: list[dict] = []
    for f in extract_followups(text):
        start = f'{f["date"]}T09:00:00' if f['date'] else None
        rrule = build_rrule(f['frequency'], f['count'])
        created.append(reminders.add(
            kind='followup',
            title=f['title'],
            proposed_text=f['title'],
            start=start,
            rrule=rrule,
        ))
    return created
