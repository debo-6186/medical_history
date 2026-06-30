"""Google Calendar integration — the only outbound network path in the app.

Reminders are pushed here ONLY after the user confirms a draft (see
core/reminders.py and the /api/reminders/{id}/confirm route). Two privacy rules
hold in code, not just prompt:
  - a 'medication' reminder's title is truncated to MED_TITLE_PREFIX_LEN leading
    characters before it leaves the device;
  - nothing else (dosage, file names, record excerpts) is ever sent — callers
    pass only a title + time + recurrence.

Auth is the Google "Desktop app" installed-app flow: a one-time consent
(`authorize()`) opens a browser on the laptop and stores a refresh token under
rag_db/. Until that happens `is_authorized()` is False and `create_reminder()`
raises NotAuthorizedError — drafting/listing reminders still works offline.
"""
from datetime import datetime, timedelta

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from core.config import (
    GOOGLE_CALENDAR_ID,
    GOOGLE_CREDENTIALS_PATH,
    GOOGLE_TOKEN_PATH,
    MED_TITLE_PREFIX_LEN,
)
from pathlib import Path

# Narrowest scope that can create events — no read access to existing calendars.
SCOPES = ['https://www.googleapis.com/auth/calendar.events']

# Default visible duration of a reminder event (a point-in-time nudge).
_EVENT_MINUTES = 15

_FREQ_MAP = {
    'daily': 'DAILY',
    'twice daily': 'DAILY',
    'weekly': 'WEEKLY',
    'monthly': 'MONTHLY',
    'yearly': 'YEARLY',
}


class NotAuthorizedError(RuntimeError):
    """Raised when a Calendar call is attempted before OAuth consent."""


# --- Auth ------------------------------------------------------------------

def _load_credentials() -> Credentials | None:
    """Load stored credentials, refreshing if expired. None if not yet
    authorized (no token, or refresh failed)."""
    token_path = Path(GOOGLE_TOKEN_PATH)
    if not token_path.exists():
        return None
    try:
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    except Exception:  # noqa: BLE001 - corrupt/incompatible token
        return None
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
        except Exception:  # noqa: BLE001 - refresh failed; treat as unauthorized
            return None
    return creds if creds and creds.valid else None


def is_authorized() -> bool:
    return _load_credentials() is not None


def authorize() -> bool:
    """Run the one-time installed-app consent flow and store the token.

    Blocking and interactive — opens a browser on the laptop. Requires the
    OAuth client secrets at GOOGLE_CREDENTIALS_PATH. Call from a worker thread.
    """
    creds_path = Path(GOOGLE_CREDENTIALS_PATH)
    if not creds_path.exists():
        raise FileNotFoundError(
            f'Google OAuth client secrets not found at {creds_path}. Create a '
            '"Desktop app" OAuth client in Google Cloud Console (with the '
            'Calendar API enabled) and save the downloaded JSON there.'
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
    creds = flow.run_local_server(port=0)
    Path(GOOGLE_TOKEN_PATH).write_text(creds.to_json())
    return True


def _service():
    creds = _load_credentials()
    if creds is None:
        raise NotAuthorizedError(
            'Google Calendar is not connected. Authorize it first '
            '(POST /api/google/authorize).'
        )
    return build('calendar', 'v3', credentials=creds, cache_discovery=False)


# --- Recurrence ------------------------------------------------------------

def build_rrule(frequency: str | None, count: int | None = None) -> str | None:
    """Map a coarse frequency (+ optional occurrence count, e.g. a medication
    tenure in days) to an RFC-5545 RRULE string, or None for a one-off."""
    if not frequency:
        return None
    freq = _FREQ_MAP.get(frequency.strip().lower())
    if freq is None:
        return None
    rule = f'RRULE:FREQ={freq}'
    if count and count > 0:
        rule += f';COUNT={count}'
    return rule


# --- Event creation --------------------------------------------------------

def _display_title(title: str, kind: str | None) -> str:
    """Apply the privacy truncation: medication titles are reduced to their
    first MED_TITLE_PREFIX_LEN characters before being sent to Google."""
    if kind == 'medication':
        return title[:MED_TITLE_PREFIX_LEN]
    return title


def create_reminder(
    title: str,
    start: datetime,
    rrule: str | None = None,
    kind: str | None = None,
    timezone_name: str = 'UTC',
) -> str:
    """Create a Calendar event with a popup reminder. Returns the event id.

    `title` is the full local title; the value actually sent is run through the
    privacy truncation for medication reminders. Raises NotAuthorizedError if
    Google is not connected yet.
    """
    summary = _display_title(title, kind)
    end = start + timedelta(minutes=_EVENT_MINUTES)
    body = {
        'summary': summary,
        'start': {'dateTime': start.isoformat(), 'timeZone': timezone_name},
        'end': {'dateTime': end.isoformat(), 'timeZone': timezone_name},
        'reminders': {
            'useDefault': False,
            'overrides': [{'method': 'popup', 'minutes': 0}],
        },
    }
    if rrule:
        body['recurrence'] = [rrule]
    event = _service().events().insert(
        calendarId=GOOGLE_CALENDAR_ID, body=body).execute()
    return event['id']
