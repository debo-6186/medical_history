"""Unit tests for core/calendar.py helpers that need no network/auth:
RRULE construction, the medication-title privacy truncation, and the
unauthorized guard.
"""
from datetime import datetime

import pytest

import core.calendar as gcal


def test_build_rrule_one_off():
    assert gcal.build_rrule('once') is None
    assert gcal.build_rrule(None) is None


def test_build_rrule_daily_with_count():
    assert gcal.build_rrule('daily', 7) == 'RRULE:FREQ=DAILY;COUNT=7'


def test_build_rrule_weekly_open_ended():
    assert gcal.build_rrule('weekly', 0) == 'RRULE:FREQ=WEEKLY'


def test_build_rrule_unknown_frequency():
    assert gcal.build_rrule('fortnightly') is None


def test_medication_title_is_truncated():
    # MED_TITLE_PREFIX_LEN defaults to 4 — only the first 4 chars leave the box.
    assert gcal._display_title('Metformin', 'medication') == 'Metf'


def test_non_medication_title_is_full():
    assert gcal._display_title('Take a walk', 'manual') == 'Take a walk'
    assert gcal._display_title('HRCT chest follow-up', 'followup') == \
        'HRCT chest follow-up'


def test_is_authorized_false_without_token(monkeypatch):
    monkeypatch.setattr(gcal, '_load_credentials', lambda: None)
    assert gcal.is_authorized() is False


def test_create_reminder_raises_when_unauthorized(monkeypatch):
    monkeypatch.setattr(gcal, '_load_credentials', lambda: None)
    with pytest.raises(gcal.NotAuthorizedError):
        gcal.create_reminder('Take walk', datetime(2026, 7, 1, 9, 0, 0))
