"""Unit tests for the local pending-reminders store (core/reminders.py).

Each test points REMINDERS_DB_PATH at a fresh temp sqlite file.
"""
import pytest

import core.reminders as reminders


@pytest.fixture(autouse=True)
def _temp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(reminders, 'REMINDERS_DB_PATH', str(tmp_path / 'r.db'))


def test_add_defaults_to_pending():
    r = reminders.add('manual', 'Take walk', 'Take a walk',
                      start='2026-07-01T09:00:00', rrule='RRULE:FREQ=DAILY')
    assert r['id'] > 0
    assert r['status'] == 'pending'
    assert r['gcal_event_id'] is None
    assert reminders.get(r['id'])['title'] == 'Take walk'


def test_list_and_status_filter():
    a = reminders.add('manual', 'a', 'a')
    b = reminders.add('manual', 'b', 'b')
    reminders.mark_dismissed(b['id'])
    pending = reminders.list_reminders('pending')
    assert [r['id'] for r in pending] == [a['id']]
    assert len(reminders.list_reminders()) == 2          # newest-first, all


def test_update_fields():
    r = reminders.add('manual', 'x', 'x')
    u = reminders.update_fields(r['id'], proposed_text='edited',
                                start='2026-07-02T08:00:00')
    assert u['proposed_text'] == 'edited'
    assert u['start'] == '2026-07-02T08:00:00'


def test_update_ignores_unknown_columns():
    r = reminders.add('manual', 'x', 'x')
    u = reminders.update_fields(r['id'], status='scheduled', id=999)
    assert u['status'] == 'pending'                      # not editable here
    assert u['id'] == r['id']


def test_mark_scheduled():
    r = reminders.add('medication', 'Metformin', 'Take Metformin')
    u = reminders.mark_scheduled(r['id'], 'evt123')
    assert u['status'] == 'scheduled'
    assert u['gcal_event_id'] == 'evt123'


def test_invalid_kind_rejected():
    with pytest.raises(ValueError):
        reminders.add('bogus', 'x', 'x')
