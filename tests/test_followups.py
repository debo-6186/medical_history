"""Unit tests for follow-up extraction (core/followups.py). ollama is mocked."""
import json

import core.followups as followups
import core.reminders as reminders


def _resp(content: str) -> dict:
    return {'message': {'content': content}}


def _mock(monkeypatch, payload):
    monkeypatch.setattr(
        followups.ollama, 'chat', lambda **kw: _resp(json.dumps(payload)))


def test_extract_parses_items(monkeypatch):
    _mock(monkeypatch, {'followups': [
        {'title': 'Repeat CMV test', 'date': '2026-09-26',
         'frequency': 'once', 'count': 0}]})
    out = followups.extract_followups(
        'saw doctor 2026-05-26, repeat CMV after 4 months')
    assert out == [{'title': 'Repeat CMV test', 'date': '2026-09-26',
                    'frequency': 'once', 'count': 0}]


def test_extract_empty_when_none(monkeypatch):
    _mock(monkeypatch, {'followups': []})
    assert followups.extract_followups('just a note, nothing to do') == []


def test_extract_empty_on_model_failure(monkeypatch):
    def boom(**kw):
        raise RuntimeError('ollama down')

    monkeypatch.setattr(followups.ollama, 'chat', boom)
    assert followups.extract_followups('x') == []


def test_extract_skips_titleless_items(monkeypatch):
    _mock(monkeypatch, {'followups': [
        {'title': '', 'date': '2026-09-26'},
        {'title': 'HRCT chest', 'date': None}]})
    out = followups.extract_followups('x')
    assert len(out) == 1
    assert out[0]['title'] == 'HRCT chest'
    assert out[0]['date'] is None


def test_create_pending_persists_followup(monkeypatch, tmp_path):
    monkeypatch.setattr(reminders, 'REMINDERS_DB_PATH', str(tmp_path / 'r.db'))
    _mock(monkeypatch, {'followups': [
        {'title': 'Repeat CMV test', 'date': '2026-09-26',
         'frequency': 'once', 'count': 0}]})
    created = followups.create_pending_followups('...record...')
    assert len(created) == 1
    r = created[0]
    assert r['kind'] == 'followup'
    assert r['start'] == '2026-09-26T09:00:00'
    assert r['rrule'] is None                 # one-off
    assert r['status'] == 'pending'
    assert reminders.list_reminders('pending')[0]['id'] == r['id']
