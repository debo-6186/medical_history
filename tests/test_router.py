"""Unit tests for the intent router (core/agent/router.py).

ollama.chat is mocked to return canned JSON so no model is needed.
"""
import json

from core.agent import router


def _resp(content: str) -> dict:
    return {'message': {'content': content}}


def _mock_chat(monkeypatch, content):
    monkeypatch.setattr(router.ollama, 'chat', lambda **kw: _resp(content))


def test_records_intent(monkeypatch):
    _mock_chat(monkeypatch, json.dumps({'intent': 'records'}))
    assert router.classify_intent('what was my CRP in May') == 'records'


def test_general_intent(monkeypatch):
    _mock_chat(monkeypatch, json.dumps({'intent': 'general'}))
    assert router.classify_intent('what does a high CRP mean') == 'general'


def test_mixed_intent(monkeypatch):
    _mock_chat(monkeypatch, json.dumps({'intent': 'mixed'}))
    assert router.classify_intent('is my CRP of 12 high') == 'mixed'


def test_schedule_intent(monkeypatch):
    _mock_chat(monkeypatch, json.dumps({'intent': 'schedule'}))
    assert router.classify_intent('remind me to take my medicine') == 'schedule'


def test_invalid_intent_defaults_to_records(monkeypatch):
    _mock_chat(monkeypatch, json.dumps({'intent': 'banana'}))
    assert router.classify_intent('anything') == 'records'


def test_malformed_json_defaults_to_records(monkeypatch):
    _mock_chat(monkeypatch, 'not json at all')
    assert router.classify_intent('anything') == 'records'


def test_model_failure_defaults_to_records(monkeypatch):
    def boom(**kw):
        raise RuntimeError('ollama unreachable')

    monkeypatch.setattr(router.ollama, 'chat', boom)
    assert router.classify_intent('anything') == 'records'
