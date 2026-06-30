"""Flow tests for the agent graph (core/agent/graph.py).

The boundaries the nodes call (intent classification, query expansion,
retrieval, vector-store count, term reformulation) are mocked, so the graph's
routing and retry logic is exercised without a model or a vector store.
build_messages / sources_from_hits run for real (no model calls).
"""
from core.agent import graph as graph_mod
from core.agent import nodes


def _hit(text, fname):
    return (text, {'filename': fname, 'doc_type': 'report'}, 0.1)


def _run(question, history=None):
    state = None
    for kind, payload in graph_mod.run_turn(question, history or []):
        if kind == 'result':
            state = payload
    return state


def test_strong_retrieval_is_single_hop(monkeypatch):
    monkeypatch.setattr(nodes, 'classify_intent', lambda q, h=None: 'records')
    monkeypatch.setattr(nodes, 'expand_query',
                        lambda q: ('vt', ['crp'], False, None))
    monkeypatch.setattr(nodes.vector_store, 'count', lambda: 5)
    calls = {'n': 0}

    def fake_retrieve(*a, **k):
        calls['n'] += 1
        return [_hit('crp 5', 'r1.pdf'), _hit('crp 6', 'r2.pdf')]

    monkeypatch.setattr(nodes, 'retrieve', fake_retrieve)

    state = _run('what was my crp')
    assert calls['n'] == 1                     # no retry
    assert state['retrieval_attempts'] == 1
    assert len(state['sources']) == 2
    assert state['messages'][0]['role'] == 'system'
    assert state['user_msg']


def test_weak_retrieval_triggers_one_retry(monkeypatch):
    monkeypatch.setattr(nodes, 'classify_intent', lambda q, h=None: 'records')
    monkeypatch.setattr(nodes, 'expand_query',
                        lambda q: ('vt', ['crp'], False, None))
    monkeypatch.setattr(nodes.vector_store, 'count', lambda: 5)
    monkeypatch.setattr(nodes, '_reformulate_terms',
                        lambda q, p: ['c-reactive protein'])
    calls = {'n': 0}

    def fake_retrieve(vector_text, keywords, *a, **k):
        calls['n'] += 1
        if calls['n'] == 1:
            return []                          # weak first hop
        return [_hit('crp 12', 'r1.pdf'), _hit('crp 9', 'r2.pdf')]

    monkeypatch.setattr(nodes, 'retrieve', fake_retrieve)

    state = _run('obscure phrasing')
    assert calls['n'] == 2                     # exactly one retry
    assert state['retrieval_attempts'] == 2
    assert len(state['sources']) == 2


def test_persistently_weak_retrieval_is_capped(monkeypatch):
    monkeypatch.setattr(nodes, 'classify_intent', lambda q, h=None: 'records')
    monkeypatch.setattr(nodes, 'expand_query',
                        lambda q: ('vt', ['x'], False, None))
    monkeypatch.setattr(nodes.vector_store, 'count', lambda: 5)
    monkeypatch.setattr(nodes, '_reformulate_terms', lambda q, p: ['y'])
    calls = {'n': 0}

    def fake_retrieve(*a, **k):
        calls['n'] += 1
        return []                              # always weak

    monkeypatch.setattr(nodes, 'retrieve', fake_retrieve)

    state = _run('nothing matches')
    assert calls['n'] == 2                     # capped at MAX_RETRIEVAL_ATTEMPTS
    assert state['retrieval_attempts'] == 2
    assert state['sources'] == []


def test_empty_store_does_not_retry(monkeypatch):
    monkeypatch.setattr(nodes, 'classify_intent', lambda q, h=None: 'records')
    monkeypatch.setattr(nodes, 'expand_query',
                        lambda q: ('vt', ['x'], False, None))
    monkeypatch.setattr(nodes.vector_store, 'count', lambda: 0)  # nothing ingested
    calls = {'n': 0}

    def fake_retrieve(*a, **k):
        calls['n'] += 1
        return []

    monkeypatch.setattr(nodes, 'retrieve', fake_retrieve)

    state = _run('anything')
    assert calls['n'] == 1                     # no point retrying an empty store
    assert state['retrieval_attempts'] == 1


def test_general_intent_has_no_retrieval_or_sources(monkeypatch):
    monkeypatch.setattr(nodes, 'classify_intent', lambda q, h=None: 'general')

    def fail_retrieve(*a, **k):
        raise AssertionError('general branch must not retrieve')

    monkeypatch.setattr(nodes, 'retrieve', fail_retrieve)

    state = _run('what does high crp mean')
    assert state['intent'] == 'general'
    assert state['sources'] == []
    assert state['messages'][-1]['content'] == 'what does high crp mean'


def test_mixed_intent_retrieves_once_and_keeps_sources(monkeypatch):
    monkeypatch.setattr(nodes, 'classify_intent', lambda q, h=None: 'mixed')
    monkeypatch.setattr(nodes, 'expand_query',
                        lambda q: ('vt', ['crp'], False, None))
    calls = {'n': 0}

    def fake_retrieve(*a, **k):
        calls['n'] += 1
        return [_hit('crp 12', 'r1.pdf')]

    monkeypatch.setattr(nodes, 'retrieve', fake_retrieve)

    state = _run('is my crp of 12 high')
    assert state['intent'] == 'mixed'
    assert calls['n'] == 1                     # single hop, no retry cycle
    assert len(state['sources']) == 1
    assert 'Record excerpts:' in state['messages'][-1]['content']
