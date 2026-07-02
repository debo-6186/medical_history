"""Unit tests for the medication store (core/med_store.py)."""
import pytest

import core.med_store as med_store


@pytest.fixture(autouse=True)
def _db(monkeypatch, tmp_path):
    monkeypatch.setattr(med_store, 'MED_STORE_DB_PATH', str(tmp_path / 'm.db'))


def test_add_and_list():
    r = med_store.add('Metformin', generic='Metformin', region='IN',
                      dosage='500 mg', frequency='twice daily', tenure='30 days',
                      start_date='2026-07-01', source_doc_id='rx.pdf',
                      reminder_id=5)
    assert r['id'] > 0
    assert r['reminder_id'] == 5
    assert r['source_doc_id'] == 'rx.pdf'
    listed = med_store.list_medications()
    assert len(listed) == 1
    assert listed[0]['name'] == 'Metformin'


def test_timing_persisted():
    r = med_store.add('Metformin', timing='After breakfast')
    assert r['timing'] == 'After breakfast'
    assert med_store.list_medications()[0]['timing'] == 'After breakfast'


def test_tenure_days():
    assert med_store.tenure_to_count('7 days', 'once daily') == 7


def test_tenure_weeks_and_months():
    assert med_store.tenure_to_count('2 weeks', 'once daily') == 14
    assert med_store.tenure_to_count('1 month', 'twice daily') == 30


def test_tenure_weekly_frequency_counts_in_weeks():
    assert med_store.tenure_to_count('30 days', 'weekly') == 4      # round(30/7)


def test_tenure_open_ended_is_zero():
    assert med_store.tenure_to_count('ongoing', 'once daily') == 0
    assert med_store.tenure_to_count('', 'once daily') == 0
