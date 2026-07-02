"""Unit tests for medicine autocomplete (core/medicines.py) over a tiny FTS DB."""
import sqlite3

import pytest

import core.medicines as medicines


def _build(path, rows):
    conn = sqlite3.connect(path)
    conn.execute('CREATE VIRTUAL TABLE medicines USING '
                 'fts5(name, generic, region UNINDEXED, mtype UNINDEXED)')
    conn.executemany(
        'INSERT INTO medicines (name, generic, region, mtype) VALUES (?,?,?,?)',
        rows)
    conn.commit()
    conn.close()


@pytest.fixture
def idx(monkeypatch, tmp_path):
    path = tmp_path / 'medicines.db'
    _build(str(path), [
        ('Augmentin 625 Tablet', 'Amoxycillin Clavulanic Acid', 'IN', 'brand'),
        ('Metformin', 'Metformin', 'generic', 'generic'),
        ('PANADOL TABLET', 'Paracetamol', 'SG', 'brand'),
        ('TYLENOL', 'Acetaminophen', 'US', 'brand'),
    ])
    monkeypatch.setattr(medicines, 'MEDICINES_DB_PATH', str(path))
    return path


def test_available(idx):
    assert medicines.is_available() is True


def test_prefix_search(idx):
    hits = medicines.search('augm', region='IN')
    assert any(h['name'].startswith('Augmentin') for h in hits)


def test_region_filter_excludes_other_countries(idx):
    assert medicines.search('tyl', region='IN') == []      # Tylenol is US


def test_generic_backbone_always_included(idx):
    hits = medicines.search('metfor', region='IN')
    assert any(h['region'] == 'generic' for h in hits)


def test_blank_query_returns_empty(idx):
    assert medicines.search('   ', region='IN') == []


def test_missing_index_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(medicines, 'MEDICINES_DB_PATH', str(tmp_path / 'none.db'))
    assert medicines.is_available() is False
    assert medicines.search('augment') == []
