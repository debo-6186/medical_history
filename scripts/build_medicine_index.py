"""Build the offline medicine-name index (India + Singapore + US + generics).

Run ONCE, with internet:
    uv run python scripts/build_medicine_index.py

Writes an FTS5 sqlite DB to MEDICINES_DB_PATH (default ./medicines.db) which the
app then reads offline (core/medicines.py). Re-run to refresh from upstream.

Sources — all free, no login required:
  IN  raw.githubusercontent.com/junioralive/Indian-Medicine-Dataset (brand + composition)
  SG  data.gov.sg d_767279312753558cbf19d48344577084 (HSA register: product + active ingredient)
  US  accessdata.fda.gov/cder/ndctext.zip (FDA NDC: proprietary + non-proprietary name)
Each source degrades independently — if one is unreachable the others still build.
"""
import csv
import io
import json
import re
import sqlite3
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import MEDICINES_DB_PATH  # noqa: E402

IN_URL = ('https://raw.githubusercontent.com/junioralive/'
          'Indian-Medicine-Dataset/main/DATA/indian_medicine_data.csv')
SG_POLL = ('https://api-open.data.gov.sg/v1/public/api/datasets/'
           'd_767279312753558cbf19d48344577084/poll-download')
US_URL = 'https://www.accessdata.fda.gov/cder/ndctext.zip'

_STRENGTH_RE = re.compile(r'\([^)]*\)')     # drop "(500mg)" strengths
_WS_RE = re.compile(r'\s+')


def _get(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'medical-history/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _clean(s: str | None) -> str:
    return _WS_RE.sub(' ', (s or '').strip())


def fetch_india() -> list[tuple]:
    print('India: downloading…')
    text = _get(IN_URL).decode('utf-8', 'replace')
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        name = _clean(row.get('name'))
        if not name:
            continue
        comp = ' '.join(p for p in (row.get('short_composition1'),
                                    row.get('short_composition2')) if p)
        rows.append((name, _clean(_STRENGTH_RE.sub('', comp)), 'IN', 'brand'))
    print(f'India: {len(rows)} rows')
    return rows


def fetch_singapore() -> list[tuple]:
    print('Singapore: resolving download URL…')
    url = json.loads(_get(SG_POLL))['data']['url']
    print('Singapore: downloading…')
    text = _get(url).decode('utf-8', 'replace')
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        name = _clean(row.get('product_name'))
        if not name:
            continue
        generic = _clean((row.get('active_ingredients') or '').replace('&&', ', '))
        rows.append((name, generic, 'SG', 'brand'))
    print(f'Singapore: {len(rows)} rows')
    return rows


def fetch_us() -> list[tuple]:
    print('US (FDA NDC): downloading…')
    blob = _get(US_URL)
    rows = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        product = next(n for n in zf.namelist() if n.lower().startswith('product'))
        with zf.open(product) as raw:
            reader = csv.DictReader(
                io.TextIOWrapper(raw, encoding='latin-1'), delimiter='\t')
            for row in reader:
                generic = _clean(row.get('NONPROPRIETARYNAME'))
                name = _clean(row.get('PROPRIETARYNAME')) or generic
                if not name:
                    continue
                rows.append((name, generic, 'US', 'brand'))
    print(f'US: {len(rows)} rows')
    return rows


def generic_backbone(rows: list[tuple]) -> list[tuple]:
    """A universal generic list from every source's active ingredients, so an
    ingredient search works regardless of the selected country."""
    seen: set[str] = set()
    out = []
    for _name, generic, _region, _mtype in rows:
        for part in re.split(r'[;,]', generic or ''):
            g = _clean(part)
            if len(g) >= 3 and g.lower() not in seen:
                seen.add(g.lower())
                out.append((g, g, 'generic', 'generic'))
    print(f'Generics: {len(out)} rows')
    return out


def dedupe(rows: list[tuple]) -> list[tuple]:
    seen: set[tuple] = set()
    out = []
    for r in rows:
        key = (r[0].lower(), r[2])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def build(rows: list[tuple]) -> None:
    path = Path(MEDICINES_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    conn.execute('CREATE VIRTUAL TABLE medicines USING '
                 'fts5(name, generic, region UNINDEXED, mtype UNINDEXED)')
    conn.executemany(
        'INSERT INTO medicines (name, generic, region, mtype) VALUES (?,?,?,?)',
        rows)
    conn.execute("INSERT INTO medicines(medicines) VALUES ('optimize')")
    conn.commit()
    conn.close()
    print(f'Wrote {len(rows)} rows -> {path}')


def main() -> None:
    rows: list[tuple] = []
    for fetch in (fetch_india, fetch_singapore, fetch_us):
        try:
            rows += fetch()
        except Exception as exc:  # noqa: BLE001 - one source failing is non-fatal
            print(f'WARN: {fetch.__name__} failed: {exc}')
    if not rows:
        print('No data fetched — aborting.')
        sys.exit(1)
    rows = dedupe(rows)
    rows += generic_backbone(rows)
    rows = dedupe(rows)
    build(rows)


if __name__ == '__main__':
    main()
