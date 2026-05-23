"""Document text extraction (PDF text-layer / image OCR) and LLM analysis.

Refactored from the original analyze_reports.py into an importable library:
no argparse, no stdout dumping of document content.
"""
import re
from pathlib import Path

import ollama
import pymupdf

from core.config import (
    ANALYSIS_NUM_CTX,
    CHAT_MODEL,
    MODEL_KEEP_ALIVE,
    MODEL_THINKING,
    OCR_MODEL,
)

PDF_EXTENSIONS = {'.pdf'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS

MIN_TEXT_LAYER_CHARS = 20
OCR_RENDER_DPI = 200

# --- Scanned-PDF detection -------------------------------------------------
# A page is a single full-page image (a scan) if it has exactly one image
# covering at least this fraction of the page.
FULL_PAGE_IMAGE_COVERAGE = 0.90
# A PDF is treated as scanned if its vector-text coverage falls below this.
LOW_TEXT_RATIO_THRESHOLD = 0.05
# Producers known to flatten / overlay OCR on scanned pages.
SCANNER_PRODUCERS = (
    'camscanner',
    'tinyscanner',
    'adobe scan',
    'office lens',
    'scanbot',
    'genius scan',
    'scanner pro',
)

REPORT_PROMPT = (
    'You are a medical assistant. Below is the combined text of one or more '
    'medical reports. Each report is wrapped in markers like '
    '`[[SOURCE: <filename>]] ... [[/SOURCE: <filename>]]`. You MUST cite the '
    'source filename for EVERY fact, value, finding, and recommendation you '
    'state, using the format `(source: <filename>)` inline at the end of the '
    'sentence or bullet. Never use the source marker syntax itself in your '
    'output — only the `(source: ...)` citation.\n\n'
    'First output exactly one line, before anything else:\n'
    '`DOCUMENT_DATE: YYYY-MM-DD` — the date the sample was collected for '
    'testing; if that is not stated, the date the report was issued. Convert '
    'whatever date format the report uses into YYYY-MM-DD. Write '
    '`DOCUMENT_DATE: NONE` only if no date appears anywhere. Then a blank '
    'line, then the sections below.\n\n'
    'Produce the following sections:\n\n'
    '## 1. Patient overview\n'
    'Summarize demographics, history, and clinical context if present. Cite '
    'sources for each fact.\n\n'
    '## 2. Report-by-report findings\n'
    'For each source file, use a sub-heading `### <filename>` and list its key '
    'findings, abnormal values, and clinical impressions as bullets. Each '
    'bullet must still end with `(source: <filename>)`.\n\n'
    '## 3. Parameter values\n'
    'List EVERY measurable parameter found across all reports as a bullet list, '
    'one entry per parameter, in this exact format:\n'
    '  - <Parameter name>: <value with unit> (reference range if given) (source: <filename>)\n'
    'If the same parameter appears in multiple reports, list each occurrence as '
    'its own bullet so trends are visible. If a parameter is mentioned but its '
    'value is missing, unreadable, ambiguous, or inconclusive, write the value '
    'as exactly `NA` or `could not find` — DO NOT guess, infer, or leave it '
    'blank. Still cite the source.\n\n'
    '## 4. Combined assessment\n'
    'Synthesize findings across reports: trends, correlations, and any '
    'consistent or conflicting findings. Cite all sources contributing to each '
    'point.\n\n'
    '## 5. Potential next steps\n'
    'Suggest possible follow-up tests, specialist referrals, or treatment '
    'considerations. Flag any urgent concerns. Cite the sources that motivate '
    'each suggestion.\n\n'
    'Do not invent information not present in the reports. End with a one-line '
    'disclaimer that this is educational information, not medical advice.'
)

PRESCRIPTION_PROMPT = (
    'You are a medical assistant. Below is the combined text of one or more '
    'doctor prescriptions. Each prescription is wrapped in markers like '
    '`[[SOURCE: <filename>]] ... [[/SOURCE: <filename>]]`. You MUST cite the '
    'source filename for EVERY fact, medication, dose, and instruction you '
    'state, using the format `(source: <filename>)` inline at the end of the '
    'sentence or bullet. Never use the source marker syntax itself in your '
    'output — only the `(source: ...)` citation.\n\n'
    'First output exactly one line, before anything else:\n'
    '`DOCUMENT_DATE: YYYY-MM-DD` — the date the prescription was written. '
    'Convert whatever date format the prescription uses into YYYY-MM-DD. '
    'Write `DOCUMENT_DATE: NONE` only if no date appears anywhere. Then a '
    'blank line, then the sections below.\n\n'
    'Produce the following sections:\n\n'
    '## 1. Patient & prescriber overview\n'
    'Summarize patient demographics, the prescriber/clinic, prescription '
    'date, and any stated diagnosis or reason for the visit if present. Cite '
    'sources for each fact.\n\n'
    '## 2. Prescription-by-prescription details\n'
    'For each source file, use a sub-heading `### <filename>` and list its '
    'medications and instructions as bullets. Each bullet must still end with '
    '`(source: <filename>)`.\n\n'
    '## 3. Medication list\n'
    'List EVERY medication found across all prescriptions as a bullet list, '
    'one entry per medication, in this exact format:\n'
    '  - <Medication name>: <strength/dose>, <form>, <frequency>, <duration>, '
    '<quantity>, <route> — <purpose if stated> (source: <filename>)\n'
    'If the same medication appears in multiple prescriptions, list each '
    'occurrence as its own bullet so changes are visible. If any field is '
    'missing, unreadable, ambiguous, or inconclusive, write it as exactly '
    '`NA` or `could not find` — DO NOT guess or infer. Still cite the source.\n\n'
    '## 4. Combined assessment\n'
    'Synthesize across prescriptions: dose changes over time, duplicated or '
    'overlapping medications, and any potentially conflicting instructions. '
    'Cite all sources contributing to each point.\n\n'
    '## 5. Potential next steps\n'
    'Suggest things to clarify with the prescriber, refill timing, and any '
    'urgent concerns (e.g. possible interactions). Cite the sources that '
    'motivate each suggestion.\n\n'
    'Do not invent information not present in the prescriptions. End with a '
    'one-line disclaimer that this is educational information, not medical '
    'advice.'
)

PROMPTS = {
    'report': REPORT_PROMPT,
    'prescription': PRESCRIPTION_PROMPT,
}

IMAGE_OCR_PROMPT = (
    'This is a medical report image (lab result, prescription, scan report, '
    'or similar). Extract ALL readable text verbatim, preserving structure '
    '(headings, tables, key-value pairs, lists). Do not summarize, do not '
    'invent values. If something is illegible, write [illegible].'
)


def _is_scanned_page(page: pymupdf.Page) -> bool:
    """True only if the page has no usable text layer.

    Judged by extractable text, NOT by image coverage: a digital report can
    legitimately embed large images (ultrasound frames, charts, logos)
    alongside typed text — that is a digital report with pictures, not a scan.
    A genuinely scanned page is a single bitmap and yields almost no text.
    """
    return len(page.get_text().strip()) < MIN_TEXT_LAYER_CHARS


def _image_coverage(page: pymupdf.Page) -> float:
    """Fraction of the page area covered by image placements."""
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return 0.0
    total = 0.0
    for img in page.get_images(full=True):
        try:
            for rect in page.get_image_rects(img[0]):
                total += rect.width * rect.height
        except Exception:
            continue
    return total / page_area


def _is_full_page_image(page: pymupdf.Page) -> bool:
    """True if the page is a single bitmap covering ~the whole page (a scan)."""
    if len(page.get_images(full=True)) != 1:
        return False
    return _image_coverage(page) >= FULL_PAGE_IMAGE_COVERAGE


def _producers(doc: pymupdf.Document) -> list[str]:
    """Every /Producer and /Creator string, including XMP <pdf:Producer>."""
    producers: set[str] = set()
    meta = doc.metadata or {}
    for key in ('producer', 'creator'):
        if meta.get(key):
            producers.add(meta[key].strip())
    try:
        xmp = doc.get_xml_metadata()
        if xmp:
            for match in re.findall(r'<pdf:Producer>(.*?)</pdf:Producer>', xmp):
                producers.add(match.strip())
    except Exception:
        pass
    return sorted(p for p in producers if p)


def _touchup_textedit_pages(doc: pymupdf.Document) -> list[int]:
    """1-based page numbers carrying CamScanner-style TouchUp_TextEdit markers."""
    found: list[int] = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        for annot in page.annots() or []:
            try:
                if 'TouchUp_TextEdit' in str(annot.info):
                    found.append(page_num + 1)
                    break
            except Exception:
                continue
        try:
            raw_contents = page.get_contents()
        except Exception:
            raw_contents = None
        for xref in raw_contents or []:
            try:
                stream = doc.xref_stream(xref)
            except Exception:
                continue
            if stream and b'TouchUp_TextEdit' in stream:
                found.append(page_num + 1)
                break
    return sorted(set(found))


def _page_text_ratio(page: pymupdf.Page) -> float:
    """Fraction of the page area covered by text blocks (1.0 on failure)."""
    page_area = abs(page.rect.width * page.rect.height)
    if page_area <= 0:
        return 0.0
    try:
        text_area = sum(
            abs(pymupdf.Rect(block[:4]).get_area())
            for block in page.get_text_blocks()
        )
    except Exception:
        return 1.0
    return text_area / page_area


def is_scanned_pdf(doc: pymupdf.Document) -> tuple[bool, list[str]]:
    """Document-level scanned-PDF detector (multi-signal).

    Returns (is_scanned, reasons). The PDF is scanned if ANY signal fires:
    zero pages, no fonts anywhere, CamScanner OCR markers, a scanner-app
    producer, every page being a single full-page image, or near-zero text
    coverage. Image coverage alone is deliberately NOT used — digital reports
    legitimately embed large images (x-rays, ultrasound frames, charts).
    """
    n_pages = doc.page_count
    if n_pages == 0:
        return True, ['zero pages']

    reasons: list[str] = []

    if not any(page.get_fonts() for page in doc):
        reasons.append('no fonts on any page')

    cam_pages = _touchup_textedit_pages(doc)
    if cam_pages:
        reasons.append(f'CamScanner TouchUp_TextEdit markers on page(s) {cam_pages}')

    for producer in _producers(doc):
        if any(s in producer.lower() for s in SCANNER_PRODUCERS):
            reasons.append(f'scanner-app producer: {producer!r}')
            break

    if sum(_is_full_page_image(page) for page in doc) == n_pages:
        reasons.append(f'all {n_pages} page(s) are a single full-page image')

    text_ratio = sum(_page_text_ratio(page) for page in doc) / n_pages
    if text_ratio <= LOW_TEXT_RATIO_THRESHOLD:
        reasons.append(
            f'text coverage {text_ratio:.0%} <= {LOW_TEXT_RATIO_THRESHOLD:.0%}'
        )

    return bool(reasons), reasons


def _ocr_image_bytes(image_bytes: bytes) -> str:
    response = ollama.chat(
        model=OCR_MODEL,
        messages=[{
            'role': 'user',
            'content': IMAGE_OCR_PROMPT,
            'images': [image_bytes],
        }],
    )
    return response['message']['content']


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF.

    A document-level multi-signal check (`is_scanned_pdf`) decides the whole
    PDF; a per-page text-layer check still catches individual scanned pages in
    an otherwise-native PDF. A page is OCR'd if EITHER signal flags it scanned.
    """
    pages_out = []
    with pymupdf.open(str(pdf_path)) as doc:
        if doc.page_count == 0:
            return ''

        doc_scanned, reasons = is_scanned_pdf(doc)
        if doc_scanned:
            print(f'{pdf_path.name}: scanned document ({reasons[0]}) -> OCR every page')
        else:
            print(f'{pdf_path.name}: native PDF -> text layer (per-page OCR fallback)')

        for i, page in enumerate(doc, 1):
            if doc_scanned or _is_scanned_page(page):
                print(f'  Page {i}: OCR with {OCR_MODEL}')
                pix = page.get_pixmap(dpi=OCR_RENDER_DPI)
                text = _ocr_image_bytes(pix.tobytes('png')).strip()
                source = 'OCR'
            else:
                text = page.get_text().strip()
                source = 'text-layer'
            pages_out.append(f'--- Page {i} [{source}] ---\n{text}')
    return '\n\n'.join(pages_out)


def ocr_image(image_path: Path) -> str:
    return _ocr_image_bytes(image_path.read_bytes())


def extract_file(path: Path) -> str:
    """Extract text from one PDF or image file."""
    suffix = path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return extract_pdf_text(path)
    if suffix in IMAGE_EXTENSIONS:
        print(f'{path.name}: image (scanned) -> OCR with {OCR_MODEL}')
        return ocr_image(path)
    raise ValueError(f'Unsupported file type: {path}')


def analyze_extracted(
    filename: str,
    text: str,
    report_type: str = 'report',
    user_prompt: str | None = None,
) -> str:
    """Run the structured LLM analysis over already-extracted text for one file."""
    if report_type not in PROMPTS:
        raise ValueError(f'Unknown report_type: {report_type!r}')
    block = f'[[SOURCE: {filename}]]\n{text}\n[[/SOURCE: {filename}]]'
    instructions = PROMPTS[report_type]
    if user_prompt and user_prompt.strip():
        instructions += (
            '\n\n## Additional user request\n'
            'In addition to the structured analysis above, also address the '
            f'following from the user:\n{user_prompt.strip()}'
        )
    full_prompt = f'{instructions}\n\nMedical reports:\n{block}'
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{'role': 'user', 'content': full_prompt}],
        think=MODEL_THINKING,
        keep_alive=MODEL_KEEP_ALIVE,
        options={'num_ctx': ANALYSIS_NUM_CTX},
    )
    return response['message']['content']
