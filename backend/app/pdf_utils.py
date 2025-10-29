# pdf_utils.py
from io import BytesIO
from typing import Optional
from pdfminer.high_level import extract_text

# Make PyMuPDF optional. If not installed, we still work (pdfminer first).
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    _HAS_PYMUPDF = False


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """
    Robust PDF text extractor:
    1) Try pdfminer.six (fast for text PDFs)
    2) Fallback to PyMuPDF if available (handles many tricky PDFs)
    Returns "" if neither succeeds.
    """
    # 1) pdfminer first
    try:
        t = extract_text(BytesIO(data)) or ""
        # accept if it looks like real text
        if len(t.strip()) > 100:
            return t
    except Exception:
        pass

    # 2) fallback to PyMuPDF if present
    if _HAS_PYMUPDF:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            pages = []
            for page in doc:
                pages.append(page.get_text("text"))
            txt = "\n".join(pages)
            if len(txt.strip()) > 0:
                return txt
        except Exception:
            pass

    # if all failed, return empty string
    return ""
