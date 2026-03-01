"""Extract raw text from PDF or image lab reports using PyMuPDF and Tesseract."""

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Max chars of extracted text to log (avoid huge logs); rest is summarized
_MAX_LOG_TEXT_LEN = 4000


def _log_extracted_text(text: str, source: str) -> None:
    """Log the text extracted by Tesseract OCR (truncated if very long)."""
    if not text:
        logger.info("[%s] Extracted text: (empty)", source)
        return
    total = len(text)
    if total <= _MAX_LOG_TEXT_LEN:
        logger.info("[%s] Extracted text:\n%s", source, text)
    else:
        logger.info(
            "[%s] Extracted text (first %s of %s chars):\n%s\n... (truncated, total %s chars)",
            source,
            _MAX_LOG_TEXT_LEN,
            total,
            text[:_MAX_LOG_TEXT_LEN],
            total,
        )


try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None


def extract_text_from_pdf(bytes_data: bytes) -> str:
    """Extract text from PDF using PyMuPDF. Fallback to OCR if text is minimal."""
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is required for PDF extraction. Install with: pip install pymupdf")

    logger.info("PDF extraction started (PyMuPDF), size=%s bytes", len(bytes_data))
    doc = fitz.open(stream=bytes_data, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    full_text = "\n".join(text_parts).strip()
    logger.info("PDF text extracted: %s chars, %s pages", len(full_text), len(text_parts))

    # If very little text, try OCR via pdf2image + tesseract
    if len(full_text) < 100 and convert_from_bytes and pytesseract and Image:
        logger.info("PDF had minimal text (<100 chars), falling back to OCR (pdf2image + Tesseract)")
        images = convert_from_bytes(bytes_data, dpi=200)
        ocr_parts = []
        for idx, img in enumerate(images):
            page_text = pytesseract.image_to_string(img)
            ocr_parts.append(page_text)
            logger.debug("OCR page %s: %s chars", idx + 1, len(page_text))
        full_text = "\n".join(ocr_parts).strip()
        logger.info("OCR completed: total %s chars from %s page(s)", len(full_text), len(images))
        _log_extracted_text(full_text, source="PDF OCR (Tesseract)")

    return full_text


def extract_text_from_image(bytes_data: bytes) -> str:
    """Extract text from image using Tesseract OCR."""
    if pytesseract is None or Image is None:
        raise ImportError("pytesseract and Pillow are required for image OCR. Install with: pip install pytesseract pillow")

    logger.info("Image OCR started (Tesseract), size=%s bytes", len(bytes_data))
    img = Image.open(io.BytesIO(bytes_data))
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    text = pytesseract.image_to_string(img).strip()
    logger.info("Image OCR completed: %s chars extracted", len(text))
    _log_extracted_text(text, source="Image OCR (Tesseract)")
    return text


def extract_text_from_file(file) -> str:
    """
    Extract raw text from an uploaded file (PDF or image).
    file: file-like object (e.g. Streamlit UploadedFile) with .read() and .name
    Returns raw text string. No disk write; uses in-memory bytes.
    """
    raw = file.read()
    name = (getattr(file, "name", "") or "").lower()
    logger.info("File upload: name=%s, size=%s bytes", name or "(unnamed)", len(raw))

    if name.endswith(".pdf"):
        logger.info("Detected PDF; extracting with PyMuPDF")
        return extract_text_from_pdf(raw)

    if name.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
        logger.info("Detected image; extracting with Tesseract OCR")
        return extract_text_from_image(raw)

    # Heuristic: first bytes for PDF magic
    if raw[:4] == b"%PDF":
        logger.info("File has PDF magic bytes; extracting with PyMuPDF")
        return extract_text_from_pdf(raw)

    # Default: try as image
    logger.info("Treating file as image; extracting with Tesseract OCR")
    return extract_text_from_image(raw)
