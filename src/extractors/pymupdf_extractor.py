from pathlib import Path

try:
    import fitz  # type: ignore  # PyMuPDF
except ImportError:  # pragma: no cover - optional dependency
    fitz = None


def extract_text_pymupdf(pdf_path: Path) -> str:
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is not installed")

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

