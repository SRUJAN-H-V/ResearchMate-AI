
import pymupdf


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using its file path."""

    with pymupdf.open(pdf_path) as document:
        return extract_pages(document)


def extract_text_from_pdf_bytes(pdf_bytes):
    """Extract text from a PDF uploaded through Streamlit."""

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as document:
        return extract_pages(document)


def extract_pages(document):
    """Extract text and page numbers from a PDF document."""

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text()

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    if not pages:
        raise ValueError(
            "No readable text found in the PDF."
        )

    return pages
