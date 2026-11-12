
import pymupdf


def extract_text_from_pdf(pdf_path):
    """Extract text from every page of a PDF."""

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    document.close()

    if not pages:
        raise ValueError(
            "No readable text found in the PDF."
        )

    return pages
