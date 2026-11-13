
import pymupdf
import pytest

from utils.pdf_processor import (
    extract_text_from_pdf,
    extract_text_from_pdf_bytes
)


def create_test_pdf():
    """Create a temporary PDF in memory for testing."""

    document = pymupdf.open()

    page = document.new_page()
    page.insert_text(
        (50, 50),
        "Artificial intelligence is transforming education."
    )

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def test_extract_text_from_pdf(tmp_path):
    """Test extraction from a PDF file."""

    pdf_bytes = create_test_pdf()
    pdf_path = tmp_path / "sample.pdf"

    pdf_path.write_bytes(pdf_bytes)

    pages = extract_text_from_pdf(str(pdf_path))

    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert "Artificial intelligence" in pages[0]["text"]


def test_extract_text_from_pdf_bytes():
    """Test extraction from uploaded PDF bytes."""

    pdf_bytes = create_test_pdf()

    pages = extract_text_from_pdf_bytes(pdf_bytes)

    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert "Artificial intelligence" in pages[0]["text"]


def test_page_numbers_are_preserved():
    """Test that page numbering starts from 1."""

    document = pymupdf.open()

    for i in range(3):
        page = document.new_page()
        page.insert_text((50, 50), f"Content of page {i + 1}")

    pdf_bytes = document.tobytes()
    document.close()

    pages = extract_text_from_pdf_bytes(pdf_bytes)

    assert len(pages) == 3
    assert [page["page"] for page in pages] == [1, 2, 3]


def test_empty_pdf_raises_value_error():
    """Test that a PDF without readable text raises an error."""

    document = pymupdf.open()
    document.new_page()

    pdf_bytes = document.tobytes()
    document.close()

    with pytest.raises(ValueError, match="No readable text found"):
        extract_text_from_pdf_bytes(pdf_bytes)


def test_invalid_pdf_raises_error():
    """Test that invalid PDF bytes are rejected."""

    with pytest.raises(Exception):
        extract_text_from_pdf_bytes(b"This is not a PDF")
