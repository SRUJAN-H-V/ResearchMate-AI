
import pytest
from utils.text_chunker import create_chunks


def test_create_chunks_returns_documents():
    pages = [
        {"page": 1, "text": "Artificial intelligence is transforming education."}
    ]

    chunks = create_chunks(pages)

    assert len(chunks) > 0
    assert chunks[0].page_content
    assert chunks[0].metadata["page"] == 1


def test_create_chunks_multiple_pages():
    pages = [
        {"page": 1, "text": "This is content from page one."},
        {"page": 2, "text": "This is content from page two."}
    ]

    chunks = create_chunks(pages)

    page_numbers = {chunk.metadata["page"] for chunk in chunks}

    assert 1 in page_numbers
    assert 2 in page_numbers


def test_chunk_size_within_limit():
    pages = [
        {"page": 1, "text": "Artificial intelligence " * 200}
    ]

    chunks = create_chunks(pages)

    for chunk in chunks:
        assert len(chunk.page_content) <= 1000


def test_empty_pages_returns_empty_list():
    chunks = create_chunks([])

    assert chunks == []


def test_page_metadata_is_preserved():
    pages = [
        {"page": 5, "text": "Research paper content for testing."}
    ]

    chunks = create_chunks(pages)

    assert chunks[0].metadata["page"] == 5
