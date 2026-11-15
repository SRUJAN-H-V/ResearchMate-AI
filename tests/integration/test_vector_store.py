
import core.vector_store as vector_store_module

from utils.text_chunker import create_chunks
from core.vector_store import (
    create_vector_store,
    search_documents
)


# Sample research paper content
sample_pages = [
    {
        "page": 1,
        "text": """
        Artificial Intelligence is transforming higher education.

        Students use AI tools for summarization, idea generation,
        writing assistance and understanding complex topics.

        Faculty members have concerns about academic integrity,
        plagiarism and the impact of AI on critical thinking.

        Universities need institutional policies and AI awareness
        programs to encourage responsible AI usage.
        """
    },
    {
        "page": 2,
        "text": """
        The research investigates the perspectives of faculty
        members and students regarding artificial intelligence.

        The study highlights the importance of responsible AI
        adoption, educational policies and faculty training.
        """
    }
]


def test_vector_store_creation_and_search(tmp_path, monkeypatch):

    # Step 1: Use a temporary database directory.
    monkeypatch.setattr(
        vector_store_module,
        "PERSIST_DIRECTORY",
        str(tmp_path)
    )

    # Step 2: Create text chunks.
    chunks = create_chunks(sample_pages)

    assert len(chunks) > 0

    print("\nTotal chunks:", len(chunks))

    # Step 3: Store chunks in ChromaDB.
    vector_store = create_vector_store(
        chunks,
        collection_name="pytest_research_paper"
    )

    assert vector_store is not None

    print("\nChromaDB created successfully!")

    # Step 4: Search for relevant information.
    question = (
        "How do students and faculty members "
        "view artificial intelligence in education?"
    )

    results = search_documents(
        vector_store,
        question,
        k=4
    )

    # Step 5: Verify search results.
    assert len(results) > 0

    print("\nSearch results:")

    for i, result in enumerate(results, start=1):

        print(f"\n--- Result {i} ---")
        print("Source page:", result.metadata["page"])
        print(result.page_content[:300])

    # Step 6: Verify page metadata.
    for result in results:
        assert "page" in result.metadata
