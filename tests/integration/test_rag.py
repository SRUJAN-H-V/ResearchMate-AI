
from unittest.mock import patch

import core.vector_store as vector_store_module

from utils.text_chunker import create_chunks
from core.vector_store import create_vector_store
from core.rag_chain import answer_question


# Sample research paper
sample_pages = [
    {
        "page": 1,
        "text": """
        The research paper investigates artificial intelligence
        usage in higher education.

        Students commonly use AI for summarization, idea generation
        and writing assistance.
        """
    },
    {
        "page": 2,
        "text": """
        Faculty members express concerns about academic integrity,
        plagiarism and the impact of AI on critical thinking.

        The paper recommends institutional policies and AI
        awareness programs.
        """
    }
]


def test_rag_question_answering(tmp_path, monkeypatch):

    # Step 1: Use a temporary ChromaDB directory.
    monkeypatch.setattr(
        vector_store_module,
        "PERSIST_DIRECTORY",
        str(tmp_path)
    )

    # Step 2: Convert research paper into chunks.
    chunks = create_chunks(sample_pages)

    assert len(chunks) > 0

    print("\nTotal chunks:", len(chunks))

    # Step 3: Create vector database.
    vector_store = create_vector_store(
        chunks,
        collection_name="pytest_rag_collection"
    )

    assert vector_store is not None

    print("\nChromaDB created successfully!")

    # Step 4: Ask a research question.
    question = (
        "What concerns do faculty members have "
        "about artificial intelligence?"
    )

    print("\nQuestion:", question)

    # Step 5: Mock the LLM response.
    # This isolates the RAG retrieval and prompt pipeline
    # from Gemini API calls during this test.
    with patch(
        "core.rag_chain.llm.invoke"
    ) as mock_llm:

        mock_llm.return_value = (
            "Faculty members are concerned about academic "
            "integrity, plagiarism and critical thinking. "
            "These concerns are discussed on page 2."
        )

        # Step 6: Execute RAG question answering.
        answer = answer_question(
            vector_store,
            question
        )

        # Verify that the LLM was called.
        mock_llm.assert_called_once()

    # Step 7: Verify the generated answer.
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0

    assert "faculty members" in answer.lower()

    print("\nResearchMate AI Answer:\n")
    print(answer)


def test_rag_empty_question():

    with patch("core.rag_chain.search_documents") as mock_search:

        from core.rag_chain import answer_question

        import pytest

        with pytest.raises(
            ValueError,
            match="Please enter a question."
        ):
            answer_question(object(), "",)


def test_rag_missing_vector_store():

    import pytest

    with pytest.raises(
        ValueError,
        match="No research paper is connected to RAG."
    ):
        answer_question(
            None,
            "What is the research about?"
        )


def test_rag_no_relevant_documents():

    with patch(
        "core.rag_chain.search_documents",
        return_value=[]
    ):

        answer = answer_question(
            object(),
            "What is the research about?"
        )

        assert "could not find relevant information" in answer.lower()
