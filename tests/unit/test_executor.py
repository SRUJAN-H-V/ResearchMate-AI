
from unittest.mock import patch

from core.agent_executor import execute_agent


# Sample research paper content
paper_content = "This is sample research paper content."


# Test 1: Summary request
def test_execute_summary_agent():

    with patch(
        "core.agent_executor.summarize_paper",
        return_value="Summary generated successfully"
    ) as mock_agent:

        result = execute_agent(
            "Summarize this research paper",
            paper_content,
            None
        )

        assert result["intent"] == "summary"
        assert result["response"] == "Summary generated successfully"

        mock_agent.assert_called_once_with(paper_content)


# Test 2: Concept extraction request
def test_execute_concept_agent():

    with patch(
        "core.agent_executor.extract_concepts",
        return_value="Concepts extracted successfully"
    ) as mock_agent:

        result = execute_agent(
            "Explain the algorithms used",
            paper_content,
            None
        )

        assert result["intent"] == "concept"
        assert result["response"] == "Concepts extracted successfully"

        mock_agent.assert_called_once_with(paper_content)


# Test 3: Viva question request
def test_execute_viva_agent():

    with patch(
        "core.agent_executor.generate_viva_questions",
        return_value="Viva questions generated successfully"
    ) as mock_agent:

        result = execute_agent(
            "Prepare viva questions",
            paper_content,
            None
        )

        assert result["intent"] == "viva"
        assert result["response"] == "Viva questions generated successfully"

        mock_agent.assert_called_once_with(paper_content)


# Test 4: Chat / RAG request
def test_execute_rag_agent():

    vector_store = object()

    with patch(
        "core.agent_executor.answer_question",
        return_value="RAG answer generated successfully"
    ) as mock_agent:

        result = execute_agent(
            "What is the significance of this paper?",
            paper_content,
            vector_store
        )

        assert result["intent"] == "chat"
        assert result["response"] == "RAG answer generated successfully"

        mock_agent.assert_called_once_with(
            vector_store,
            "What is the significance of this paper?"
        )


# Test 5: Check returned dictionary structure
def test_execute_agent_result_structure():

    with patch(
        "core.agent_executor.summarize_paper",
        return_value="Test summary"
    ):

        result = execute_agent(
            "Give me a summary",
            paper_content,
            None
        )

        assert isinstance(result, dict)
        assert "intent" in result
        assert "response" in result
