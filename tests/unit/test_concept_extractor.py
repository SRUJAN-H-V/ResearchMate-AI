
from unittest.mock import patch

import pytest

from agents.concept_extractor import extract_concepts


# Test 1: Valid research paper content
def test_extract_concepts_success():

    expected_response = """
    Key Concepts:
    - Artificial Intelligence
    - Machine Learning

    Algorithms:
    - Random Forest
    """

    with patch(
        "agents.concept_extractor.concept_chain"
    ) as mock_chain:

        mock_chain.invoke.return_value = expected_response

        result = extract_concepts(
            "This research paper discusses Artificial Intelligence."
        )

        assert result == expected_response
        mock_chain.invoke.assert_called_once()


# Test 2: Empty input
def test_extract_concepts_empty_input():

    with pytest.raises(
        ValueError,
        match="No research paper content provided."
    ):
        extract_concepts("")


# Test 3: Whitespace input
def test_extract_concepts_whitespace_input():

    with pytest.raises(ValueError):
        extract_concepts("   ")


# Test 4: None input
def test_extract_concepts_none_input():

    with pytest.raises(ValueError):
        extract_concepts(None)


# Test 5: Invalid input type
def test_extract_concepts_invalid_input():

    with pytest.raises(ValueError):
        extract_concepts(123)


# Test 6: Response should be a string
def test_extract_concepts_returns_text():

    expected_response = "Key Concepts: AI and Machine Learning"

    with patch(
        "agents.concept_extractor.concept_chain"
    ) as mock_chain:

        mock_chain.invoke.return_value = expected_response

        result = extract_concepts("Research paper content")

        assert isinstance(result, str)
        assert "Key Concepts" in result
