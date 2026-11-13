
from agents.viva_agent import viva_prompt


sample_content = """
The research paper investigates the perspectives of
faculty members and students regarding artificial
intelligence in higher education.

Students commonly use AI for summarization,
idea generation and writing assistance.

Faculty members express concerns about academic
integrity, plagiarism and critical thinking.

The paper highlights the need for institutional
policies and AI awareness programs.
"""


# Test 1: Check whether the prompt is created.
def test_viva_prompt_creation():

    messages = viva_prompt.format_messages(
        paper_content=sample_content
    )

    assert len(messages) == 2


# Test 2: Check whether system and human messages exist.
def test_viva_prompt_roles():

    messages = viva_prompt.format_messages(
        paper_content=sample_content
    )

    assert messages[0].type == "system"
    assert messages[1].type == "human"


# Test 3: Check whether paper content is inserted.

def test_viva_prompt_contains_paper_content():

    messages = viva_prompt.format_messages(
        paper_content=sample_content
    )

    # Replace line breaks with spaces before checking.
    content = " ".join(messages[1].content.lower().split())

    assert "artificial intelligence" in content



# Test 4: Check whether the prompt is not empty.
def test_viva_prompt_not_empty():

    messages = viva_prompt.format_messages(
        paper_content=sample_content
    )

    for message in messages:
        assert message.content.strip() != ""


# Test 5: Check whether the prompt handles different content.
def test_viva_prompt_with_different_content():

    content = "This paper studies machine learning."

    messages = viva_prompt.format_messages(
        paper_content=content
    )

    assert content in messages[1].content
