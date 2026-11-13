
from agents.summarizer import map_prompt, reduce_prompt


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


# Test 1: Check whether Map Prompt is created.
def test_map_prompt_creation():

    messages = map_prompt.format_messages(
        paper_chunk=sample_content
    )

    assert len(messages) == 2


# Test 2: Check Map Prompt roles.
def test_map_prompt_roles():

    messages = map_prompt.format_messages(
        paper_chunk=sample_content
    )

    assert messages[0].type == "system"
    assert messages[1].type == "human"


# Test 3: Check whether paper content is inserted.
def test_map_prompt_contains_content():

    messages = map_prompt.format_messages(
        paper_chunk=sample_content
    )

    content = " ".join(messages[1].content.lower().split())

    assert "artificial intelligence" in content


# Test 4: Check whether Reduce Prompt is created.
def test_reduce_prompt_creation():

    messages = reduce_prompt.format_messages(
        section_summaries="The paper examines AI usage."
    )

    assert len(messages) == 2


# Test 5: Check Reduce Prompt roles.
def test_reduce_prompt_roles():

    messages = reduce_prompt.format_messages(
        section_summaries="The paper examines AI usage."
    )

    assert messages[0].type == "system"
    assert messages[1].type == "human"


# Test 6: Check whether section summaries are inserted.
def test_reduce_prompt_contains_summary():

    summary = "The paper examines AI usage in higher education."

    messages = reduce_prompt.format_messages(
        section_summaries=summary
    )

    assert summary in messages[1].content


# Test 7: Check whether prompts are not empty.
def test_prompts_not_empty():

    map_messages = map_prompt.format_messages(
        paper_chunk=sample_content
    )

    reduce_messages = reduce_prompt.format_messages(
        section_summaries="Sample research summary."
    )

    for message in map_messages + reduce_messages:
        assert message.content.strip() != ""
