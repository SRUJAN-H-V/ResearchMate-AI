
from langchain_core.prompts import ChatPromptTemplate

from core.llm import llm, extract_response_text


# ==================================================
# 1. VIVA GENERATION PROMPT
# ==================================================

viva_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are ResearchMate AI's Viva Preparation Agent.

Your task is to prepare students for viva examinations
based on academic research papers.

Generate questions and answers at three difficulty levels:

1. Easy - Basic understanding and definitions.
2. Medium - Methodology, concepts and applications.
3. Difficult - Critical analysis, limitations and reasoning.

For each question:
- Provide a clear and accurate answer.
- Keep answers concise but informative.
- Base answers only on the provided research paper.
- Do not invent information.
- Mention when the paper does not provide an answer.

Generate exactly 5 questions for each difficulty level.

Use clear headings and number every question."""
    ),
    (
        "human",
        "Prepare viva questions and answers from this research paper:\n\n{paper_content}"
    )
])


# ==================================================
# 2. VIVA GENERATION CHAIN
# ==================================================

viva_chain = viva_prompt | llm


# ==================================================
# 3. VIVA QUESTION GENERATION FUNCTION
# ==================================================

def generate_viva_questions(paper_content):

    if not isinstance(paper_content, str) or not paper_content.strip():
        raise ValueError("No research paper content provided.")

    response = viva_chain.invoke({
        "paper_content": paper_content
    })

    return extract_response_text(response)
