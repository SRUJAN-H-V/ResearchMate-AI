
from langchain_core.prompts import ChatPromptTemplate
from core.llm import llm


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

Example:

Question: What is the main objective of a research paper?
Answer: To investigate a specific problem or research question
and present findings supported by evidence.

Now generate viva questions using the provided paper content.

Generate 5 questions for each difficulty level."""
    ),

    (
        "human",
        "Prepare viva questions and answers from this research paper:\n\n{paper_content}"
    )

])


# Create the LCEL chain
viva_chain = viva_prompt | llm


def generate_viva_questions(paper_content):
    """Generate viva questions and answers."""

    response = viva_chain.invoke({
        "paper_content": paper_content
    })

    return response.text

