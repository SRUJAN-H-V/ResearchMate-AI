
from langchain_core.prompts import ChatPromptTemplate
from core.llm import llm


# Prompt template for concept extraction
concept_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are ResearchMate AI's Concept Extraction Agent.

Your task is to identify important information
from academic research papers.

Extract the following:

1. Key Concepts
2. Technical Terminology
3. Algorithms and Techniques
4. Research Methodology
5. Tools and Technologies
6. Important Findings
7. Limitations and Challenges

Instructions:
- Present information using clear headings.
- Explain technical terms briefly.
- Preserve the terminology used in the paper.
- Do not invent information.
- If information is unavailable, mention that.
- Use concise bullet points."""
    ),
    (
        "human",
        "Extract the important concepts from this research paper:\n\n{paper_content}"
    )
])


# Create the LCEL chain
concept_chain = concept_prompt | llm


def extract_concepts(paper_content):
    """Extract important concepts from research paper content."""

    response = concept_chain.invoke({
        "paper_content": paper_content
    })

    return response.text
