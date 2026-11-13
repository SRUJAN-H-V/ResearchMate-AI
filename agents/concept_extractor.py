
from langchain_core.prompts import ChatPromptTemplate

from core.llm import llm, extract_response_text


# ==================================================
# 1. CONCEPT EXTRACTION PROMPT
# ==================================================

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
- Use concise bullet points.
- Base your response only on the provided paper content."""
    ),
    (
        "human",
        "Extract the important concepts from this research paper:\n\n{paper_content}"
    )
])


# ==================================================
# 2. CONCEPT EXTRACTION CHAIN
# ==================================================

concept_chain = concept_prompt | llm


# ==================================================
# 3. CONCEPT EXTRACTION FUNCTION
# ==================================================

def extract_concepts(paper_content):

    if not isinstance(paper_content, str) or not paper_content.strip():
        raise ValueError("No research paper content provided.")

    response = concept_chain.invoke({
        "paper_content": paper_content
    })

    return extract_response_text(response)
