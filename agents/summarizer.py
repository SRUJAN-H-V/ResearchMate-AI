
from langchain_core.prompts import ChatPromptTemplate
from core.llm import llm


# Prompt template for research paper summarization
summarizer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are ResearchMate AI's Summarizer Agent.

Your task is to summarize academic research papers.

Organize the summary using these headings:

1. Research Problem
2. Objectives
3. Proposed Methodology
4. Key Findings
5. Conclusion
6. Limitations

Instructions:
- Use clear and simple academic language.
- Preserve important technical terms.
- Do not invent information.
- If information is unavailable, mention that.
- Keep the summary concise and informative."""
    ),
    (
        "human",
        "Summarize the following research paper content:\n\n{paper_content}"
    )
])


# Create the LangChain LCEL chain
summarizer_chain = summarizer_prompt | llm


def summarize_paper(paper_content):
    """Generate a structured research paper summary."""

    response = summarizer_chain.invoke({
        "paper_content": paper_content
    })

    return response.text
