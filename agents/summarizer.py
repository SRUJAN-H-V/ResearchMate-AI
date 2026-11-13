
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import llm, extract_response_text


# ==================================================
# 1. MAP PROMPT: SUMMARIZE INDIVIDUAL PAPER SECTIONS
# ==================================================

map_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are ResearchMate AI's Research Paper Summarizer.

Your task is to summarize the provided section of an
academic research paper.

Extract:
1. Research problem
2. Objectives
3. Methodology
4. Important concepts and techniques
5. Key findings
6. Limitations

Instructions:
- Preserve important technical terms.
- Use clear and simple academic language.
- Include only information available in the section.
- Do not invent information.
- Keep the summary concise.
- Mention page numbers if provided."""
    ),
    (
        "human",
        "Summarize this research paper section:\n\n{paper_chunk}"
    )
])

map_chain = map_prompt | llm


# ==================================================
# 2. REDUCE PROMPT: GENERATE FINAL STRUCTURED SUMMARY
# ==================================================

reduce_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are ResearchMate AI's Final Summary Agent.

You will receive summaries of different sections of
the same research paper.

Combine them into one structured research paper summary.

Use these headings:

1. Research Problem
2. Objectives
3. Proposed Methodology
4. Key Findings
5. Conclusion
6. Limitations

Instructions:
- Combine related information.
- Remove repeated points.
- Preserve important technical details.
- Do not introduce information that is not present
  in the provided section summaries.
- If information is unavailable, clearly mention it.
- Use simple academic language.
- Do not confuse findings with the conclusion."""
    ),
    (
        "human",
        "Combine the following section summaries:\n\n{section_summaries}"
    )
])

reduce_chain = reduce_prompt | llm


# ==================================================
# 3. MAP-REDUCE SUMMARIZATION FUNCTION
# ==================================================

def summarize_paper(paper_content):

    # Validate input
    if not isinstance(paper_content, str) or not paper_content.strip():
        raise ValueError("No research paper content provided.")

    # Divide the research paper into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(paper_content)

    if not chunks:
        raise ValueError("No readable text chunks found.")

    section_summaries = []

    # --------------------------------------------------
    # MAP: Summarize each chunk individually
    # --------------------------------------------------

    for index, chunk in enumerate(chunks):

        print(
            f"Summarizing section {index + 1} "
            f"of {len(chunks)}..."
        )

        response = map_chain.invoke({
            "paper_chunk": chunk
        })

        # Convert Gemini/Ollama response into plain text
        summary_text = extract_response_text(response)

        section_summaries.append(summary_text)

    # --------------------------------------------------
    # REDUCE: Combine all intermediate summaries
    # --------------------------------------------------

    combined_summaries = "\n\n".join(section_summaries)

    print("Generating final structured summary...")

    final_response = reduce_chain.invoke({
        "section_summaries": combined_summaries
    })

    # Return clean text instead of response objects
    final_summary = extract_response_text(final_response)

    return final_summary
