
from langchain_core.prompts import ChatPromptTemplate

from core.llm import llm, extract_response_text
from core.vector_store import search_documents


# ==================================================
# 1. RAG PROMPT TEMPLATE
# ==================================================

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are ResearchMate AI, an assistant that helps users
understand academic research papers.

Answer the user's question using ONLY the provided
research paper context.

Rules:
1. Give clear and accurate answers.
2. Do not invent information that is not in the context.
3. Mention source page numbers for important information.
4. If the answer is not available in the context,
   clearly state that the research paper does not provide it.
5. Use simple language while preserving technical accuracy.
6. Do not use external knowledge.
7. If multiple pages support an answer, mention all
   relevant page numbers.

Research Paper Context:
{context}"""
    ),
    (
        "human",
        "{question}"
    )
])


# ==================================================
# 2. RAG QUESTION ANSWERING FUNCTION
# ==================================================

def answer_question(vector_store, question):

    # Validate input
    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    if vector_store is None:
        raise ValueError("No research paper is connected to RAG.")

    # --------------------------------------------------
    # STEP 1: RETRIEVE RELEVANT DOCUMENT CHUNKS
    # --------------------------------------------------

    documents = search_documents(
        vector_store,
        question,
        k=3
    )

    # Check whether relevant documents were retrieved
    if not documents:
        return (
            "I could not find relevant information "
            "in the connected research paper to answer this question."
        )

    # --------------------------------------------------
    # STEP 2: PREPARE CONTEXT WITH PAGE NUMBERS
    # --------------------------------------------------

    context_parts = []

    for document in documents:

        page = document.metadata.get("page", "Unknown")

        context_parts.append(
            f"[Source: Page {page}]\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    # --------------------------------------------------
    # STEP 3: CREATE LCEL CHAIN
    # --------------------------------------------------

    rag_chain = rag_prompt | llm

    # --------------------------------------------------
    # STEP 4: GENERATE ANSWER
    # --------------------------------------------------

    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    # Convert Gemini/Ollama response into plain text
    answer = extract_response_text(response)

    return answer
