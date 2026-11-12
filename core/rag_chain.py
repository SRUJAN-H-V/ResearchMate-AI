
from langchain_core.prompts import ChatPromptTemplate
from core.llm import llm
from core.vector_store import search_documents


# Define the prompt template
rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are ResearchMate AI, an assistant that helps users
understand research papers.

Answer the user's question using ONLY the provided research paper context.

Rules:
1. Give clear and accurate answers.
2. Do not invent information that is not in the context.
3. Mention the source page number for important information.
4. If the answer is not available in the context, clearly say so.
5. Use simple language while preserving technical accuracy.

Research Paper Context:
{context}"""
    ),
    (
        "human",
        "{question}"
    )
])


def answer_question(vector_store, question):
    """Retrieve relevant chunks and generate an answer using Gemini."""

    # Step 1: Retrieve relevant chunks
    documents = search_documents(
        vector_store,
        question,
        k=3
    )

    # Step 2: Prepare context with page numbers
    context = ""

    for document in documents:
        page = document.metadata.get("page", "Unknown")

        context += (
            f"\n[Source: Page {page}]\n"
            f"{document.page_content}\n"
        )

    # Step 3: Create LCEL chain
    rag_chain = rag_prompt | llm

    # Step 4: Generate answer
    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    return response.text
