
import hashlib
import streamlit as st


# ==================================================
# 1. CONFIGURE STREAMLIT PAGE
# ==================================================

st.set_page_config(
    page_title="ResearchMate AI",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

@st.cache_data(show_spinner="Reading and chunking the PDF...")
def process_pdf(pdf_bytes):

    from utils.pdf_processor import extract_text_from_pdf_bytes
    from utils.text_chunker import create_chunks

    pages = extract_text_from_pdf_bytes(pdf_bytes)
    chunks = create_chunks(pages)

    return pages, chunks


def get_pdf_hash(pdf_bytes):
    """Generate a fingerprint based on PDF contents."""

    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def friendly_error(error):
    """Convert common errors into readable messages."""

    message = str(error)
    lower_message = message.lower()

    if "429" in message or "resource_exhausted" in lower_message:
        return "Gemini quota or rate limit reached. Ollama fallback may also have failed."

    if "503" in message or "unavailable" in lower_message:
        return "The AI service is temporarily unavailable. Please try again."

    if "504" in message or "deadline_exceeded" in lower_message:
        return "The AI request timed out. Please try again."

    if "connection refused" in lower_message:
        return "Could not connect to Ollama. Make sure Ollama is running."

    return message


# ==================================================
# 3. INITIALIZE SESSION STATE
# ==================================================

defaults = {
    "vector_store": None,
    "rag_answer": None,
    "paper_summary": None,
    "concept_results": None,
    "viva_results": None,
    "paper_hash": None,
    "index_message": None,
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# 4. APPLICATION HEADER
# ==================================================

st.title("📚 ResearchMate AI")

st.subheader("Your AI-powered Research Paper Assistant")

st.write(
    "Upload a research paper to generate summaries, "
    "extract key concepts, prepare for viva examinations, "
    "and ask questions using AI-powered document retrieval."
)

st.divider()


# ==================================================
# 5. PDF UPLOAD SECTION
# ==================================================

st.header("Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose a research paper (PDF)",
    type=["pdf"]
)


# ==================================================
# 6. PROCESS UPLOADED PDF
# ==================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    pdf_hash = get_pdf_hash(pdf_bytes)

    # Reset previous results when a different PDF is uploaded.
    if st.session_state.paper_hash != pdf_hash:

        st.session_state.vector_store = None
        st.session_state.rag_answer = None
        st.session_state.paper_summary = None
        st.session_state.concept_results = None
        st.session_state.viva_results = None
        st.session_state.index_message = None

        st.session_state.paper_hash = pdf_hash

    st.success("Research paper uploaded successfully!")

    st.write("**File name:**", uploaded_file.name)

    st.write(
        "**File size:**",
        f"{uploaded_file.size / 1024:.2f} KB"
    )

    try:

        # --------------------------------------------------
        # STEP 1: EXTRACT TEXT AND CREATE CHUNKS
        # --------------------------------------------------

        pages, chunks = process_pdf(pdf_bytes)

        paper_content = "\n\n".join(
            page["text"] for page in pages
        )

        st.divider()

        st.header("Paper Processing Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Pages Extracted", len(pages))

        with col2:
            st.metric(
                "Characters Extracted",
                f"{len(paper_content):,}"
            )

        with col3:
            st.metric("Total Chunks", len(chunks))

        st.success(
            "PDF text extracted and chunked successfully!"
        )

        # --------------------------------------------------
        # STEP 2: PREVIEW EXTRACTED TEXT AND CHUNKS
        # --------------------------------------------------

        with st.expander("Preview Extracted Text"):

            st.text(paper_content[:5000])

            if len(paper_content) > 5000:
                st.caption(
                    "Showing the first 5,000 characters."
                )

        with st.expander("Preview Text Chunks"):

            for index, chunk in enumerate(chunks[:3]):

                st.write(f"**Chunk {index + 1}**")

                st.write(
                    f"Source Page: {chunk.metadata.get('page', 'Unknown')}"
                )

                st.text(chunk.page_content)

                st.divider()

        # --------------------------------------------------
        # STEP 3: INDEX PAPER IN CHROMADB
        # --------------------------------------------------

        st.divider()

        st.header("Vector Database")

        st.write(
            "ResearchMate AI uses local Ollama embeddings "
            "to store and retrieve research paper sections "
            "in ChromaDB. Each PDF has its own collection."
        )

        if st.button("Index Paper / Connect to ChromaDB"):

            with st.spinner(
                "Creating local embeddings and connecting to ChromaDB..."
            ):

                try:

                    from core.vector_store import (
                        get_or_create_vector_store
                    )

                    vector_store, newly_indexed = (
                        get_or_create_vector_store(
                            chunks,
                            pdf_hash
                        )
                    )

                    st.session_state.vector_store = vector_store

                    if newly_indexed:

                        st.session_state.index_message = (
                            f"Successfully indexed {len(chunks)} chunks "
                            "using local Ollama embeddings."
                        )

                    else:

                        st.session_state.index_message = (
                            f"Reused the existing index containing "
                            f"{len(chunks)} chunks. "
                            "No new embeddings were required."
                        )

                except Exception as error:

                    st.session_state.vector_store = None

                    st.error(
                        "Indexing failed: "
                        f"{friendly_error(error)}"
                    )

        if (
            st.session_state.vector_store is not None
            and st.session_state.index_message
        ):

            st.success(st.session_state.index_message)

        # --------------------------------------------------
        # STEP 4: RESEARCH PAPER SUMMARIZER
        # --------------------------------------------------

        st.divider()

        st.header("Research Paper Summary")

        st.write(
            "Generate a structured summary of your "
            "uploaded research paper."
        )

        if st.button("Generate Research Paper Summary"):

            with st.spinner(
                "Summarizing the research paper..."
            ):

                try:

                    from agents.summarizer import summarize_paper

                    summary = summarize_paper(paper_content)

                    st.session_state.paper_summary = summary

                except Exception as error:

                    st.error(
                        "Unable to generate summary: "
                        f"{friendly_error(error)}"
                    )

        if st.session_state.paper_summary:

            st.subheader("Generated Summary")

            st.markdown(
                st.session_state.paper_summary
            )

        # --------------------------------------------------
        # STEP 5: CONCEPT EXTRACTION
        # --------------------------------------------------

        st.divider()

        st.header("Key Concept Extraction")

        st.write(
            "Extract important concepts, technical terms, "
            "algorithms, methodology and findings."
        )

        if st.button("Extract Key Concepts"):

            with st.spinner(
                "Extracting concepts from the research paper..."
            ):

                try:

                    from agents.concept_extractor import extract_concepts

                    concepts = extract_concepts(paper_content)

                    st.session_state.concept_results = concepts

                except Exception as error:

                    st.error(
                        "Unable to extract concepts: "
                        f"{friendly_error(error)}"
                    )

        if st.session_state.concept_results:

            st.subheader("Extracted Concepts")

            st.markdown(
                st.session_state.concept_results
            )

        # --------------------------------------------------
        # STEP 6: VIVA PREPARATION
        # --------------------------------------------------

        st.divider()

        st.header("Viva Preparation")

        st.write(
            "Generate easy, medium and difficult viva "
            "questions with answers based on your research paper."
        )

        if st.button("Generate Viva Questions"):

            with st.spinner(
                "Preparing viva questions and answers..."
            ):

                try:

                    from agents.viva_agent import generate_viva_questions

                    viva = generate_viva_questions(paper_content)

                    st.session_state.viva_results = viva

                except Exception as error:

                    st.error(
                        "Unable to generate viva questions: "
                        f"{friendly_error(error)}"
                    )

        if st.session_state.viva_results:

            st.subheader("Generated Viva Questions")

            st.markdown(
                st.session_state.viva_results
            )

        # --------------------------------------------------
        # STEP 7: CHAT WITH RESEARCH PAPER (RAG)
        # --------------------------------------------------

        st.divider()

        st.header("Chat with Research Paper")

        st.write(
            "Ask questions about your uploaded research paper. "
            "ResearchMate AI retrieves relevant sections from "
            "ChromaDB and generates an answer."
        )

        if st.session_state.vector_store is not None:

            with st.form("research_question_form"):

                user_question = st.text_input(
                    "Enter your question",
                    placeholder="What is the main objective of this paper?"
                )

                submit_question = st.form_submit_button(
                    "Ask ResearchMate AI"
                )

            if submit_question:

                if user_question.strip():

                    with st.spinner(
                        "Searching the research paper and generating an answer..."
                    ):

                        try:

                            from core.rag_chain import answer_question

                            response = answer_question(
                                st.session_state.vector_store,
                                user_question
                            )

                            st.session_state.rag_answer = response

                        except Exception as error:

                            st.error(
                                "Unable to generate answer: "
                                f"{friendly_error(error)}"
                            )

                else:

                    st.warning("Please enter a question.")

            if st.session_state.rag_answer:

                st.subheader("Answer")

                st.markdown(
                    st.session_state.rag_answer
                )

        else:

            st.info(
                "Please click 'Index Paper / Connect to ChromaDB' "
                "before using the research paper chat."
            )

    except Exception as error:

        st.error(
            f"PDF processing failed: {friendly_error(error)}"
        )

else:

    st.info("Upload a research paper PDF to get started.")


# ==================================================
# 7. FOOTER
# ==================================================

st.divider()

st.caption(
    "ResearchMate AI | Agentic AI Research Assistant"
)
