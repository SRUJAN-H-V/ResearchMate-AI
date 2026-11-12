
import streamlit as st

from core.vector_store import load_existing_vector_store
from utils.pdf_processor import extract_text_from_pdf_bytes
from utils.text_chunker import create_chunks


# --------------------------------------------------
# 1. Configure Streamlit Page
# --------------------------------------------------

st.set_page_config(
    page_title="ResearchMate AI",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# 2. Application Header
# --------------------------------------------------

st.title("📚 ResearchMate AI")

st.subheader("Your AI-powered Research Paper Assistant")

st.write(
    "Upload a research paper to generate summaries, "
    "extract key concepts, prepare presentations, "
    "and practice for viva examinations."
)

st.divider()


# --------------------------------------------------
# 3. PDF Upload Section
# --------------------------------------------------

st.header("Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose a research paper (PDF)",
    type=["pdf"]
)


# --------------------------------------------------
# 4. Process Uploaded PDF
# --------------------------------------------------

if uploaded_file is not None:

    st.success("Research paper uploaded successfully!")

    st.write("**File name:**", uploaded_file.name)

    st.write(
        "**File size:**",
        f"{uploaded_file.size / 1024:.2f} KB"
    )

    try:

        # ------------------------------------------
        # Step 1: Extract PDF Text
        # ------------------------------------------

        pages = extract_text_from_pdf_bytes(
            uploaded_file.getvalue()
        )

        paper_content = "\n\n".join(
            page["text"] for page in pages
        )

        st.divider()

        st.header("Paper Processing Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Pages Extracted",
                len(pages)
            )

        with col2:
            st.metric(
                "Characters Extracted",
                f"{len(paper_content):,}"
            )

        st.success(
            "PDF text extracted successfully!"
        )


        # ------------------------------------------
        # Step 2: Preview Extracted Text
        # ------------------------------------------

        with st.expander("Preview Extracted Text"):

            st.text(paper_content[:5000])

            if len(paper_content) > 5000:

                st.caption(
                    "Showing the first 5,000 characters."
                )


        # ------------------------------------------
        # Step 3: Create Text Chunks
        # ------------------------------------------

        chunks = create_chunks(pages)

        st.divider()

        st.header("Text Chunking Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Total Chunks",
                len(chunks)
            )

        with col2:
            st.metric(
                "Chunk Size",
                "1000 characters"
            )

        st.success(
            "Research paper divided into chunks successfully!"
        )


        # ------------------------------------------
        # Step 4: Preview First Three Chunks
        # ------------------------------------------

        with st.expander("Preview Text Chunks"):

            for index, chunk in enumerate(chunks[:3]):

                st.write(
                    f"**Chunk {index + 1}**"
                )

                st.write(
                    f"Source Page: {chunk.metadata['page']}"
                )

                st.text(
                    chunk.page_content
                )

                st.divider()


        # ------------------------------------------
        # Step 5: Connect to ChromaDB
        # ------------------------------------------

        st.divider()

        st.header("Vector Database")

        if st.button("Connect to ChromaDB"):

            try:

                vector_store = load_existing_vector_store()

                document_count = (
                    vector_store._collection.count()
                )

                st.session_state.vector_store = vector_store

                st.metric(
                    "Stored Chunks",
                    document_count
                )

                if document_count == len(chunks):

                    st.success(
                        "ChromaDB connected successfully! "
                        "Your research paper chunks are available."
                    )

                else:

                    st.warning(
                        f"Database contains {document_count} chunks, "
                        f"but the uploaded paper has {len(chunks)} chunks. "
                        "The database may contain a different paper."
                    )

            except Exception as error:

                st.error(
                    f"ChromaDB connection failed: {error}"
                )


    except Exception as error:

        st.error(
            f"PDF processing failed: {error}"
        )


# --------------------------------------------------
# 6. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "ResearchMate AI | Agentic AI Research Assistant"
)
