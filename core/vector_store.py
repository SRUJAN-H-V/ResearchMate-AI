
import os

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

# Separate database for local Ollama embeddings.
# Do not reuse the old Gemini embedding database.
PERSIST_DIRECTORY = "chroma_db_ollama"

EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text"
)


# ==================================================
# 2. INITIALIZE LOCAL EMBEDDINGS
# ==================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# ==================================================
# 3. COLLECTION NAME GENERATOR
# ==================================================

def collection_name_for(pdf_hash):
    """
    Generate a unique ChromaDB collection name
    for each research paper.
    """

    if not pdf_hash:
        raise ValueError("PDF hash is required.")

    # Keep the collection name within ChromaDB limits.
    return f"paper_{pdf_hash[:32]}"


# ==================================================
# 4. CREATE VECTOR STORE
# ==================================================

def create_vector_store(chunks, collection_name="research_paper"):
    """
    Convert document chunks into local embeddings
    and store them in ChromaDB.
    """

    if not chunks:
        raise ValueError("No document chunks provided.")

    ids = [
        f"chunk-{i}"
        for i in range(len(chunks))
    ]

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=ids,
        collection_name=collection_name,
        persist_directory=PERSIST_DIRECTORY
    )

    return vector_store


# ==================================================
# 5. LOAD EXISTING VECTOR STORE
# ==================================================

def load_existing_vector_store(collection_name="research_paper"):
    """
    Load an existing ChromaDB collection using
    the same local embedding model.
    """

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    return vector_store


# ==================================================
# 6. SEARCH DOCUMENTS
# ==================================================

def search_documents(vector_store, question, k=4):
    """
    Retrieve the most relevant document chunks
    using semantic similarity.
    """

    if vector_store is None:
        raise ValueError("Vector store is not initialized.")

    if not question or not question.strip():
        raise ValueError("Search question cannot be empty.")

    results = vector_store.similarity_search(
        question,
        k=k
    )

    return results


# ==================================================
# 7. GET OR CREATE VECTOR STORE
# ==================================================

def get_or_create_vector_store(chunks, pdf_hash):
    """
    Return (vector_store, was_newly_indexed).

    If the PDF has already been indexed and the
    stored chunk count matches, reuse the collection.

    Otherwise, create a fresh local embedding index.
    """

    if not chunks:
        raise ValueError("No document chunks provided.")

    name = collection_name_for(pdf_hash)

    # Load the collection.
    vector_store = load_existing_vector_store(name)

    # Check how many chunks are already stored.
    stored_data = vector_store.get()
    stored_ids = stored_data.get("ids", [])

    stored_count = len(stored_ids)

    # Reuse the collection if it is complete.
    if stored_count == len(chunks):

        print("Existing document index found. Reusing it.")

        return vector_store, False

    # Remove an incomplete collection.
    if stored_count > 0:

        print("Incomplete index found. Rebuilding it.")

        vector_store.delete_collection()

    # Create a new collection.
    print("Creating local Ollama embeddings...")

    vector_store = create_vector_store(
        chunks,
        collection_name=name
    )

    print("Document successfully indexed in ChromaDB.")

    return vector_store, True
