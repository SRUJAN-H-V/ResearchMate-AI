
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Initialize Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


def create_vector_store(chunks):
    """Convert chunks into embeddings and store them in ChromaDB."""

    ids = [f"chunk-{i}" for i in range(len(chunks))]

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=ids,
        collection_name="research_paper",
        persist_directory="chroma_db"
    )

    return vector_store


def search_documents(vector_store, question, k=4):
    """Retrieve the most relevant chunks for a question."""

    results = vector_store.similarity_search(
        question,
        k=k
    )

    return results


def load_existing_vector_store():
    """Load the existing ChromaDB database."""

    vector_store = Chroma(
        collection_name="research_paper",
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )

    return vector_store
