
from langchain_core.documents import Document

from core.vector_store import (
    create_vector_store,
    search_documents
)

sample_documents = [
    Document(
        page_content="The research uses Random Forest "
                     "for detecting fraudulent transactions.",
        metadata={"page": 1}
    ),
    Document(
        page_content="The proposed system analyzes "
                     "historical financial transaction patterns.",
        metadata={"page": 2}
    ),
    Document(
        page_content="The findings show improved fraud "
                     "detection accuracy.",
        metadata={"page": 3}
    )
]

try:
    print("Starting local vector store test...\n")

    vector_store = create_vector_store(
        sample_documents,
        collection_name="local_embedding_test"
    )

    print("Documents successfully stored!")

    results = search_documents(
        vector_store,
        "Which algorithm is used for fraud detection?",
        k=2
    )

    print("\n========== SEARCH RESULTS ==========\n")

    for document in results:
        print("Page:", document.metadata.get("page"))
        print("Content:", document.page_content)
        print("-" * 50)

    print("\nLocal vector store test completed!")

except Exception as error:
    print("\nVector store test failed!")
    print("Error:", error)
