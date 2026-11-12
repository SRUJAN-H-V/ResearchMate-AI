
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def create_chunks(pages):
    """Split extracted PDF pages into smaller text chunks."""

    documents = []

    for page in pages:
        document = Document(
            page_content=page["text"],
            metadata={
                "page": page["page"]
            }
        )

        documents.append(document)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    return chunks
