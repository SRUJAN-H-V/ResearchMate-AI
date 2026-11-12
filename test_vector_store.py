
from utils.pdf_processor import extract_text_from_pdf
from utils.text_chunker import create_chunks
from core.vector_store import create_vector_store, search_documents

# Step 1: Extract PDF text
pages = extract_text_from_pdf("sample_paper.pdf")

# Step 2: Create chunks
chunks = create_chunks(pages)

print("Total chunks:", len(chunks))

# Step 3: Store chunks in ChromaDB
print("\nCreating vector database...")

vector_store = create_vector_store(chunks)

print("Vector database created successfully!")

# Step 4: Search for relevant information
question = "What are the perspectives of faculty and students on artificial intelligence in higher education?"

print("\nSearching for:", question)

results = search_documents(vector_store, question)

# Step 5: Display retrieved results
for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Source page:", result.metadata["page"])
    print(result.page_content[:500])
