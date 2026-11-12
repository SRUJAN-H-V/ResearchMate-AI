
from utils.pdf_processor import extract_text_from_pdf
from utils.text_chunker import create_chunks
from core.vector_store import create_vector_store
from core.rag_chain import answer_question


# Step 1: Extract PDF
pages = extract_text_from_pdf("sample_paper.pdf")

# Step 2: Create chunks
chunks = create_chunks(pages)

# Step 3: Create vector database
vector_store = create_vector_store(chunks)

# Step 4: Ask a question
question = "How do faculty and students differ in their attitudes toward AI?"

print("Question:", question)
print("\nGenerating answer...\n")

# Step 5: Generate RAG-based answer
answer = answer_question(vector_store, question)

print("ResearchMate AI Answer:\n")
print(answer)
