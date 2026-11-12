
from utils.pdf_processor import extract_text_from_pdf
from utils.text_chunker import create_chunks

pages = extract_text_from_pdf("sample_paper.pdf")

chunks = create_chunks(pages)

print("Chunking successful!")
print("Total pages:", len(pages))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {i} ---")
    print("Page:", chunk.metadata["page"])
    print(chunk.page_content[:300])
