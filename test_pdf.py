from utils.pdf_processor import extract_text_from_pdf

pdf_path = "sample_paper.pdf"

try:
    pages = extract_text_from_pdf(pdf_path)

    print("PDF extracted successfully!")
    print("Total pages:", len(pages))

    for page in pages[:2]:
        print(f"\n--- Page {page['page']} ---")
        print(page["text"][:1000])

except Exception as e:
    print("Error:", e)