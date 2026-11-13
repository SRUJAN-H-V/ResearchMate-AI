from agents.summarizer import summarize_paper
from core.llm import (
    get_used_providers,
    clear_used_providers
)

sample_content = """
The research paper investigates the perspectives of
faculty members and students regarding artificial
intelligence in higher education.

Students commonly use AI for summarization,
idea generation and writing assistance.

Faculty members express concerns about academic
integrity, plagiarism and critical thinking.

The paper highlights the need for institutional
policies and AI awareness programs.
"""

try:
    clear_used_providers()

    print("Starting live summarization test...\n")

    summary = summarize_paper(sample_content)

    print("\n========== FINAL SUMMARY ==========\n")
    print(summary)

    print("\n========== MODELS USED ==========")
    print(get_used_providers())

    print("\nSummarization test completed!")

except Exception as error:
    print("\nSummarization test failed!")
    print("Error:", error)