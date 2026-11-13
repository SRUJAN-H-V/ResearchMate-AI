
from agents.concept_extractor import extract_concepts
from core.llm import clear_used_providers, get_used_providers

sample_content = """
The research paper investigates the use of machine learning
for detecting fraudulent financial transactions.

The proposed methodology uses a Random Forest algorithm
trained on historical transaction data.

The model analyzes transaction patterns and identifies
suspicious activities.

The findings indicate that machine learning can improve
fraud detection accuracy and reduce manual investigation.
"""

try:
    clear_used_providers()

    print("Starting concept extraction test...\n")

    result = extract_concepts(sample_content)

    print("\n========== EXTRACTED CONCEPTS ==========\n")
    print(result)

    print("\n========== MODELS USED ==========")
    print(get_used_providers())

    print("\nConcept extraction test completed!")

except Exception as error:
    print("\nConcept extraction test failed!")
    print("Error:", error)
