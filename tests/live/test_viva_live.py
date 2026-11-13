
from agents.viva_agent import generate_viva_questions
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

    print("Starting viva question generation test...\n")

    result = generate_viva_questions(sample_content)

    print("\n========== VIVA QUESTIONS AND ANSWERS ==========\n")
    print(result)

    print("\n========== MODELS USED ==========")
    print(get_used_providers())

    print("\nViva generation test completed!")

except Exception as error:
    print("\nViva generation test failed!")
    print("Error:", error)
