import time
from core.llm import test_gemini

start = time.time()

print("Sending request to Gemini...")

try:
    result = test_gemini()

    print("\nGemini Response:\n")
    print(result)

except Exception as e:
    print("Error:", e)

finally:
    print(f"\nTotal time: {time.time() - start:.2f} seconds")