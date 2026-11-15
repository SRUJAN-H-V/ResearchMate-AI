
import time

from core.llm import test_gemini as run_gemini_check


def test_gemini_connection():

    start = time.perf_counter()

    print("\nSending request to Gemini...")

    # Make a real Gemini API request
    result = run_gemini_check()

    elapsed_time = time.perf_counter() - start

    print("\nGemini Response:\n")
    print(result)

    print(f"\nTotal time: {elapsed_time:.2f} seconds")

    # Verify that Gemini returned a valid response
    assert isinstance(result, str)
    assert len(result.strip()) > 0
