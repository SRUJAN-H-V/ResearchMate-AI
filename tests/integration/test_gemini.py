
import time
from core.llm import test_gemini as run_gemini_check


def main():
    start = time.time()

    print("Sending request to Gemini...")

    try:
        result = run_gemini_check()

        print("\nGemini Response:\n")
        print(result)

    except Exception as e:
        print("Error:", e)

    finally:
        print(f"\nTotal time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
