from core.llm import test_hybrid as run_hybrid_check


def main():
    result = run_hybrid_check()

    print("\nModel Used:", result["provider"])
    print("\nResponse:")
    print(result["response"])


if __name__ == "__main__":
    main()